import pytest

torch = pytest.importorskip("torch")

from prism.models.proxy_codec import ProxyCodec
from prism.losses.distillation_loss import DistillationLoss


def test_ste_cannot_route_discrete_gradient_to_backbone():
    # Regression: this test used to assert feature.grad is None, preserving a bug.
    proxy = ProxyCodec().requires_grad_(False)
    feature = (80.0 + 100.0 * torch.rand(2, 1, 8, 8)).requires_grad_()
    dqp = torch.zeros(2, 1, 2, 2, requires_grad=True)
    result = proxy(feature, dqp, torch.tensor([1, 4]), torch.tensor([27., 37.]))
    result["reconstruction"].square().mean().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    assert feature.grad.abs().sum() > 0
    assert dqp.grad is not None and torch.isfinite(dqp.grad).all()
    assert dqp.grad.abs().sum() > 0


def test_rate_direction_and_frame_conditioning():
    proxy = ProxyCodec()
    feature = torch.arange(64.0).reshape(1, 1, 8, 8).repeat(2, 1, 1, 1).requires_grad_()
    dqp = torch.zeros(2, 1, 2, 2, requires_grad=True)
    low = proxy.R_hat(feature, dqp, torch.tensor([1, 4]), torch.tensor([27., 27.]))
    high = proxy.R_hat(feature, dqp, torch.tensor([1, 4]), torch.tensor([37., 37.]))
    assert torch.all(high < low) and low[0] != low[1]
    low.sum().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    assert feature.grad.abs().sum() > 0
    assert dqp.grad is not None and torch.isfinite(dqp.grad).all()
    assert torch.all(dqp.grad < 0)


def test_semantic_gradient_reaches_only_continuous_branch():
    raw = torch.ones(1, 3, 4, 4)
    dx = torch.zeros(1, 1, 4, 4, requires_grad=True)
    dq = torch.zeros(1, 1, 17, 30, requires_grad=True)
    projection = torch.randn(48, 768)
    losses = DistillationLoss()(x_raw=raw, residual=dx, delta_qp=dq,
        oracle_qp=torch.ones_like(dq), dino_target=torch.randn(1, 768),
        semantic_encoder=lambda x: x.flatten(1) @ projection, rate=torch.zeros(1))
    losses["semantic"].backward(retain_graph=True)
    assert dx.grad is not None and torch.isfinite(dx.grad).all()
    assert dq.grad is None
    losses["align"].backward()
    assert torch.all(dq.grad < 0)


def test_dct_is_fixed_orthogonal_and_matches_reference():
    from scipy.fft import dctn

    proxy = ProxyCodec()
    torch.testing.assert_close(proxy.dct_basis @ proxy.dct_basis.T, torch.eye(8))
    assert "dct_basis" in dict(proxy.named_buffers())
    assert not proxy.dct_basis.requires_grad
    image = torch.arange(2 * 16 * 24.0).reshape(2, 1, 16, 24) % 251
    blocks = (image - 128).unfold(2, 8, 8).unfold(3, 8, 8)
    expected = torch.from_numpy(dctn(blocks.numpy(), axes=(-2, -1), norm="ortho"))
    torch.testing.assert_close(proxy._transform(image), expected, atol=1e-4, rtol=1e-5)


def test_frequency_weighted_dead_zone_and_inverse_dct():
    proxy = ProxyCodec()
    basis = proxy.dct_basis
    # At QP=22 the step is 8. The low AC survives; equal high AC is in its zero bin.
    low = 128 + 8 * torch.outer(basis[1], basis[0])
    high = 128 + 8 * torch.outer(basis[7], basis[7])
    image = torch.stack([low, high])[:, None].requires_grad_()
    result = proxy(image, torch.zeros(2, 1, 1, 1), torch.ones(2).long(),
                   torch.full((2,), 22.0))
    torch.testing.assert_close(result["reconstruction"][0, 0], low)
    torch.testing.assert_close(result["reconstruction"][1, 0], torch.full((8, 8), 128.0))
    # A coefficient killed by the dead zone must still receive a surrogate slope.
    (result["reconstruction"][1, 0] * (high - 128)).sum().backward()
    assert torch.isfinite(image.grad).all() and image.grad[1].abs().sum() > 0


@pytest.mark.parametrize("shape", [(1, 1, 1, 1), (2, 3, 9, 19), (1, 1, 17, 8)])
def test_padding_channels_and_rate_api_agree(shape):
    proxy = ProxyCodec(channels=shape[1])
    image = torch.full(shape, 100.0, requires_grad=True)
    dqp = torch.zeros(shape[0], 1, 2, 3)
    frame = torch.ones(shape[0], dtype=torch.long)
    qp = torch.full((shape[0],), 28.0)
    result = proxy(image, dqp, frame, qp)
    assert result["reconstruction"].shape == shape
    assert result["rate"].shape == (shape[0],)
    torch.testing.assert_close(result["rate"], proxy.R_hat(image, dqp, frame, qp))
    result["reconstruction"].mean().backward()
    assert torch.isfinite(image.grad).all() and torch.all(image.grad > 0)


def test_rate_preserves_brightness_and_detects_ac_and_dc_activity():
    proxy = ProxyCodec()
    flat = torch.full((1, 1, 16, 16), 80.0)
    edge = flat.clone()
    edge[..., 8:] += 40  # Constant within each block: only inter-block DC changes.
    texture = flat + 10 * ((torch.arange(16)[:, None] + torch.arange(16)) % 2)
    images = torch.cat([flat, flat + 50, edge, texture]).requires_grad_()
    rate = proxy.R_hat(images, torch.zeros(4, 1, 2, 2), torch.ones(4).long(),
                       torch.full((4,), 28.0))
    torch.testing.assert_close(rate[0], rate[1])
    assert torch.all(rate[2:] > rate[:2])
    rate.sum().backward()
    torch.testing.assert_close(images.grad[:2], torch.zeros_like(images.grad[:2]))
    assert images.grad[2:].abs().sum() > 0
    # Brightness shifts have zero derivative, even for textured/stepped images.
    torch.testing.assert_close(images.grad.flatten(1).sum(1), torch.zeros(4), atol=1e-5, rtol=0)


def test_rate_gradients_match_finite_differences():
    proxy = ProxyCodec().double().requires_grad_(False)
    generator = torch.Generator().manual_seed(13)
    image = (80 + 100 * torch.rand(1, 1, 8, 16, generator=generator,
                                 dtype=torch.float64)).requires_grad_()
    dqp = torch.zeros(1, 1, 1, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x, q: proxy.R_hat(x, q, torch.tensor([1]), torch.tensor([28.0])),
        (image, dqp))


def test_effective_qp_clamping():
    proxy = ProxyCodec()
    image = torch.full((4, 1, 8, 8), 100.0)
    dqp = torch.tensor([-100.0, -28.0, 23.0, 100.0]).reshape(4, 1, 1, 1).requires_grad_()
    result = proxy(image, dqp, torch.ones(4).long(), torch.full((4,), 28.0))
    torch.testing.assert_close(result["reconstruction"][0], result["reconstruction"][1])
    torch.testing.assert_close(result["reconstruction"][2], result["reconstruction"][3])
    torch.testing.assert_close(result["rate"][0], result["rate"][1])
    torch.testing.assert_close(result["rate"][2], result["rate"][3])
    result["rate"].sum().backward()
    assert dqp.grad[0].item() == dqp.grad[3].item() == 0.0


def test_affine_rate_calibration_is_per_qp_and_frame():
    proxy = ProxyCodec()
    image = (80 + torch.arange(64.0).reshape(1, 1, 8, 8)).repeat(3, 1, 1, 1)
    dqp, frame, qp = torch.zeros(3, 1, 1, 1), torch.tensor([1, 1, 4]), torch.tensor([24., 36., 24.])
    before = proxy.R_hat(image, dqp, frame, qp).detach()
    old_bias = torch.nn.functional.softplus(proxy.rate_log_bias[24, 1]).detach()
    with torch.no_grad():
        proxy.rate_log_scale[24, 1] += torch.tensor(2.0).log()
        proxy.rate_log_bias[24, 1] = torch.expm1(torch.tensor(3.0)).log()
    after = proxy.R_hat(image, dqp, frame, qp)
    torch.testing.assert_close(after[0], 2 * (before[0] - old_bias) + 3)
    torch.testing.assert_close(after[1:], before[1:])
    after.sum().backward()
    for parameter in proxy.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("qp", [24.0, 28.0, 36.0])
@pytest.mark.parametrize("compose_encoder", [False, True])
def test_post_codec_semantic_gradient_at_zero_residual(qp, compose_encoder):
    generator = torch.Generator().manual_seed(13)
    raw = 80 + 100 * torch.rand(1, 3, 16, 16, generator=generator)
    dx = torch.zeros(1, 1, 16, 16, requires_grad=True)
    dq = torch.zeros(1, 1, 2, 2, requires_grad=True)
    proxy = ProxyCodec(channels=3).requires_grad_(False)
    encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(768, 768, bias=False))
    with torch.no_grad():
        encoder[1].weight.copy_(torch.eye(768))
    encoder.requires_grad_(False)
    target = encoder(raw).detach().requires_grad_()
    encode = lambda x: proxy(x, dq, torch.tensor([1]), torch.tensor([qp]))["reconstruction"]
    arguments = {"semantic_encoder": lambda x: encoder(encode(x))} if compose_encoder else {
        "semantic_encoder": encoder, "x_reconstructed": encode(raw + dx)}
    losses = DistillationLoss()(x_raw=raw, residual=dx, delta_qp=dq,
        oracle_qp=torch.zeros_like(dq), dino_target=target, rate=torch.zeros(1), **arguments)
    assert losses["semantic"].item() > 1e-6
    # Isolate semantic loss so neither rate nor regularization can hide a detach.
    losses["semantic"].backward()
    assert dx.grad is not None and torch.isfinite(dx.grad).all()
    assert dx.grad.norm().item() > 1e-8
    assert target.grad is None and all(p.grad is None for p in encoder.parameters())


def test_autocast_keeps_pixel_domain_rate_and_gradients_finite():
    proxy = ProxyCodec()
    image = torch.full((1, 1, 9, 9), 200.0, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = proxy(image, torch.zeros(1, 1, 1, 1), torch.tensor([1]), torch.tensor([28.0]))
        loss = result["reconstruction"].mean() + result["rate"].mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(image.grad).all()
