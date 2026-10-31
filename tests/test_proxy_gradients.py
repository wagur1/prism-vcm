import pytest

torch = pytest.importorskip("torch")

from prism.models.proxy_codec import ProxyCodec
from prism.losses.distillation_loss import DistillationLoss


def test_ste_cannot_route_discrete_gradient_to_backbone():
    proxy = ProxyCodec()
    feature = torch.rand(2, 1, 8, 8, requires_grad=True)
    dqp = torch.zeros(2, 1, 2, 2, requires_grad=True)
    result = proxy(feature, dqp, torch.tensor([1, 4]), torch.tensor([27., 37.]))
    result["reconstruction"].square().mean().backward()
    assert feature.grad is None and dqp.grad is None
    assert proxy.analysis.weight.grad is not None
    assert torch.isfinite(proxy.analysis.weight.grad).all()


def test_rate_direction_and_frame_conditioning():
    proxy = ProxyCodec()
    feature = torch.ones(2, 1, 8, 8, requires_grad=True)
    dqp = torch.zeros(2, 1, 2, 2, requires_grad=True)
    low = proxy.R_hat(feature, dqp, torch.tensor([1, 4]), torch.tensor([27., 27.]))
    high = proxy.R_hat(feature, dqp, torch.tensor([1, 4]), torch.tensor([37., 37.]))
    assert torch.all(high < low) and low[0] != low[1]
    low.sum().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    assert dqp.grad is None


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
