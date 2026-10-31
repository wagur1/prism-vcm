import pytest

torch = pytest.importorskip("torch")

from prism.models.student import MobileNetV4HybridController, upsample_qp_map


def test_student_heads_and_qp_conditioning():
    torch.manual_seed(3)
    model = MobileNetV4HybridController(backbone="small").eval()
    x = torch.randn(2, 3, 256, 256)
    qp = torch.tensor([[27 / 51], [47 / 51]])
    heads = model(x, qp)
    assert heads["delta_qp"].shape == (2, 1, 17, 30)
    assert heads["residual"].shape == (2, 1, 256, 256)
    assert heads["delta_qp"].abs().max() <= 6
    assert heads["residual"].abs().max() <= 8
    encoded = model(x, qp, for_encoding=True)["delta_qp"]
    assert torch.equal(encoded, encoded.round())
    assert not torch.allclose(model(x[:1], qp[:1])["delta_qp"], model(x[:1], qp[1:])["delta_qp"])


def test_timm_backbone_without_download():
    pytest.importorskip("timm")
    model = MobileNetV4HybridController(backbone="timm").eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 256, 256), torch.ones(1, 1) * 32 / 51)["delta_qp"].shape == (1, 1, 17, 30)


def test_ctu_to_qg_centers_and_padded_edge():
    source = torch.arange(30, dtype=torch.float32)[None, None, None, :].expand(1, 1, 17, 30)
    grid = upsample_qp_map(source)
    assert grid.shape == (1, 1, 68, 120)
    expected = ((torch.arange(120) + 0.5) / 4 - 0.5).clamp(0, 29)
    torch.testing.assert_close(grid[0, 0, 0], expected)
    torch.testing.assert_close(grid[0, 0, -1], expected)
    ones = torch.ones(1, 1, 17, 30) * 6
    torch.testing.assert_close(upsample_qp_map(ones), torch.ones_like(grid) * 6)


@pytest.mark.provisional
def test_latency_budget_requires_t4_pilot():
    # TODO [CẦN XÁC MINH]: T4 TRT FP16, 200 warmup, 3 x 10000 full-pipeline samples.
    pytest.skip("Provisional p50<=5/p99<=10 ms budget; no T4 timing evidence in CPU suite")
