import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from prism.evaluation.bootstrap import paired_bd_bootstrap
from prism.evaluation.evaluator import h1_pass, secondary_h2, transfer_status
from prism.evaluation.metrics import RDPoint


def test_bd_bootstrap_recomputes_raw_curves():
    a = [RDPoint(47 - i * 5, float(i), 100 * 2**i) for i in range(5)]
    b = [RDPoint(p.qp, p.score, p.bitstream_bytes * 9 // 10) for p in a]
    data = {seed: {"a": (a, b), "b": (a, b)} for seed in (1, 2, 3)}
    result = paired_bd_bootstrap(data, unit="image")
    assert result["valid"] and result["ci95"] == pytest.approx([-10., -10.])


def test_missing_ci_cannot_pass_h1_and_secondary_boundaries():
    comparisons = {key: {"valid": True, "percent": -4, "ci95": None}
                   for key in ("M4-vs-M2", "M4-vs-M3")}
    assert not h1_pass(comparisons, independent_seeds=[1, 2, 3], resamples=1000,
                       task_floor_passed=True, holm_results=[{"reject": True}] * 10)
    assert secondary_h2(-1, .5) and not secondary_h2(-2.9, .5)
    assert [transfer_status(value) for value in (2, 3, 5, 5.1)] == ["PASS", "INCONCLUSIVE", "INCONCLUSIVE", "FAIL"]
