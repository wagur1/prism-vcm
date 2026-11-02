import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from prism.evaluation.bootstrap import paired_bd_bootstrap
from prism.evaluation.evaluator import evaluate, h1_pass, secondary_h2, transfer_status
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


def test_evaluator_supports_crf_primary_and_cqp_legacy_and_rejects_mismatches():
    crf_qps = [36, 32, 28, 24, 20]
    m1 = [{"qp": q, "score": float(i + 1), "bitstream_bytes": 100 * 2**i} for i, q in enumerate(crf_qps)]
    m2 = [{"qp": q, "score": float(i + 1), "bitstream_bytes": 90 * 2**i} for i, q in enumerate(crf_qps)]
    report = evaluate({
        "rc_mode": "CRF",
        "same_gop_and_tools": True,
        "curves": {"M1": m1, "M2": m2, "M3": m1, "M4": m2},
    })
    assert report["rc_mode"] == "CRF"
    assert report["operating_points"] == [20, 24, 28, 32, 36]
    assert report["M2-vs-M1"]["valid"] and report["M2-vs-M1"]["percent"] == pytest.approx(-10.0)
    assert report["M4-vs-M2"]["valid"] and report["M4-vs-M2"]["percent"] == pytest.approx(0.0)

    # Reject mode-grid mismatch (CQP mode with CRF points)
    with pytest.raises(ValueError, match="Expected exact CQP matrix"):
        evaluate({"rc_mode": "CQP", "same_gop_and_tools": True, "curves": {"M1": m1, "M2": m2, "M3": m1, "M4": m2}})

    # Reject duplicate points even when set matches
    dup = m1 + [m1[0]]
    with pytest.raises(ValueError, match="without duplicates"):
        evaluate({"rc_mode": "CRF", "same_gop_and_tools": True, "curves": {"M1": dup, "M2": m2, "M3": m1, "M4": m2}})
