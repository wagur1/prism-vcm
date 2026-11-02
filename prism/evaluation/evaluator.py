"""Offline raw-RD report runner; no inferred measurements or extrapolated scores."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from .metrics import RDPoint, bd_rate, holm_bonferroni


RC_GRIDS = {
    "CRF": {20, 24, 28, 32, 36},
    "CQP": {27, 32, 37, 42, 47},
}


def evaluate(raw: dict) -> dict:
    curves = {name: [RDPoint(**point) for point in points]
              for name, points in raw["curves"].items()}
    if not {"M1", "M2", "M3", "M4"}.issubset(curves):
        raise ValueError("Report all four pre-registered baselines")
    rc_mode = raw.get("rc_mode")
    if rc_mode not in RC_GRIDS or not raw.get("same_gop_and_tools"):
        raise ValueError("Primary comparisons require CRF or CQP and identical GOP/tools")
    qps = RC_GRIDS[rc_mode]
    if any(len(curve) != len(qps) or {point.qp for point in curve} != qps for curve in curves.values()):
        raise ValueError(f"Expected exact {rc_mode} matrix {sorted(qps)} without duplicates")
    report = {f"M4-vs-{anchor}": asdict(bd_rate(curves[anchor], curves["M4"]))
              for anchor in ("M1", "M2", "M3")}
    report["M2-vs-M1"] = asdict(bd_rate(curves["M1"], curves["M2"]))
    report["M3-vs-M1"] = asdict(bd_rate(curves["M1"], curves["M3"]))
    report["rc_mode"] = rc_mode
    report["operating_points"] = sorted(qps)
    report["inference_status"] = "descriptive_only"
    # Independent seed and paired-cluster evidence is mandatory for an H1 claim.
    if "p_values" in raw:
        if len(raw["p_values"]) != 10:
            raise ValueError("Holm family must contain M4-vs-{M2,M3} x 5QP")
        report["holm_family"] = holm_bonferroni(raw["p_values"])
    return report


def h1_pass(comparisons: dict[str, dict], *, independent_seeds: list[int],
            resamples: int, task_floor_passed: bool, holm_results: list[dict]) -> bool:
    if len(set(independent_seeds)) < 3 or resamples < 1000 or not task_floor_passed:
        return False
    if len(holm_results) != 10 or not all(item["reject"] for item in holm_results):
        return False
    for name in ("M4-vs-M2", "M4-vs-M3"):
        value = comparisons.get(name, {})
        interval = value.get("ci95")
        if (not value.get("valid") or value.get("percent") is None or
                not math.isfinite(value["percent"]) or value["percent"] > -3.0 or
                not interval or len(interval) != 2 or not all(math.isfinite(x) for x in interval) or
                interval[0] > interval[1] or interval[1] >= 0):
            return False
    return True


def secondary_h2(bd_percent: float, ap_small_delta_pp: float) -> bool:
    return 1.0 <= -bd_percent < 2.9 and ap_small_delta_pp >= 0.5


def transfer_status(bd_percent: float) -> str:
    return "PASS" if bd_percent <= 2 else "FAIL" if bd_percent > 5 else "INCONCLUSIVE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evaluate(json.loads(args.raw.read_text(encoding="utf-8"))),
                                     indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
