#!/usr/bin/env python3
"""esc#11 measurement-only QG16 recalibration from existing telemetry; no encoding."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prism.codec.gate0 import (QG16_CELL_ACTIVE_COVERAGE, QG16_MIN_JOINT_COVERAGE,
                              project_telemetry_qg, qg_net_samples, qg_statistics)

CONDITIONS = ("zero", "plus6", "minus6", "checkerboard")


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def input_map(width: int, height: int, condition: str) -> list[list[int]]:
    """Exact spike stimulus: checkerboard changes sign every CTU64 (4 QG16s)."""
    return [[(6 if (y // 4 + x // 4) % 2 else -6) if condition == "checkerboard"
             else {"zero": 0, "plus6": 6, "minus6": -6}[condition]
             for x in range((width + 15) // 16)] for y in range((height + 15) // 16)]


def proposed_band(summary: dict, profile: dict, frame_class: str, aggregate_floor: float) -> dict:
    """No rounding or upper-bound widening; conflicts remain visible for council."""
    if frame_class == "first_frame":
        band = list(profile["first_frame"])
        floor = max(aggregate_floor, band[0])
        band[0] = floor
        rule = "AG-2 esc#11 fixed first-frame framework [5.5, 6.5]"
    else:
        frozen = profile.get("compensated", profile.get("compensated_per_direction"))
        floor = max(aggregate_floor, frozen[0])
        band = [max(floor, summary["p10"]), summary["p90"]] if summary["count"] else None
        rule = "[max(type7 p10, frozen direction floor, frozen aggregate floor), type7 p90]"
    return {"proposed_band": band, "gross_floor": floor, "band_rule": rule,
            "floor_conflict": band is None or band[0] > band[1],
            "status": "PROPOSAL"}


def recalibrate(evidence: Path, output: Path, *,
                criteria: Path = ROOT / "gate0/criteria_frozen_v2.json",
                qp_groups=(27, 32, 37, 42, 47), gops=(1, 10), expected_frames=10) -> dict:
    """Read the complete matrix, preserving QP labels and effective CRF settings.

    Uses the SAME projection/samples as verify_net_effect_qg. It intentionally
    does not call the gate verifier: these data propose bands for ratification.
    """
    evidence, output, criteria = evidence.resolve(), output.resolve(), criteria.resolve()
    if output == criteria or output.name.startswith("criteria_frozen") or output.parent == evidence:
        raise ValueError("Proposal must not overwrite frozen criteria or source evidence")
    frozen = json.loads(criteria.read_text(encoding="utf-8"))
    profiles, floor = frozen["bands_magnitude"], frozen["aggregate_floor_magnitude"]
    groups, pooled = defaultdict(list), defaultdict(list)
    per_frame, provenance, problems, settings = [], [], [], []
    # Check the complete requested matrix before measurement begins.
    for qp in qp_groups:
        for gop in gops:
            for condition in CONDITIONS:
                for suffix in (".telemetry.csv", ".job.json"):
                    path = evidence / f"qp{qp}_gop{gop}_{condition}{suffix}"
                    if not path.is_file():
                        raise FileNotFoundError(path)
    for qp in qp_groups:
        for gop in gops:
            loaded, reference = {}, None
            for condition in CONDITIONS:
                prefix = evidence / f"qp{qp}_gop{gop}_{condition}"
                job_path, telemetry = prefix.with_suffix(".job.json"), prefix.with_suffix(".telemetry.csv")
                job = json.loads(job_path.read_text(encoding="utf-8"))
                if job["map"] != condition or job["config"]["qp"] != qp or job["config"]["gop"] != gop:
                    raise ValueError(f"Job attribution mismatch: {prefix.name}")
                signature = {k: job[k] for k in ("width", "height", "config", "pin")}
                if reference is not None and reference != signature:
                    raise ValueError(f"Zero/condition video dimensions/config/build mismatch: {prefix.name}")
                reference = signature
                log = prefix.with_suffix(".encode.log")
                if log.exists() and "x265_prism_telemetry_v2:" in log.read_text(encoding="utf-8", errors="replace"):
                    raise ValueError(f"Native telemetry diagnostic: {prefix.name}")
                loaded[condition] = project_telemetry_qg(telemetry, job["width"], job["height"])
                if set(loaded[condition]["frames"]) != set(range(expected_frames)):
                    raise ValueError(f"Incomplete frame matrix: {prefix.name}")
                provenance.extend({"file": p.name, "sha256": sha256(p)} for p in (job_path, telemetry))
            width, height = reference["width"], reference["height"]
            settings.append({"qp_group": qp, "gop": gop, **reference,
                             "visible_qg16": [(width + 15) // 16, (height + 15) // 16]})
            for condition in CONDITIONS:
                samples = qg_net_samples(loaded["zero"], loaded[condition], input_map(width, height, condition))
                for row in samples:
                    sign, cls = row["direction"], row["frame_class"]
                    label = f"gop{gop}-{'all-intra' if gop == 1 else 'mixed'}"
                    key = (condition, sign, label, qp, cls)
                    groups[key].extend(row["values"])
                    pooled[(condition, sign, cls)].extend(row["values"])
                    record = {k: v for k, v in row.items() if k != "values"}
                    record.update(condition=condition, qp_group=qp, gop=gop,
                                  gop_class=label, **qg_statistics(row["values"]))
                    per_frame.append(record)
                    if row["coverage"]["joint_fraction"] < QG16_MIN_JOINT_COVERAGE:
                        problems.append({"condition": condition, "qp_group": qp, "gop": gop,
                                         "frame": row["frame"], "direction": sign,
                                         "reason": "joint active coverage below 0.8; INVALID for gate"})
    group_reports, bands = [], []
    for (condition, sign, label, qp, cls), values in sorted(groups.items()):
        summary = qg_statistics(values)
        record = {"condition": condition, "direction": sign, "gop_class": label,
                  "qp_group": qp, "frame_class": cls, **summary}
        if sign:
            record.update(proposed_band(summary, profiles[condition], cls, floor))
        group_reports.append(record)
    for (condition, sign, cls), values in sorted(pooled.items()):
        if not sign:
            continue  # zero self-comparisons are diagnostics, never a signed band.
        summary = qg_statistics(values)
        bands.append({"condition": condition, "direction": sign, "frame_class": cls,
                      **summary, **proposed_band(summary, profiles[condition], cls, floor),
                      "minimum_fidelity": 1.0 if cls == "first_frame" else profiles[condition]["fidelity"]})
    proposal = {
        "schema": "gate0-band-calibration-v3-proposal", "status": "PROPOSAL",
        "council_ratification_required": True, "auto_apply": False, "gate_verdict": None,
        "metric": "net-effect-qg16", "evidence_directory": str(evidence),
        "criteria_source": {"file": str(criteria), "sha256": sha256(criteria)},
        "source_files": provenance, "settings": settings,
        "method": {
            "dedup": "(frame,qg_x,qg_y,qg_size); all final_qp/base_qp_at_cu pairs must agree",
            "projection": "sum(intersection_area * (final_qp-base_qp_at_cu))/sum(active intersection_area)",
            "sign": "input map cell; checkerboard flips every 64 pixels, not every 16",
            "frame_class": "frame 0 first/uncompensated; frame >=1 compensated, including GOP1 I slices",
            "cell_active_coverage_threshold": QG16_CELL_ACTIVE_COVERAGE,
            "minimum_joint_coverage": QG16_MIN_JOINT_COVERAGE,
            "quantiles": "type 7, no rounding; per-cell distribution, not per-CU or frame-median quantiles",
            "aggregate_floor_magnitude": floor,
            "source_video": "same run's fixed deterministic synthetic generator; dimensions/config/pin and frame/POC checked",
            "inference": "descriptive only; cells and repeated CRF/QP-label data are not independent replicates",
        },
        "bands": bands, "groups": group_reports, "per_frame": per_frame,
        "coverage_problems": problems,
        "freeze_protocol": "Council ratifies and freezes v1.6 criteria/SHA before ONE offline confirmatory pass; hard-stop afterward. No encoding or automatic criterion changes.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(proposal, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    return proposal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "gate0/band_calibration_v3_proposal.json")
    parser.add_argument("--criteria", type=Path, default=ROOT / "gate0/criteria_frozen_v2.json")
    args = parser.parse_args()
    proposal = recalibrate(args.evidence, args.output, criteria=args.criteria)
    print(json.dumps({"status": proposal["status"], "output": str(args.output),
                      "bands": proposal["bands"], "coverage_problems": len(proposal["coverage_problems"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
