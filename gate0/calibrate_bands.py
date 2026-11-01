#!/usr/bin/env python3
"""Offline, measurement-only band PROPOSAL. Never changes or decides GATE 0."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

CONDITIONS = ("zero", "plus6", "minus6", "checkerboard")
FIELDS = ("condition", "qp_group", "gop", "frame", "poc", "x", "y",
          "base_qp_frame", "zero_base_qp_frame", "final_qp", "zero_final_qp",
          "requested_offset", "net_offset_effect", "clipping_flag", "zero_clipping_flag")


def percentile(values: list[float], fraction: float) -> float:
    """Linear interpolation at (n-1)*fraction (inclusive/type 7 quantile)."""
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def stats(values: list[float]) -> dict:
    return {"count": len(values), "median": percentile(values, .5),
            "p10": percentile(values, .1), "p90": percentile(values, .9)}


def finite(row: dict, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key}")
    return value


def read_floors(path: Path) -> dict:
    """Only lower endpoints/gross floors are consumed; no upper/fidelity bands."""
    frozen = json.loads(path.read_text(encoding="utf-8"))
    floors = {"aggregate": frozen["aggregate_median_floor"],
              "first_frame": frozen["checkerboard"]["frame_0_uncompensated"][0],
              "plus6:1": frozen["uniform_maps"]["plus6"]["sign_1_compensated"][0],
              "minus6:-1": frozen["uniform_maps"]["minus6"]["sign_-1_compensated"][0],
              "checkerboard:1": frozen["checkerboard"]["sign_1_compensated"][0],
              "checkerboard:-1": frozen["checkerboard"]["sign_-1_compensated"][0]}
    return {key: float(value) for key, value in floors.items()}


def read_run(prefix: Path, expected_frames: int) -> tuple[dict, dict, dict]:
    log = prefix.with_suffix(".encode.log")
    if log.exists() and "x265_prism_telemetry_v2:" in log.read_text(encoding="utf-8", errors="replace"):
        raise ValueError(f"Native telemetry diagnostic: {prefix.name}")
    frames, cus, trace = {}, {}, {}
    with prefix.with_suffix(".telemetry.csv").open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if None in row or None in row.values():
                raise ValueError(f"Malformed telemetry row: {prefix.name}")
            frame = int(row["frame"])
            if row["record_type"] == "frame":
                if frame in frames:
                    raise ValueError("Duplicate telemetry frame")
                finite(row, "base_qp_rc")
                finite(row, "weighted_satd_or_blurred_complexity")
                finite(row, "qCompress")
                frames[frame] = row
            elif row["record_type"] == "cu":
                key = (frame, int(row["x"]), int(row["y"]))
                if key in cus:
                    raise ValueError("Duplicate telemetry CU")
                for field in ("requested_offset", "final_qp", "base_qp_at_cu"):
                    finite(row, field)
                if row["clipping_flag"] not in {"0", "1"}:
                    raise ValueError("Invalid clipping flag")
                cus[key] = row
            else:
                raise ValueError("Unknown telemetry record_type")
    with prefix.with_suffix(".trace.csv").open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            key = (int(row["frame"]), int(row["x"]), int(row["y"]))
            if key in trace:
                raise ValueError("Duplicate v1 CU")
            finite(row, "qp")
            trace[key] = row
    if trace.keys() != cus.keys():
        raise ValueError(f"v1/v2 CU coverage mismatch: {prefix.name}")
    if not cus or {key[0] for key in cus} != frames.keys() or len(frames) != expected_frames:
        raise ValueError(f"Incomplete frame coverage: {prefix.name}")
    if len({int(row["poc"]) for row in frames.values()}) != len(frames):
        raise ValueError("Duplicate POC")
    if {int(row["encode_order"]) for row in frames.values()} != set(range(expected_frames)):
        raise ValueError("Incomplete encode order")
    for key, row in cus.items():
        f = frames[key[0]]
        if (row["poc"], row["encode_order"], row["slice_type"]) != (
                f["poc"], f["encode_order"], f["slice_type"]):
            raise ValueError("Frame/CU identity mismatch")
        if finite(row, "final_qp") != finite(trace[key], "qp"):
            raise ValueError("v1/v2 final QP mismatch")
        if any(row[name] != trace[key][name] for name in ("has_residual", "skip")):
            raise ValueError("v1/v2 residual/skip mismatch")
    residual = {key: row for key, row in cus.items()
                if trace[key]["has_residual"] == "1" and trace[key]["skip"] == "0"}
    return frames, residual, cus


def gop_class(gop: int, poc: int) -> str:
    if gop == 1:
        return "gop1-all-intra"
    if gop != 10:
        raise ValueError(f"Unsupported GOP {gop}")
    return "gop10-first-frame" if poc == 0 else "gop10-inter"


def calibrate(evidence: Path, criteria: Path, output: Path, *,
              qp_groups: tuple[int, ...] = (27, 32, 37, 42, 47),
              gops: tuple[int, ...] = (1, 10), expected_frames: int = 10) -> dict:
    """Require the complete requested matrix; stream per-CU results to a sidecar.

    QP groups are the existing harness labels; actual CRF/RC settings are retained
    separately so repeated CRF settings cannot be mistaken for a CRF sweep.
    """
    evidence, output = evidence.resolve(), output.resolve()
    if output == criteria.resolve():
        raise ValueError("Proposal output cannot overwrite frozen criteria")
    floors = read_floors(criteria)
    groups, clipping, requests = defaultdict(list), defaultdict(list), defaultdict(list)
    frame_stats, provenance = [], []
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(".net_effects.csv")
    temporary = sidecar.with_suffix(".csv.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=FIELDS)
            writer.writeheader()
            for qp in qp_groups:
                for gop in gops:
                    loaded = {}
                    settings = None
                    for condition in CONDITIONS:
                        prefix = evidence / f"qp{qp}_gop{gop}_{condition}"
                        job = json.loads(prefix.with_suffix(".job.json").read_text(encoding="utf-8"))
                        config = job["config"]
                        if job["map"] != condition or config["qp"] != qp or config["gop"] != gop:
                            raise ValueError(f"Job attribution mismatch: {prefix.name}")
                        if settings is not None and settings != config:
                            raise ValueError("Offset and zero jobs use different encoder settings")
                        settings = config
                        loaded[condition] = read_run(prefix, expected_frames)
                        for suffix in (".job.json", ".trace.csv", ".telemetry.csv"):
                            path = prefix.with_suffix(suffix)
                            with path.open("rb") as source:
                                digest = hashlib.file_digest(source, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _sha(source)
                            provenance.append({"file": path.name, "sha256": digest})
                    zero_frames, zero, _ = loaded["zero"]
                    for condition in CONDITIONS:
                        frames, residual, all_cus = loaded[condition]
                        if frames.keys() != zero_frames.keys():
                            raise ValueError("Condition/zero PTS mismatch")
                        for frame, info in sorted(frames.items()):
                            zinfo = zero_frames[frame]
                            if info["poc"] != zinfo["poc"] or info["slice_type"] != zinfo["slice_type"]:
                                raise ValueError("Condition/zero frame type mismatch")
                            if info["rc_mode"] != zinfo["rc_mode"] or info["qCompress"] != zinfo["qCompress"]:
                                raise ValueError("Condition/zero RC mismatch")
                            poc = int(info["poc"])
                            base, zbase = finite(info, "base_qp_rc"), finite(zinfo, "base_qp_rc")
                            shared = sorted(key for key in residual if key[0] == frame and key in zero)
                            if not shared:
                                raise ValueError(f"No shared residual CUs: {condition}/{frame}")
                            values, reqs, clips = [], [], []
                            for key in shared:
                                cu, zcu = residual[key], zero[key]
                                local, zlocal = finite(cu, "final_qp"), finite(zcu, "final_qp")
                                requested = finite(cu, "requested_offset")
                                if finite(zcu, "requested_offset") != 0:
                                    raise ValueError("Zero telemetry contains a nonzero requested offset")
                                if condition == "plus6" and requested != 6 or condition == "minus6" and requested != -6:
                                    raise ValueError("Uniform requested offset mismatch")
                                if condition == "checkerboard" and requested not in {-6, 6}:
                                    raise ValueError("Checkerboard requested offset mismatch")
                                net = (local - base) - (zlocal - zbase)
                                clip = int(cu["clipping_flag"])
                                writer.writerow(dict(zip(FIELDS, (condition, qp, gop, frame, poc, key[1], key[2],
                                    base, zbase, local, zlocal, requested, net, clip, int(zcu["clipping_flag"])))))
                                values.append(net); reqs.append(requested); clips.append(clip)
                                if requested:
                                    direction = 1 if requested > 0 else -1
                                    group = (condition, direction, gop_class(gop, poc), qp)
                                    groups[group].append(direction * net)
                                    clipping[group].append(clip)
                                    requests[group].append(requested)
                            residual_count = sum(key[0] == frame for key in residual)
                            frame_stats.append({"condition": condition, "qp_group": qp, "gop": gop,
                                "frame": frame, "poc": poc, "encode_order": int(info["encode_order"]),
                                "slice_type": info["slice_type"], "base_qp": base, "zero_base_qp": zbase,
                                "net_offset_effect": stats(values), "requested_offset": stats(reqs),
                                "clipping_rate": sum(clips) / len(clips),
                                "shared_residual_count": len(shared), "residual_count": residual_count,
                                "shared_residual_coverage": len(shared) / residual_count,
                                "all_residual_clipping_rate": sum(int(cu["clipping_flag"]) for key, cu in residual.items()
                                    if key[0] == frame) / residual_count,
                                "all_cu_count": sum(key[0] == frame for key in all_cus),
                                "effective_config": settings, "rc_mode": info["rc_mode"],
                                "weighted_satd_or_blurred_complexity": finite(info, "weighted_satd_or_blurred_complexity"),
                                "qCompress": finite(info, "qCompress")})
        proposal_groups = []
        for group, values in sorted(groups.items()):
            condition, direction, cls, qp = group
            summary = stats(values)
            # gop1 contains both first and subsequent POCs; retain the strongest
            # applicable gross floor instead of diluting it by pooling frames.
            floor = max(floors["aggregate"], floors[f"{condition}:{direction}"],
                        floors["first_frame"] if cls != "gop10-inter" else 0)
            lower, upper = max(floor, math.floor(summary["p10"])), math.ceil(summary["p90"])
            proposal_groups.append({"condition": condition, "direction": direction, "gop_class": cls,
                "qp_group": qp, **summary, "fidelity": sum(v > 0 for v in values) / len(values),
                "clipping_rate": sum(clipping[group]) / len(values),
                "requested_offset": stats(requests[group]), "gross_floor": floor,
                "proposed_band": [lower, upper], "status": "PROPOSAL",
                "floor_conflict": lower > upper})
        asymmetry = []
        for qp in qp_groups:
            for cls in sorted({group[2] for group in groups}):
                for family, plus, minus in (("uniform", "plus6", "minus6"),
                                             ("checkerboard", "checkerboard", "checkerboard")):
                    pos, neg = groups.get((plus, 1, cls, qp)), groups.get((minus, -1, cls, qp))
                    if pos and neg:
                        asymmetry.append({"map_family": family, "gop_class": cls, "qp_group": qp,
                            "plus_median": percentile(pos, .5), "minus_median": percentile(neg, .5),
                            "plus_median_minus_minus_median": percentile(pos, .5) - percentile(neg, .5)})
        proposal = {"status": "PROPOSAL", "council_ratification_required": True,
            "auto_apply": False, "gate_verdict_unchanged": True,
            "evidence_directory": str(evidence), "criteria_floors": floors,
            "formula": "(QP_local_offset - QP_base_frame_offset) - (QP_local_zero - QP_base_frame_zero)",
            "statistics_units": "direction * net_offset_effect; minus median is a positive-direction magnitude",
            "band_rule": "[max(existing_gross_floor, floor(p10)), ceil(p90)]",
            "quantiles": "linear interpolation, (n-1)*p",
            "limitations": ["Shared residual leaf origins, not independent QGs or clips; samples may share a QG.",
                "QP groups are harness labels; inspect effective_config.crf before claiming a CRF sweep.",
                "Calibration data alone cannot ratify bands; independent validation is required.",
                "Conditions remain separate because uniform and checkerboard gross floors differ."],
            "per_frame": frame_stats, "groups": proposal_groups, "asymmetry": asymmetry,
            "inputs": provenance, "net_effects_csv": sidecar.name}
        temporary.replace(sidecar)
        output.write_text(json.dumps(proposal, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return proposal
    finally:
        temporary.unlink(missing_ok=True)


def _sha(source) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("band_calibration_proposal.json"))
    args = parser.parse_args()
    calibrate(args.evidence, Path(__file__).with_name("criteria_frozen.json"), args.output)
    print(f"PROPOSAL only: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
