"""Gate evidence validation; average CSV QP is never spatial evidence."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def frame_bits(csv_path: Path, expected: int = 10) -> list[int]:
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [{key.strip(): value.strip() for key, value in row.items() if key}
                for row in csv.DictReader(stream)]
    bits = [int(float(row["Bits"])) for row in rows
            if row.get("Bits", "").replace(".", "", 1).isdigit()]
    if len(bits) != expected or any(value <= 0 for value in bits):
        raise ValueError("Expected ten positive per-frame bit counts from x265 --csv")
    return bits


def residual_trace(path: Path, width: int, height: int,
                   stats_out: dict[str, dict[str, int]] | None = None) -> dict[tuple[int, int, int], float]:
    """Read x265_prism_residual_v1: frame,x,y,qp,has_residual,skip (pixel coordinates)."""
    valid: dict[tuple[int, int, int], float] = {}
    seen: set[tuple[int, int, int]] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["frame", "x", "y", "qp", "has_residual", "skip"]:
            raise ValueError("Expected x265_prism_residual_v1 CSV header")
        if stats_out is not None:
            stats_out.clear()
        for row in reader:
            frame, x, y = (int(row[key]) for key in ("frame", "x", "y"))
            if stats_out is not None:
                entry = stats_out.setdefault(str(frame), {"rows": 0, "residual": 0, "skip": 0})
                entry["rows"] += 1
                if row["has_residual"] == "1":
                    entry["residual"] += 1
                if row["skip"] == "1":
                    entry["skip"] += 1
            key = (frame, x, y)
            if key in seen:
                raise ValueError("Duplicate CU trace coordinates; candidates/stale traces are not evidence")
            seen.add(key)
            if row["skip"] not in {"0", "1"} or row["has_residual"] not in {"0", "1"}:
                raise ValueError("Residual and skip flags must be 0 or 1")
            if not 0 <= frame < 10 or not (0 <= x < width and 0 <= y < height):
                continue  # Exclude padding and retain only the visible 30 x 17 CTUs at 1080p.
            if int(row["skip"]) or not int(row["has_residual"]):
                continue
            qp = float(row["qp"])
            if not math.isfinite(qp) or not 0 <= qp <= 51:
                raise ValueError("Invalid realized residual-block QP")
            valid[(frame, x, y)] = qp
    if {key[0] for key in valid} != set(range(10)):
        raise ValueError("Residual-block trace must cover all ten frames")
    return valid


def _read_telemetry_cus(path: Path) -> dict[tuple[int, int, int], tuple[float, float, float]]:
    """Read x265_prism_telemetry_v2 CU rows: (frame,x,y) -> (final_qp, base_qp_at_cu, requested_offset)."""
    out: dict[tuple[int, int, int], tuple[float, float, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("record_type") != "cu" or row.get("skip") == "1" or row.get("has_residual") != "1":
                continue
            key = (int(row["frame"]), int(row["x"]), int(row["y"]))
            out[key] = (float(row["final_qp"]), float(row["base_qp_at_cu"]),
                        float(row["requested_offset"] or 0.0))
    return out


def verify_net_effect(zero_telemetry: Path, condition_telemetry: Path,
                      offsets: np.ndarray, *,
                      bands: dict[int, tuple[float, float]] | None = None,
                      min_fidelity: float = 1.0,
                      min_aggregate_median: float = 4.9,
                      i_frames: set[int] | None = None) -> dict:
    """CONFIG v1.5 criteria (AG-2 ratified esc#10): band the NET offset effect.

    net CU effect = (final_qp - base_qp_at_cu) of the offset run minus the same
    quantity of the zero run (cancels CRF base drift, CX-1 esc#8). Oriented by
    map sign; band on |oriented median| per frame-class: frame 0 (uncompensated)
    [5.5, 6.5]; frames >= 1 per-direction ratified bands (default plus [4.9, 7.0],
    minus [4.9, 7.0], checkerboard [4.9, 7.2]). Sign fidelity per direction:
    uniform 1.0, checkerboard >= 0.98. Aggregate |median| >= 4.9.
    Frame classes follow feedback position (ZC-2 warning esc#10), not slice type.
    """
    import numpy as np

    intra = {0} if i_frames is None else set(i_frames)
    zero_cus = _read_telemetry_cus(zero_telemetry)
    cond_cus = _read_telemetry_cus(condition_telemetry)
    shared = zero_cus.keys() & cond_cus.keys()
    if not shared:
        raise ValueError("No shared residual CUs between zero and condition telemetry")
    signs = sorted({int(np.sign(v)) for row in offsets for v in row if int(np.sign(v))})
    result: dict = {"metric": "net-effect", "per_frame": {}}
    for sign in signs:
        per_frame: dict[int, list[float]] = {}
        fidelity_bad = 0
        fidelity_total = 0
        for key in shared:
            frame, x, y = key
            if int(np.sign(offsets[y // 16, x // 16])) != sign:
                continue
            net_zero = zero_cus[key][0] - zero_cus[key][1]
            net_cond = cond_cus[key][0] - cond_cus[key][1]
            oriented = sign * (net_cond - net_zero)
            fidelity_total += 1
            if oriented <= 0:
                fidelity_bad += 1
            per_frame.setdefault(frame, []).append(oriented)
        if not fidelity_total:
            raise ValueError(f"No oriented residual CUs for sign {sign}")
        fidelity = 1.0 - fidelity_bad / fidelity_total
        if fidelity < min_fidelity:
            raise ValueError(
                f"Sign-fidelity violation sign {sign}: {fidelity_bad}/{fidelity_total} "
                f"CUs with net delta <= 0 (fidelity={fidelity:.4f} < {min_fidelity})")
        for frame in sorted(per_frame):
            entries = per_frame[frame]
            median = float(np.median(entries))
            kind = "I" if frame in intra else "inter"
            if frame == 0:
                band = (5.5, 6.5)
            elif bands is not None and sign in bands:
                band = bands[sign]
            else:
                band = (4.9, 7.0)
            if not band[0] <= median <= band[1]:
                raise ValueError(
                    f"{kind} frame {frame} sign {sign}: |net| median {median:.3f} outside "
                    f"v1.5 band {band} (CONFIG v1.5, AG-2 ratified esc#10)")
            result["per_frame"][f"{frame}:{sign}"] = {
                "pairs": len(entries), "median": median, "kind": kind,
                "fidelity": fidelity}
        aggregate = float(np.median([v for entries in per_frame.values() for v in entries]))
        if abs(aggregate) < min_aggregate_median:
            raise ValueError(
                f"Aggregate net effect too weak for sign {sign}: "
                f"|median|={abs(aggregate):.3f} < {min_aggregate_median} (CONFIG v1.5)")
        result[str(sign)] = aggregate
    return result


QG16_CELL_ACTIVE_COVERAGE = 1.0
QG16_MIN_JOINT_COVERAGE = 0.8  # esc#11 ZC-3: >20% excluded => INVALID evidence.


class QGInvalidEvidence(ValueError):
    """Insufficient active support; no gate verdict can be inferred."""


def qg_statistics(values: list[float]) -> dict:
    """Cell statistics; quantiles use inclusive linear interpolation (type 7)."""
    if not values:
        return {"count": 0, "median": None, "median_abs": None,
                "p10": None, "p90": None, "fidelity": None}

    def quantile(ordered, p):
        index = (len(ordered) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    ordered = sorted(values)
    return {"count": len(values), "median": quantile(ordered, .5),
            "median_abs": quantile(sorted(abs(v) for v in values), .5),
            "p10": quantile(ordered, .1), "p90": quantile(ordered, .9),
            "fidelity": sum(v > 0 for v in values) / len(values)}


def project_telemetry_qg(path: Path, width: int, height: int) -> dict:
    """Deduplicate effective QGs, validate their tiling, and project onto QG16.

    QG16 cells are intersected with the visible image (120x68 at 1080p,
    last row 8 pixels high). Padding is discarded, never clamped. Active
    means ANY leaf has residual and is not skipped, but ALL leaves of a QG
    must agree on (final_qp, base_qp_at_cu), including inactive leaves.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Positive visible dimensions required")
    columns, rows = (width + 15) // 16, (height + 15) // 16
    groups, identities, frame_rows = {}, {}, set()
    required = {"record_type", "frame", "poc", "encode_order", "slice_type",
                "qg_x", "qg_y", "qg_size", "final_qp", "base_qp_at_cu",
                "has_residual", "skip"}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing QG telemetry columns: {path}")
        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"Malformed QG telemetry row: {path}")
            frame = int(row["frame"])
            identity = (int(row["poc"]), int(row["encode_order"]), row["slice_type"],
                        row.get("rc_mode"), row.get("qCompress"))
            if frame < 0 or (frame in identities and identities[frame] != identity):
                raise ValueError(f"Frame/CU identity mismatch: {frame}")
            identities[frame] = identity
            if row["record_type"] == "frame":
                if frame in frame_rows:
                    raise ValueError(f"Duplicate telemetry frame: {frame}")
                frame_rows.add(frame)
                continue
            if row["record_type"] != "cu":
                raise ValueError("Unknown telemetry record_type")
            if row["has_residual"] not in {"0", "1"} or row["skip"] not in {"0", "1"}:
                raise ValueError("Residual and skip flags must be 0 or 1")
            x, y, size = (int(row[k]) for k in ("qg_x", "qg_y", "qg_size"))
            if size not in (16, 32, 64) or x < 0 or y < 0 or x % 16 or y % 16:
                raise ValueError(f"Invalid effective QG region: {(frame, x, y, size)}")
            qp = (float(row["final_qp"]), float(row["base_qp_at_cu"]))
            if not all(math.isfinite(v) for v in qp) or not 0 <= qp[0] <= 51:
                raise ValueError("Non-finite or invalid QG QP")
            key = (frame, x, y, size)
            active = row["has_residual"] == "1" and row["skip"] == "0"
            if key in groups:
                previous_qp, previous_active = groups[key]
                if previous_qp != qp:
                    raise ValueError(f"Inconsistent QP within effective QG {key}; cu_size is unavailable")
                active |= previous_active
            groups[key] = (qp, active)
    if not frame_rows or frame_rows != identities.keys() or {k[0] for k in groups} != frame_rows:
        raise ValueError("Incomplete telemetry frame coverage")
    if (len({v[0] for v in identities.values()}) != len(identities)
            or {v[1] for v in identities.values()} != set(range(len(identities)))):
        raise ValueError("Duplicate POC or incomplete encode order")
    area = [min(16, width - x * 16) * min(16, height - y * 16)
            for y in range(rows) for x in range(columns)]
    projected = {f: {"area": [0] * len(area), "active_area": [0] * len(area),
                     "weighted_delta": [0.0] * len(area)} for f in frame_rows}
    for (frame, x, y, size), (qp, active) in groups.items():
        if x >= width or y >= height:
            continue
        target = projected[frame]
        for cy in range(y // 16, (min(y + size, height) + 15) // 16):
            for cx in range(x // 16, (min(x + size, width) + 15) // 16):
                index = cy * columns + cx
                covered = (min(x + size, width, (cx + 1) * 16) - max(x, cx * 16)) * (
                    min(y + size, height, (cy + 1) * 16) - max(y, cy * 16))
                target["area"][index] += covered
                if target["area"][index] > area[index]:
                    raise ValueError(f"Overlapping QG coverage: frame {frame}, cell {(cx, cy)}")
                if active:
                    target["active_area"][index] += covered
                    target["weighted_delta"][index] += covered * (qp[0] - qp[1])
    for frame, target in projected.items():
        if target["area"] != area:
            raise ValueError(f"Missing QG coverage: frame {frame}")
        target["coverage"] = [a / c for a, c in zip(target.pop("active_area"), area)]
        # Inactive support is missing, never a zero-valued QP effect.
        target["delta"] = [v / (c * a) if a else None for v, c, a in
                           zip(target.pop("weighted_delta"), area, target["coverage"])]
        del target["area"]
    return {"width": width, "height": height, "columns": columns, "rows": rows,
            "identities": identities, "frames": projected, "cell_area": area}


def qg_net_samples(zero: dict, condition: dict, offsets) -> list[dict]:
    """One oriented observation per fully active visible cell, per frame/sign.

    Reports both/zero-only/condition-only/neither active coverage. The 1.0 cell
    threshold is fixed; missing/overlapping geometric support already raises
    during projection. Video/config attribution is checked by the caller's jobs.
    """
    if any(zero[k] != condition[k] for k in ("width", "height", "identities")):
        raise ValueError("Zero/condition dimensions or frame/POC/config mismatch")
    columns, rows = zero["columns"], zero["rows"]
    if len(offsets) < rows or any(len(row) < columns for row in offsets[:rows]):
        raise ValueError("Offset map does not cover visible QG16 grid")
    visible = [float(offsets[y][x]) for y in range(rows) for x in range(columns)]
    if not all(math.isfinite(v) for v in visible):
        raise ValueError("Non-finite offset map")
    signs = [(v > 0) - (v < 0) for v in visible]
    result = []
    for frame in sorted(zero["frames"]):
        z, c = zero["frames"][frame], condition["frames"][frame]
        for sign in sorted(set(signs)):
            indices = [i for i, s in enumerate(signs) if s == sign]
            counts = dict.fromkeys(("both", "zero_only", "condition_only", "neither"), 0)
            values = []
            for i in indices:
                za, ca = z["coverage"][i] >= QG16_CELL_ACTIVE_COVERAGE, c["coverage"][i] >= QG16_CELL_ACTIVE_COVERAGE
                category = "both" if za and ca else "zero_only" if za else "condition_only" if ca else "neither"
                counts[category] += 1
                if za and ca:
                    net = c["delta"][i] - z["delta"][i]
                    values.append(sign * net if sign else net)
            total_area = sum(zero["cell_area"][i] for i in indices)
            coverage = {"visible_cells": len(indices), **counts,
                        "joint_fraction": counts["both"] / len(indices),
                        "zero_active_area_fraction": sum(z["coverage"][i] * zero["cell_area"][i] for i in indices) / total_area,
                        "condition_active_area_fraction": sum(c["coverage"][i] * zero["cell_area"][i] for i in indices) / total_area}
            result.append({"frame": frame, "poc": zero["identities"][frame][0],
                           "direction": sign, "frame_class": "first_frame" if frame == 0 else "compensated",
                           "coverage": coverage, "values": values})
    return result


def verify_net_effect_qg(zero_telemetry: Path, condition_telemetry: Path,
                         offsets, *, bands: dict, min_fidelity: float,
                         min_aggregate_median: float, i_frames: set[int],
                         width: int, height: int) -> dict:
    """esc#11 QG16 gate. All decision thresholds must be supplied explicitly.

    bands = {"first_frame": {sign: (lo, hi)}, "compensated": {sign: (lo, hi)}}.
    Feedback class is frame 0 versus >=1; i_frames only labels slice syntax.
    Bands apply to median |oriented|, with signed median and type-7 p10/p90 also
    reported. Frame 0 requires fidelity 1.0; later frames use min_fidelity.
    Less than 80% joint active cells in any frame/sign is INVALID evidence.
    """
    if not 0 < min_fidelity <= 1 or not math.isfinite(min_aggregate_median) or min_aggregate_median <= 0:
        raise ValueError("Explicit positive fidelity and aggregate thresholds required")
    for kind in ("first_frame", "compensated"):
        if kind not in bands or not bands[kind]:
            raise ValueError(f"Missing explicit QG band: {kind}")
    samples = qg_net_samples(project_telemetry_qg(zero_telemetry, width, height),
                             project_telemetry_qg(condition_telemetry, width, height), offsets)
    signed = [r for r in samples if r["direction"]]
    if not signed or not any(r["frame"] == 0 for r in signed):
        raise ValueError("Signed QG evidence including frame 0 is required")
    result = {"metric": "net-effect-qg16", "per_frame": {}, "aggregate": {}, "frame_classes": {},
              "cell_active_coverage_threshold": QG16_CELL_ACTIVE_COVERAGE,
              "minimum_joint_coverage": QG16_MIN_JOINT_COVERAGE}
    aggregate, classes = {}, {}
    for row in signed:
        frame, sign, kind = row["frame"], row["direction"], row["frame_class"]
        band = bands[kind].get(sign, bands[kind].get(str(sign)))
        if (band is None or len(band) != 2 or not all(math.isfinite(v) for v in band)
                or not 0 < band[0] <= band[1]):
            raise ValueError(f"Missing or invalid explicit QG band: {kind}/{sign}")
        if row["coverage"]["joint_fraction"] < QG16_MIN_JOINT_COVERAGE:
            raise QGInvalidEvidence(f"INVALID QG evidence: insufficient active coverage frame {frame} sign {sign}: {row['coverage']}")
        stats = qg_statistics(row["values"])
        fidelity = 1.0 if frame == 0 else min_fidelity
        if stats["fidelity"] < fidelity:
            raise ValueError(f"Sign-fidelity violation frame {frame} sign {sign}: {stats['fidelity']} < {fidelity}")
        if not band[0] <= stats["median_abs"] <= band[1]:
            raise ValueError(f"{kind} frame {frame} sign {sign}: median |net| {stats['median_abs']} outside QG band {band}")
        result["per_frame"][f"{frame}:{sign}"] = {**stats, "coverage": row["coverage"],
            "frame_class": kind, "kind": "I" if frame in i_frames else "inter"}
        aggregate.setdefault(sign, []).extend(row["values"])
        classes.setdefault(f"{kind}:{sign}", []).extend(row["values"])
    for sign, values in aggregate.items():
        stats = qg_statistics(values)
        if stats["median_abs"] < min_aggregate_median:
            raise ValueError(f"Aggregate QG effect too weak for sign {sign}: {stats['median_abs']} < {min_aggregate_median}")
        result["aggregate"][str(sign)] = stats
        result[str(sign)] = stats["median"]
    result["frame_classes"] = {key: qg_statistics(v) for key, v in classes.items()}
    return result


def verify_spatial_effect(anchor: dict[tuple[int, int, int], float],
                          changed: dict[tuple[int, int, int], float],
                          offsets: np.ndarray, *,
                          min_pairs: int = 32, min_qg: int = 8,
                          min_aggregate_median: float = 3.0,
                          bands: dict[int, tuple[float, float]] | None = None,
                          min_fidelity: float = 1.0,
                          min_separation: float | None = None) -> dict:
    """CONFIG v1.4 section E-F0 CRF criteria (council esc#4 + esc#6, approved 9/9).

    Tier 1 (per frame x sign): >= min_pairs shared residual CUs over >= min_qg
    QG16 cells; sign-fidelity is enforced per min_fidelity (uniform maps 100%,
    dual-sign checkerboard 98% - AG-2 esc#8); median band by FEEDBACK STATE
    (AG-2 esc#7): frame 0 runs open-loop (uncompensated, band [5, 7] per CX-1
    esc#8); frames >= 1 are rate-control compensated with DIRECTION-ASYMMETRIC
    provisional bands (AG-2 esc#8 + CX-1 esc#8): plus [2.5, 4.5], minus
    [3.5, 6.0] (observed 4.0-5.0 across QP/GOP; wide gross-guard until CX-1
    deep-telemetry calibration locks final bands). Checkerboard: wide
    gross-guard bands + per-sign separation >= 7.0. Scope: continuous encode,
    fixed GOP, no mid-stream scene-cut. Tier 2: aggregate median >= 3.00. All
    failures FAIL loudly.
    """
    import numpy as np

    signs = sorted({int(np.sign(v)) for row in offsets for v in row if int(np.sign(v))})
    if not signs:
        raise ValueError("Offset map has no signed regions")
    shared = anchor.keys() & changed.keys()
    result: dict = {"per_frame": {}, "frames_checked": sorted({key[0] for key in anchor})}
    for sign in signs:
        samples: list[float] = []
        per_frame: dict[int, list[tuple[float, int, int]]] = {}
        for key in shared:
            frame, x, y = key
            if int(np.sign(offsets[y // 16, x // 16])) != sign:
                continue
            delta = sign * (changed[key] - anchor[key])
            samples.append(delta)
            per_frame.setdefault(frame, []).append((delta, x // 16, y // 16))
        if not samples:
            raise ValueError(f"No shared residual CUs for offset sign {sign}")
        for frame in sorted(result["frames_checked"]):
            entries = per_frame.get(frame, [])
            qg_spread = {(qx, qy) for _, qx, qy in entries}
            if len(entries) < min_pairs or len(qg_spread) < min_qg:
                raise ValueError(
                    f"Insufficient residual pairs frame {frame} sign {sign}: "
                    f"pairs={len(entries)} (<{min_pairs}) qg_spread={len(qg_spread)} (<{min_qg})")
            bad = [d for d, _, _ in entries if d <= 0]
            if bad:
                raise ValueError(
                    f"Sign-fidelity violation frame {frame} sign {sign}: "
                    f"{len(bad)}/{len(entries)} CUs with realized QP delta <= 0 "
                    "(AG-2 hard gate: 100% required)")
            deltas = [d for d, _, _ in entries]
            bad = [d for d in deltas if d <= 0]
            fidelity = 1.0 - len(bad) / len(entries)
            if fidelity < min_fidelity:
                raise ValueError(
                    f"Sign-fidelity violation frame {frame} sign {sign}: "
                    f"{len(bad)}/{len(entries)} CUs with realized QP delta <= 0 "
                    f"(fidelity={fidelity:.4f} < {min_fidelity})")
            median = float(np.median(deltas))
            state = "uncompensated" if frame == 0 else "compensated"
            if frame == 0 or bands is None:
                band = (5.0, 7.0) if frame == 0 else (2.5, 4.5)
            else:
                band = bands.get(sign, bands.get(str(sign)))
                if band is None:
                    raise ValueError(f"No band configured for sign {sign}")
            if not band[0] <= median <= band[1]:
                raise ValueError(
                    f"{state} frame {frame} sign {sign}: median {median:.3f} outside "
                    f"feedback-state band {band} (CONFIG v1.4 E-F0, AG-2/CX-1 esc#8)")
            sorted_d = sorted(deltas)
            result["per_frame"][f"{frame}:{sign}"] = {
                "pairs": len(entries), "qg_spread": len(qg_spread), "median": median,
                "state": state, "fidelity": fidelity,
                "p10": float(sorted_d[max(0, int(0.10 * len(sorted_d)) - 1)]),
                "p90": float(sorted_d[min(len(sorted_d) - 1, int(0.90 * len(sorted_d)))])}
        aggregate = float(np.median(samples))
        if aggregate < min_aggregate_median:
            raise ValueError(
                f"Aggregate realized QP effect too weak for sign {sign}: "
                f"median={aggregate:.3f} < {min_aggregate_median} (CONFIG v1.4 E-F0)")
        result[str(sign)] = aggregate
    if min_separation is not None:
        agg_keys = [k for k in result if k in ("1", "-1")]
        if len(agg_keys) == 2:
            sep = result["1"] - result["-1"]
            if sep < min_separation:
                raise ValueError(
                    f"Checkerboard per-sign separation too weak: "
                    f"median+ - median- = {sep:.3f} < {min_separation} (AG-2 esc#8)")
            result["separation"] = float(sep)
    return result
