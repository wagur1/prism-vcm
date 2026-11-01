"""Gate evidence validation; average CSV QP is never spatial evidence."""
from __future__ import annotations

import csv
from pathlib import Path

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
            if not np.isfinite(qp) or not 0 <= qp <= 51:
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
