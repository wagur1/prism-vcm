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


def residual_trace(path: Path, width: int, height: int) -> dict[tuple[int, int, int], float]:
    """Read x265_prism_residual_v1: frame,x,y,qp,has_residual,skip (pixel coordinates)."""
    valid: dict[tuple[int, int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame, x, y = (int(row[key]) for key in ("frame", "x", "y"))
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


def verify_spatial_effect(anchor: dict[tuple[int, int, int], float],
                          changed: dict[tuple[int, int, int], float],
                          offsets: np.ndarray) -> dict[str, float]:
    """Require the realized residual QP direction in each requested sign region."""
    samples: dict[int, list[float]] = {-1: [], 1: []}
    per_frame: dict[tuple[int, int], list[float]] = {}
    for key in anchor.keys() & changed.keys():
        frame, x, y = key
        sign = int(np.sign(offsets[y // 16, x // 16]))
        if sign:
            delta = sign * (changed[key] - anchor[key])
            samples[sign].append(delta)
            per_frame.setdefault((frame, sign), []).append(delta)
    result: dict[str, float] = {}
    for sign in (-1, 1):
        if np.any(np.sign(offsets) == sign):
            for frame in {key[0] for key in anchor}:
                evidence = per_frame.get((frame, sign), [])
                if not evidence or np.median(evidence) <= 0:
                    raise ValueError(f"No realized QP effect in frame {frame}, sign {sign}")
            if not samples[sign] or np.median(samples[sign]) <= 0:
                raise ValueError(f"No positive realized QP effect for offset sign {sign}")
            result[str(sign)] = float(np.median(samples[sign]))
    return result
