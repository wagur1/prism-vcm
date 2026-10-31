"""PCHIP integration on raw monotonic curves; invalid results remain JSON-safe."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class RDPoint:
    qp: int
    score: float
    bitstream_bytes: int  # Entire coded bitstream, including headers and syntax.


@dataclass(frozen=True)
class BDResult:
    percent: float | None
    valid: bool
    flag: str | None
    reason: str | None
    overlap: float
    per_qp: dict[str, list[dict]]


def whole_bitstream_bytes(bitstream: bytes | Path) -> int:
    return len(bitstream) if isinstance(bitstream, bytes) else bitstream.stat().st_size


def bd_rate(anchor: Sequence[RDPoint], candidate: Sequence[RDPoint]) -> BDResult:
    def safe_row(point: RDPoint) -> dict:
        return {key: value if np.isfinite(value) else None for key, value in asdict(point).items()}
    table = {"anchor": [safe_row(p) for p in anchor], "candidate": [safe_row(p) for p in candidate]}

    def invalid(reason: str, overlap: float = 0.0) -> BDResult:
        return BDResult(None, False, "[BD-INVALID-RAW]", reason, overlap, table)

    curves = []
    for points in (anchor, candidate):
        if len(points) < 2 or len({point.qp for point in points}) != len(points):
            return invalid("Need distinct QPs and at least two points")
        ordered = sorted(points, key=lambda point: point.qp, reverse=True)
        scores = np.array([point.score for point in ordered], float)
        rates = np.array([point.bitstream_bytes for point in ordered], float)
        if not np.isfinite(scores).all() or not np.isfinite(rates).all() or np.any(rates <= 0):
            return invalid("Nonfinite scores or nonpositive whole-bitstream bytes")
        if np.any(np.diff(scores) <= 0) or np.any(np.diff(rates) <= 0):
            return invalid("Non-monotonic raw per-QP score/rate curve")
        curves.append((scores, np.log(rates)))
    (a, ar), (b, br) = curves
    low, high = max(a[0], b[0]), min(a[-1], b[-1])
    # Pre-register overlap as intersection / larger task-score span (both curves >=70%).
    overlap = max(0.0, float((high - low) / max(a[-1] - a[0], b[-1] - b[0])))
    if high <= low or overlap < 0.70 - 1e-12:
        return invalid("Task-score overlap below 70%", overlap)
    mean_delta = (PchipInterpolator(b, br, extrapolate=False).integrate(low, high) -
                  PchipInterpolator(a, ar, extrapolate=False).integrate(low, high)) / (high - low)
    with np.errstate(over="ignore", invalid="ignore"):
        percent = float(np.expm1(mean_delta) * 100)
    if not np.isfinite(percent):
        return invalid("Unrepresentable BD result", overlap)
    return BDResult(percent, True, None, None, overlap, table)


def isotonic_pava_sensitivity(values: Sequence[float]) -> list[float]:
    """Sensitivity appendix only. Never called by bd_rate or any gate."""
    blocks: list[tuple[float, int]] = []
    for value in values:
        if not np.isfinite(value):
            raise ValueError("PAVA needs finite values")
        blocks.append((float(value), 1))
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            right, nr = blocks.pop()
            left, nl = blocks.pop()
            blocks.append(((left * nl + right * nr) / (nl + nr), nl + nr))
    return [mean for mean, count in blocks for _ in range(count)]


def cluster_bootstrap(data: dict[int, dict[str, np.ndarray]],
                      statistic: Callable[[np.ndarray], float] = lambda x: float(np.mean(x)),
                      *, unit: str, resamples: int = 1000, random_seed: int = 0) -> dict:
    """Paired differences by seed and cluster; keep whole images/sequences together.

    Each array contains paired candidate-minus-anchor observations for one unit.
    Resample independent seeds and common clusters; preserve all frames/QPs within
    a cluster. Compute equal-weight cluster/seed means, avoiding long-video bias.
    """
    if unit not in {"image", "sequence"} or len(data) < 3 or resamples < 1000:
        raise ValueError("Need >=3 independent seeds, >=1000 resamples, image/sequence units")
    seeds = sorted(data)
    clusters = sorted(data[seeds[0]])
    if len(clusters) < 2 or any(set(data[seed]) != set(clusters) for seed in seeds):
        raise ValueError("All independent seeds must share >=2 paired clusters")
    summaries = np.array([[statistic(np.asarray(data[seed][cluster])) for cluster in clusters]
                          for seed in seeds])
    if not np.isfinite(summaries).all():
        raise ValueError("Nonfinite paired cluster statistics")
    rng = np.random.default_rng(random_seed)
    draws = np.empty(resamples)
    for index in range(resamples):
        seed_ids = rng.integers(0, len(seeds), len(seeds))
        cluster_ids = rng.integers(0, len(clusters), len(clusters))
        draws[index] = summaries[np.ix_(seed_ids, cluster_ids)].mean()
    return {"estimate": float(summaries.mean()), "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "unit": unit, "independent_seeds": seeds, "clusters": len(clusters), "resamples": resamples}


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[dict]:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid p-values")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return [{"p": float(p), "adjusted_p": float(q), "reject": bool(q <= alpha)}
            for p, q in zip(values, adjusted)]
