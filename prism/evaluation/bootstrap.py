"""Paired image/sequence bootstrap that recomputes raw PCHIP BD on every draw."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .metrics import RDPoint, bd_rate


def paired_bd_bootstrap(data: dict[int, dict[str, tuple[list[RDPoint], list[RDPoint]]]],
                        *, unit: str, resamples: int = 1000, random_seed: int = 0) -> dict:
    """For additive per-unit task scores; aggregate detections externally for mAP.

    Inputs are seed -> cluster -> (anchor, candidate), with matched QPs and full
    bitstream bytes. Scores are equally weighted per unit; all frames in a video
    must already belong to its single sequence record. Non-additive metrics such
    as pooled AP require a task-specific raw-prediction aggregator instead.
    """
    if len(data) < 3 or resamples < 1000 or unit not in {"image", "sequence"}:
        raise ValueError("Need >=3 independent seeds and >=1000 image/sequence resamples")
    seeds = sorted(data)
    clusters = sorted(data[seeds[0]])
    if len(clusters) < 2 or any(set(data[seed]) != set(clusters) for seed in seeds):
        raise ValueError("Paired cluster identities must match across all seeds")
    qps = sorted(point.qp for point in data[seeds[0]][clusters[0]][0])
    if len(set(qps)) != len(qps):
        raise ValueError("Repeated QP in raw observations")
    for seed in seeds:
        for cluster in clusters:
            for curve in data[seed][cluster]:
                if sorted(point.qp for point in curve) != qps:
                    raise ValueError("All paired curves need matching QPs")

    def estimate(seed_ids: np.ndarray, cluster_ids: np.ndarray):
        curves = []
        for side in (0, 1):
            combined = []
            observations = [data[seeds[int(i)]][clusters[int(j)]][side]
                            for i in seed_ids for j in cluster_ids]
            for qp in qps:
                points = [next(point for point in curve if point.qp == qp) for curve in observations]
                combined.append(RDPoint(qp, float(np.mean([p.score for p in points])),
                                        sum(p.bitstream_bytes for p in points)))
            curves.append(combined)
        return bd_rate(*curves)

    result = asdict(estimate(np.arange(len(seeds)), np.arange(len(clusters))))
    result.update(unit=unit, independent_seeds=seeds, clusters=len(clusters), resamples=resamples)
    if not result["valid"]:
        result["ci95"] = None
        return result
    rng = np.random.default_rng(random_seed)
    draws = []
    for _ in range(resamples):
        value = estimate(rng.integers(0, len(seeds), len(seeds)),
                         rng.integers(0, len(clusters), len(clusters)))
        if not value.valid:
            # Do not silently drop bad resamples or repair them with PAVA.
            result.update(valid=False, flag="[BD-INVALID-RAW]", ci95=None,
                          reason="A raw bootstrap draw has invalid BD; no gate decision")
            return result
        draws.append(value.percent)
    result["ci95"] = np.quantile(draws, [.025, .975]).tolist()
    return result
