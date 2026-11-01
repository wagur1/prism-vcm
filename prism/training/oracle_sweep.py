"""Train-only K-map oracle sweeps with locked distortion and encode provenance."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from prism.data.splits import SplitLedger
from .manifest import EncodeManifest


def validate_candidate_maps(candidates: Sequence[np.ndarray], *,
                            distinct: bool = True) -> list[np.ndarray]:
    """Shared codec contract; proposals may coincide on empty teacher masks."""
    maps = [np.asarray(candidate, dtype=np.float32) for candidate in candidates]
    if not maps or any(m.ndim != 2 or m.shape != maps[0].shape for m in maps):
        raise ValueError("Candidate maps must share one two-dimensional grid")
    if distinct and len({m.tobytes() for m in maps}) != len(maps):
        raise ValueError("Candidate maps must be distinct")
    if any(not np.isfinite(m).all() or np.any(np.abs(m) > 6)
           or not np.array_equal(m, np.rint(m)) for m in maps):
        raise ValueError("Oracle candidates must satisfy the encoding contract")
    return maps


def rank_task_candidates(scores: Sequence[float], bits: Sequence[int],
                         lambda_rate: float = .05) -> tuple[list[float], list[int]]:
    """Tier 1a requested utility: MAXIMIZE AP50 + lambda * (zero-map bits / bits).

    Council visibility: this differs from CONFIG C's MINIMIZED distortion +
    lambda * bits / R0. Keep that original OracleSweep behavior unchanged.
    Stable ties prefer the lower candidate index (zero-map first).
    """
    d, r = np.asarray(scores, dtype=float), np.asarray(bits, dtype=float)
    if (d.ndim != 1 or not len(d) or d.shape != r.shape or not np.isfinite(d).all()
            or np.any((d < 0) | (d > 1)) or not np.isfinite(r).all()
            or np.any(r <= 0) or not np.isfinite(lambda_rate) or lambda_rate < 0):
        raise ValueError("Require finite AP in [0,1], positive rates, lambda >= 0")
    objective = d + lambda_rate * r[0] / r
    if not np.isfinite(objective).all():
        raise ValueError("Nonfinite oracle objective")
    return objective.tolist(), sorted(range(len(d)), key=lambda i: (-objective[i], i))


@dataclass(frozen=True)
class OracleConfig:
    distortion_name: str
    lambda_rate: float
    rate_reference: float
    candidates: int = 8

    def __post_init__(self) -> None:
        if not 8 <= self.candidates <= 16 or self.rate_reference <= 0 or self.lambda_rate < 0:
            raise ValueError("Oracle requires K=8..16, R0>0 and lambda>=0")
        if not self.distortion_name:
            raise ValueError("Lock the distortion definition before sweeps")


@dataclass(frozen=True)
class OracleLabel:
    sample_id: str
    delta_x_sha256: str
    delta_qp: np.ndarray
    objective: float
    distortion: float
    bits: int
    config: OracleConfig


class OracleSweep:
    def __init__(self, config: OracleConfig, ledger: SplitLedger, manifest: EncodeManifest):
        self.config, self.ledger, self.manifest = config, ledger, manifest
        self.labels: dict[str, OracleLabel] = {}

    def label(self, sample_id: str, delta_x_sha256: str,
              candidates: Sequence[np.ndarray],
              encode_and_score: Callable[[np.ndarray], tuple[float, int]],
              is_feasible: Callable[[np.ndarray], bool]) -> OracleLabel:
        self.ledger.require(sample_id, "train")
        if len(delta_x_sha256) != 64 or any(c not in "0123456789abcdef" for c in delta_x_sha256):
            raise ValueError("Hash the actual delta-x model/state for label freshness")
        old = self.labels.get(sample_id)
        if old and old.delta_x_sha256 == delta_x_sha256:
            return old
        if len(candidates) != self.config.candidates:
            raise ValueError("Supply the locked number of feasible oracle candidate maps")
        maps = validate_candidate_maps(candidates)
        for candidate in maps:
            if (not np.isfinite(candidate).all() or np.any(np.abs(candidate) > 6) or
                    not np.array_equal(candidate, np.rint(candidate)) or not is_feasible(candidate)):
                raise ValueError("Oracle candidates must satisfy the encoding contract")
        best = None
        for index, candidate in enumerate(maps):
            self.manifest.record("oracle_refresh" if old else "oracle_initial", sample_id,
                                 candidate=index, delta_x_sha256=delta_x_sha256,
                                 oracle_config=self.config.__dict__)
            # The callback must run teachers on the NEW reconstruction; input caches
            # cannot substitute for this measurement when delta-x or the map changes.
            distortion, bits = encode_and_score(candidate)
            if not np.isfinite(distortion) or bits <= 0:
                raise ValueError("Invalid measured oracle distortion/rate")
            objective = distortion + self.config.lambda_rate * bits / self.config.rate_reference
            if best is None or objective < best.objective:
                best = OracleLabel(sample_id, delta_x_sha256, candidate.copy(),
                                   objective, distortion, bits, self.config)
        assert best is not None
        self.labels[sample_id] = best
        return best
