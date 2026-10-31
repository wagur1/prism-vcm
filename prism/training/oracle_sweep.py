"""Train-only K-map oracle sweeps with locked distortion and encode provenance."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from prism.data.splits import SplitLedger
from .manifest import EncodeManifest


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
        maps = [np.asarray(candidate, dtype=np.float32) for candidate in candidates]
        if any(candidate.ndim != 2 or candidate.shape != maps[0].shape for candidate in maps):
            raise ValueError("Candidate maps must share one two-dimensional grid")
        if len({candidate.tobytes() for candidate in maps}) != len(maps):
            raise ValueError("Candidate maps must be distinct")
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
