"""Append-only accounting of every encode attempt, including failed encodes."""
from __future__ import annotations

import json
import math
from pathlib import Path
from time import time

PURPOSES = {"oracle_initial", "oracle_refresh", "calibration_probe", "evaluation", "gate0"}


class EncodeManifest:
    def __init__(self, path: Path, *, repo_sha: str, config_sha: str, build: dict, seed: int):
        self.path = path
        self.provenance = dict(repo_sha=repo_sha, config_sha=config_sha, build=build, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, purpose: str, sample_id: str, **details: object) -> None:
        if purpose not in PURPOSES:
            raise ValueError("Unknown encode purpose")
        row = dict(self.provenance, purpose=purpose, sample_id=sample_id,
                   time=time(), encodes=1, details=details)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def counts(self) -> dict[str, int]:
        counts = dict.fromkeys(sorted(PURPOSES), 0)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                counts[row["purpose"]] += row["encodes"]
        return counts


def validate_budget(*, gpu_hours: float, cpu_hours: float, job_hours: float,
                    working_gb: float, measured_intra_images: int) -> None:
    if measured_intra_images < 200:
        raise ValueError("Measure a 200-image intra pilot before extrapolating throughput")
    if not all(math.isfinite(v) and v >= 0 for v in (gpu_hours, cpu_hours, job_hours, working_gb)):
        raise ValueError("Budget estimates must be nonnegative")
    if gpu_hours > 60 or job_hours > 6 or working_gb >= 16:
        raise RuntimeError("CONFIG F budget exceeded: revise scope before running")
