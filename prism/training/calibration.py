"""Per-QP/frame-type checks and fail-closed delta-x fine-tuning state."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm, spearmanr, t


@dataclass(frozen=True)
class CalibrationCell:
    qp: int
    frame_type: str
    actual_bits: np.ndarray
    predicted_bits: np.ndarray
    actual_delta_j: np.ndarray
    predicted_delta_j: np.ndarray
    direction_delta_j: np.ndarray  # Real codec delta-x update, including dependent frames.
    semantic_passed: bool
    same_codec_state: bool
    feasible_plus_minus_one: bool
    dependent_frames_measured: bool
    split: str = "calibration"


def check_cell(cell: CalibrationCell) -> dict:
    arrays = [np.asarray(value, dtype=float) for value in
              (cell.actual_bits, cell.predicted_bits, cell.actual_delta_j,
               cell.predicted_delta_j, cell.direction_delta_j)]
    if cell.split != "calibration":
        raise ValueError("Calibration must be independent of train/dev/test")
    if any(value.ndim != 1 for value in arrays) or len({value.size for value in arrays}) != 1:
        raise ValueError("Expected equal-length paired calibration samples")
    n = arrays[0].size
    invalid = n < 100 or not all(np.isfinite(value).all() for value in arrays)
    if invalid or np.any(arrays[0] <= 0) or np.any(arrays[1] <= 0):
        return {"passed": False, "n": n, "reason": "Need >=100 finite samples and positive bits"}
    measured, predicted, actual_j, proxy_j, direction = arrays
    mape = float(np.mean(np.abs(predicted - measured) / measured))
    rho = 0.0 if np.ptp(actual_j) == 0 or np.ptp(proxy_j) == 0 else float(spearmanr(actual_j, proxy_j).statistic)
    # Fisher-z is the pre-registered approximate correlation CI, including Spearman.
    z = np.arctanh(np.clip(rho, -1 + 1e-12, 1 - 1e-12))
    interval = np.tanh(z + np.array([-1, 1]) * norm.ppf(0.975) / np.sqrt(n - 3))
    upper = float(np.mean(direction) + t.ppf(0.975, n - 1) * np.std(direction, ddof=1) / np.sqrt(n))
    passed = (mape <= 0.05 and rho >= 0.8 and upper < 0 and cell.semantic_passed and
              cell.same_codec_state and cell.feasible_plus_minus_one and cell.dependent_frames_measured)
    return {"passed": bool(passed), "n": n, "mape": mape, "spearman": rho,
            "fisher_z_ci95": interval.tolist(), "direction_mean_ci95_upper": upper}


def direction_misalignment(cells: Sequence[CalibrationCell]) -> bool:
    """F5: mismatched signs at >=2 consecutive evaluated QPs of the same type."""
    for frame_type in {cell.frame_type for cell in cells}:
        previous_bad = False
        for cell in sorted((c for c in cells if c.frame_type == frame_type), key=lambda c: c.qp):
            bad = np.sign(np.mean(cell.actual_delta_j)) != np.sign(np.mean(cell.predicted_delta_j))
            if previous_bad and bad:
                return True
            previous_bad = bool(bad)
    return False


class CalibrationGate:
    interval = 1000

    def __init__(self, expected_cells: set[tuple[int, str]]):
        if not expected_cells:
            raise ValueError("Pre-register every QP/frame-type calibration cell")
        self.expected_cells = frozenset(expected_cells)
        self.consecutive_failures = 0
        self.verified = False
        self.fine_tune_disabled = False
        self.last_step: int | None = None

    def assess(self, step: int, cells: Sequence[CalibrationCell],
               restore_supervised: Callable[[], None]) -> dict:
        if step < 0 or step % self.interval or (self.last_step is not None and step <= self.last_step):
            raise ValueError("Calibration must run at a fresh 1000-step boundary (including step 0)")
        if self.last_step is not None and step != self.last_step + self.interval:
            self.verified = False
            raise ValueError("A mandatory 1000-step calibration was missed")
        keys = [(cell.qp, cell.frame_type) for cell in cells]
        reports = [check_cell(cell) for cell in cells]
        passed = (len(keys) == len(set(keys)) and set(keys) == self.expected_cells and
                  all(report["passed"] for report in reports) and not direction_misalignment(cells))
        self.last_step = step
        self.verified = passed and not self.fine_tune_disabled
        self.consecutive_failures = 0 if passed else self.consecutive_failures + 1
        if self.consecutive_failures >= 2 and not self.fine_tune_disabled:
            # Disable first, so a failed restore cannot accidentally permit training.
            self.fine_tune_disabled = True
            self.verified = False
            restore_supervised()
        return {"passed": passed, "verified": self.verified, "cells": reports,
                "fine_tune_disabled": self.fine_tune_disabled}

    def allow_delta_x(self, step: int) -> bool:
        return bool(self.verified and not self.fine_tune_disabled and self.last_step is not None
                    and self.last_step <= step < self.last_step + self.interval)

    def apply_delta_x(self, residual: Any, step: int) -> Any:
        """Unverified delta-x is exactly zero, including nonfinite raw predictions."""
        if self.allow_delta_x(step):
            return residual
        return residual.new_zeros(residual.shape) if hasattr(residual, "new_zeros") else np.zeros_like(residual)
