"""CONFIG B: smooth, project, recheck rounding, then immediate breaker override."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.ndimage import map_coordinates

RateDelta = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class MotionStats:
    frame_index: int
    vectors: np.ndarray  # Backward sampling displacement in full-resolution luma pixels.
    source_height: int
    source_width: int
    extraction_ms: float = 0.0


@dataclass
class ControlResult:
    delta_qp: np.ndarray
    residual: np.ndarray
    log: dict[str, float | int | bool | str]


def warp(previous: np.ndarray, motion: MotionStats) -> np.ndarray:
    vectors = np.asarray(motion.vectors, dtype=np.float64)
    if vectors.shape != (*previous.shape, 2) or not np.isfinite(vectors).all():
        raise ValueError("Motion must be a finite [map_H,map_W,2] field")
    if motion.source_height <= 0 or motion.source_width <= 0:
        raise ValueError("Invalid motion source dimensions")
    yy, xx = np.indices(previous.shape, dtype=np.float64)
    return map_coordinates(previous,
        [yy + vectors[..., 1] * previous.shape[0] / motion.source_height,
         xx + vectors[..., 0] * previous.shape[1] / motion.source_width],
        order=1, mode="nearest")


class Breaker:
    """Small luma difference + phase-shift motion proxy, without future frames."""
    def __init__(self, scene_cut_threshold: float = 0.35):
        # CONFIG does not fix the photometric scene-cut threshold: freeze after pilot.
        self.scene_cut_threshold = scene_cut_threshold
        self.previous: np.ndarray | None = None
        self.hold_remaining = 0

    def detect(self, luma: np.ndarray, scene_cut: bool = False,
               velocity_1080p: float | None = None) -> tuple[bool, dict]:
        start = perf_counter()
        if luma.ndim != 2 or not np.isfinite(luma).all():
            raise ValueError("Expected finite 2D luma in pixel units [0,255]")
        yi = np.linspace(0, luma.shape[0] - 1, 36).astype(int)
        xi = np.linspace(0, luma.shape[1] - 1, 64).astype(int)
        small = luma[np.ix_(yi, xi)].astype(np.float64) / 255.0
        difference, velocity = 0.0, 0.0
        if self.previous is not None:
            difference = float(np.mean(np.abs(small - self.previous)))
            cross = np.fft.fft2(small) * np.conj(np.fft.fft2(self.previous))
            cross /= np.maximum(np.abs(cross), 1e-12)
            y, x = np.unravel_index(np.argmax(np.fft.ifft2(cross).real), small.shape)
            y = y if y <= 18 else y - 36
            x = x if x <= 32 else x - 64
            velocity = float(np.hypot(x * (luma.shape[1] / luma.shape[0]) * 1080 / 64,
                                      y * 1080 / 36))
        if velocity_1080p is not None:
            if not np.isfinite(velocity_1080p) or velocity_1080p < 0:
                raise ValueError("Invalid causal velocity")
            velocity = velocity_1080p
        trigger = scene_cut or difference > self.scene_cut_threshold or velocity > 15.0
        if trigger:
            self.hold_remaining = max(self.hold_remaining, 3)
        active = self.hold_remaining > 0
        if active:
            self.hold_remaining -= 1
            self.previous = None  # Flush photometric history for the entire hold.
        else:
            self.previous = small
        elapsed = (perf_counter() - start) * 1000
        return active, {"breaker_trigger": trigger, "breaker_active": active,
            "frame_difference": difference, "velocity_1080p": velocity,
            "detector_ms": elapsed, "detector_budget_exceeded": elapsed > 0.3}


def project_map(raw: np.ndarray, background: np.ndarray, base_qp: float,
                budget: RateDelta | None) -> tuple[np.ndarray, bool]:
    """Constant shift within simultaneous bounds; shrink foreground if infeasible.

    budget returns sum(delta bits) in a calibrated measured-rate domain. It must
    include all relevant frame costs; this is not an arithmetic sum of QP deltas.
    A non-monotone rate callback can cause a conservative bypass, never a violation.
    """
    low = np.full(raw.shape, max(-6.0, -base_qp))
    high = np.minimum(np.where(background, 4.0, 6.0), 51.0 - base_qp)
    # Integer-domain bounds also enforce valid QP for fractional base values.
    low, high = np.ceil(low), np.floor(high)

    def rounded(values: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(np.clip(values, low, high)), low, high)

    candidate = rounded(raw)
    if budget is None:
        return candidate, False

    def feasible(values: np.ndarray) -> bool:
        delta = float(budget(values))
        return np.isfinite(delta) and delta <= 0.0

    if feasible(candidate):
        return candidate, False
    # Do not saturate background past its floor to buy foreground quality.
    # Search feasible constant shifts before reducing the foreground preference.
    for factor in (1.0, 0.75, 0.5, 0.25, 0.0):
        values = np.clip(np.where(background, raw, raw * factor), low, high)
        max_shift = max(0.0, float(np.min(high - values)))
        if not feasible(rounded(values + max_shift)):
            continue
        left, right = 0.0, max_shift
        for _ in range(32):
            mid = (left + right) / 2
            if feasible(rounded(values + mid)):
                right = mid
            else:
                left = mid
        candidate = rounded(values + right)
        if feasible(candidate):  # Mandatory measured-domain check after round().
            return candidate, False
    return np.zeros_like(raw), True


class ControlContract:
    def __init__(self, zero_sum: bool = False, scene_cut_threshold: float = 0.35):
        self.breaker = Breaker(scene_cut_threshold)
        self.zero_sum = zero_sum  # Ablation only, never implicit for the primary CQP claim.
        self.previous_map: np.ndarray | None = None
        self.previous_index: int | None = None

    def apply(self, raw_map: np.ndarray, residual: np.ndarray, luma: np.ndarray,
              background: np.ndarray, *, frame_index: int, base_qp: float,
              frame_type: str = "unknown", rc_mode: str = "CQP",
              previous_motion: MotionStats | None = None,
              measured_delta_bits: RateDelta | None = None,
              scene_cut: bool = False, velocity_1080p: float | None = None) -> ControlResult:
        raw = np.asarray(raw_map, dtype=np.float64)
        bg = np.asarray(background, dtype=bool)
        dx = np.asarray(residual, dtype=np.float64)
        if raw.ndim != 2 or raw.shape != bg.shape or not np.isfinite(raw).all() or not np.isfinite(dx).all():
            raise ValueError("Invalid controller heads or background grid")
        if not 0 <= base_qp <= 51 or rc_mode not in {"CQP", "CRF", "ABR", "VBV"}:
            raise ValueError("Invalid base QP or RC mode")
        if self.previous_index is not None and frame_index != self.previous_index + 1:
            raise ValueError("Control requires contiguous causal frame indices")
        if previous_motion is not None and previous_motion.frame_index != frame_index - 1:
            raise ValueError("ST-02: motion must come from the immediately previous encoder frame")
        active, log = self.breaker.detect(luma, scene_cut, velocity_1080p)
        start = perf_counter()
        history = self.previous_map if self.previous_map is not None else np.zeros_like(raw)
        if previous_motion is not None and not active:
            history = warp(history, previous_motion)
        smoothed = 0.7 * history + 0.3 * raw
        log["warp_ms"] = (perf_counter() - start) * 1000
        log["mv_lag_frames"] = 1
        log["mv_missing"] = previous_motion is None
        log["mv_extraction_ms"] = previous_motion.extraction_ms if previous_motion else 0.0
        budget_on = self.zero_sum and rc_mode in {"ABR", "VBV"}
        budget = measured_delta_bits if budget_on else None
        if budget_on and budget is None and not active:
            raise ValueError("ABR/VBV projection requires a measured-rate domain callback")
        projected, infeasible = project_map(smoothed, bg, base_qp, budget if not active else None)
        limit = 8.0 if frame_type == "non_reference_B" else 3.0
        bounded_dx = np.clip(dx, -limit, limit)
        if infeasible:
            self.breaker.hold_remaining = max(self.breaker.hold_remaining, 2)
            self.breaker.previous = None
        bypass = active or infeasible
        log.update({"frame_index": frame_index, "rc_mode": rc_mode,
            "boundary_hit_fraction": float(np.mean((projected <= max(-6, -base_qp)) |
                 (projected >= np.minimum(np.where(bg, 4, 6), 51 - base_qp)))),
            "residual_boundary_hit_fraction": float(np.mean(np.abs(dx) >= limit)),
            "budget_on": budget_on, "budget_infeasible": infeasible, "bypass": bypass,
            "bypass_fraction": float(bypass)})
        if bypass:
            projected = np.zeros_like(raw)
            bounded_dx = np.zeros_like(dx)
            self.previous_map = None
        else:
            self.previous_map = smoothed.copy()
        log["breaker_active"] = bypass
        self.previous_index = frame_index
        return ControlResult(projected.astype(np.float32), bounded_dx.astype(np.float32), log)
