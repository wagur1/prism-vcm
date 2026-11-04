"""Opt-in, calibration-gated luma residuals in 8-bit sample units.

This inference helper neither resizes predictions nor changes QP controls. Supply
native-resolution residuals after interpolation. Missing, expired, or unmatched
calibration leaves the packed YUV420p8 payload byte-identical.
"""
from __future__ import annotations

import numpy as np

from prism.training.calibration import CalibrationGate


def _validate_pixel_inputs(payload, residual, width, height, cap):
    if any(isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer))
           or d <= 0 or d % 2 for d in (width, height)):
        raise ValueError("Frame dimensions must be positive even integers")
    width, height = int(width), int(height)
    if not isinstance(payload, bytes) or len(payload) != width * height * 3 // 2:
        raise ValueError("Expected immutable payload with exactly H*W*3/2 YUV420p8 bytes")
    if (isinstance(cap, (bool, np.bool_)) or not isinstance(cap, (int, np.integer))
            or not 0 <= cap <= 3):
        raise ValueError("cap must be an integer in [0, 3] luma levels")
    raw = np.asarray(residual)
    if raw.shape != (height, width) or raw.dtype.kind not in "iuf":
        raise ValueError("Expected real numeric residual at native [H,W] resolution")
    return raw, width, height, int(cap)


def _apply_residual_pixels(
    payload: bytes, residual: np.ndarray, width: int, height: int, *, cap: int,
) -> tuple[bytes, dict]:
    """Pure pixel transform for explicitly approved research collectors only.

    This internal helper provides no calibration authorization; inference callers
    must use apply_bounded_residual. Supply native luma-level predictions after
    caller-owned resizing, so clipping happens after interpolation. All values
    must be finite, including at cap=0. Statistics describe only luma pixels.
    Nearest-even rounding precedes [0,255] saturation; chroma is unchanged.
    """
    raw, width, height, cap = _validate_pixel_inputs(payload, residual, width, height, cap)
    if not np.isfinite(raw).all():
        raise ValueError("Residual must contain only finite predictions")
    n = width * height
    # Clip in the input dtype before float conversion, including int64/uint64.
    bounded = np.clip(raw, -cap, cap).astype(np.float64)
    luma = np.frombuffer(payload, np.uint8, count=n).reshape(height, width)
    rounded = np.rint(luma.astype(np.float64) + bounded)
    changed = np.clip(rounded, 0, 255).astype(np.uint8)
    # Signed subtraction avoids uint8 wraparound when luma decreases.
    delta = changed.astype(np.int16) - luma.astype(np.int16)
    stats = {
        "cap": cap, "num_luma_pixels": n,
        "changed_pixel_fraction": float(np.mean(delta != 0)),
        "min_pixel_delta": int(delta.min()), "max_pixel_delta": int(delta.max()),
        "max_abs_pixel_delta": int(np.abs(delta).max()),
        "mean_pixel_delta": float(delta.mean()),
        "mean_abs_pixel_delta": float(np.abs(delta).mean()),
        "residual_clipped_fraction": float(np.mean((raw < -cap) | (raw > cap))),
        "saturation_fraction": float(np.mean((rounded < 0) | (rounded > 255))),
    }
    return changed.tobytes() + payload[n:], stats


def apply_bounded_residual(
    payload: bytes,
    residual: np.ndarray,
    width: int,
    height: int,
    *,
    cap: int = 0,
    gate: CalibrationGate | None = None,
    step: int = 0,
    qp: int | None = None,
    frame_type: str = "unknown",
) -> tuple[bytes, dict]:
    """Apply an authorized residual to luma only and report actual pixel changes.

    Integer caps 0..3 bound changes even after nearest-even rounding to uint8.
    Clipping uses the legacy [0,255] sample range, not a limited-range legalization
    pass. Chroma bytes remain untouched. All statistics use luma pixels as their
    denominator; ``applied`` means authorized, not necessarily nonzero output.

    ``step`` is the calibration training step, not the video frame index. ``qp``
    and ``frame_type`` must match a cell in the currently verified gate. The caller
    must bind that gate to the actual checkpoint, codec, and calibration data;
    this helper cannot establish experimental provenance from an in-memory gate.

    Nonfinite predictions raise only on the active path. Disabled or unverified
    paths suppress them, preserving CalibrationGate's fail-closed behavior.
    Structural input errors are always rejected, including when cap is zero.
    """
    raw, width, height, cap = _validate_pixel_inputs(payload, residual, width, height, cap)
    n = width * height
    if (isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer))
            or step < 0):
        raise ValueError("step must be a nonnegative integer training step")
    if qp is not None and (isinstance(qp, (bool, np.bool_))
                           or not isinstance(qp, (int, np.integer)) or not 0 <= qp <= 51):
        raise ValueError("qp must be an integer in [0, 51] or None")
    if not isinstance(frame_type, str) or not frame_type.strip():
        raise ValueError("frame_type must be a nonempty calibration label")
    if gate is not None and not isinstance(gate, CalibrationGate):
        raise ValueError("gate must be a CalibrationGate or None, not a report or flag")
    step = int(step)
    qp = None if qp is None else int(qp)

    reason = None
    if cap == 0:
        reason = "cap_zero"
    elif gate is None:
        reason = "missing_calibration"
    elif not gate.allow_delta_x(step):
        reason = "unverified_calibration"
    elif qp is None or (qp, frame_type) not in gate.expected_cells:
        reason = "uncalibrated_cell"
    stats = {
        "cap": cap, "step": step, "calibration_qp": qp, "frame_type": frame_type,
        "applied": reason is None, "bypass_reason": reason, "num_luma_pixels": n,
        "changed_pixel_fraction": 0.0, "min_pixel_delta": 0, "max_pixel_delta": 0,
        "max_abs_pixel_delta": 0, "mean_pixel_delta": 0.0, "mean_abs_pixel_delta": 0.0,
        "residual_clipped_fraction": 0.0, "saturation_fraction": 0.0,
    }
    if reason is not None:
        return payload, stats
    if not np.isfinite(raw).all():
        raise ValueError("Active residual must contain only finite predictions")

    authorized = gate.apply_delta_x(raw, step)
    result, pixel_stats = _apply_residual_pixels(payload, authorized, width, height, cap=cap)
    stats.update(pixel_stats)
    return result, stats
