"""Area-aware QG16 projections; adaptive rate compensation is an RD hypothesis.

The exponential rate model is a proxy, not measured x265 rate control. Keep the
legacy zero-mean projection available for reproducing the Step 3 anchor.
"""
from __future__ import annotations

import numpy as np

from .x265_bridge import qg_shape


# Extra QP margin at each CRF, multiplied by the positive-QP area fraction.
# These are screening defaults, not parameters fitted on val2017.
RATE_OFFSET_SCHEDULE = ((20., 1.0), (24., .85), (28., .65), (32., .25), (36., .10))


def qg_area_weights(width: int = 1920, height: int = 1080) -> np.ndarray:
    """Return per-QG valid pixel weights on the CTU64-padded QG16 grid (68x120 for 1080p)."""
    rows, cols = qg_shape(width, height)
    y0 = np.arange(rows, dtype=np.float64) * 16.0
    y1 = np.minimum(y0 + 16.0, float(height))
    row_w = np.clip((y1 - y0) / 16.0, 0.0, 1.0)
    x0 = np.arange(cols, dtype=np.float64) * 16.0
    x1 = np.minimum(x0 + 16.0, float(width))
    col_w = np.clip((x1 - x0) / 16.0, 0.0, 1.0)
    return (row_w[:, None] * col_w[None, :]).astype(np.float32)


def _inputs(raw, mask, weights, low, high, quantize_int8):
    q = np.asarray(raw, dtype=np.float64)
    active = np.ones(q.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    w = np.ones(q.shape, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if (q.ndim != 2 or not q.size or active.shape != q.shape or w.shape != q.shape
            or not np.isfinite(q).all() or not np.isfinite(w).all() or np.any(w < 0)):
        raise ValueError("Require finite 2D QG map, matching mask and nonnegative weights")
    if not np.isfinite([low, high]).all() or not low <= 0 <= high or low == high:
        raise ValueError("Bounds must be finite, ordered and contain zero")
    if quantize_int8 and (low < -128 or high > 127):
        raise ValueError("Bounds exceed int8 range")
    return q, active, np.where(active, w, 0.0)


def _output(projected, low, high, quantize_int8):
    if not quantize_int8:
        return np.ascontiguousarray(projected, dtype=np.float32)
    return np.ascontiguousarray(np.rint(projected).clip(np.ceil(low), np.floor(high)), dtype=np.int8)


def project_zero_mean_qp_map(
    raw_qg_map: np.ndarray, *, mask: np.ndarray | None = None,
    weights: np.ndarray | None = None, low: float = -6.0, high: float = 6.0,
    quantize_int8: bool = True,
) -> np.ndarray:
    """Constrained zero-mean projection under box bounds [low, high].

    Solves for shift lambda via bisection such that:
        sum_{g in mask} w_g * clip(q_g - lambda, low, high) == 0
    outside mask is set to 0. Int8 rounding may perturb the weighted mean.
    """
    q, active, w_active = _inputs(raw_qg_map, mask, weights, low, high, quantize_int8)
    total_w = float(w_active.sum())
    if total_w <= 0.0 or not np.any(active):
        return np.zeros(q.shape, dtype=np.int8 if quantize_int8 else np.float32)

    lo_shift, hi_shift = float(q[active].min() - high), float(q[active].max() - low)
    for _ in range(36):
        mid = 0.5 * (lo_shift + hi_shift)
        clipped = np.clip(q - mid, low, high)
        weighted_sum = float((w_active * clipped).sum())
        if weighted_sum > 0.0:
            lo_shift = mid
        else:
            hi_shift = mid
    lam = 0.5 * (lo_shift + hi_shift)
    projected = np.where(active, np.clip(q - lam, low, high), 0.0)
    return _output(projected, low, high, quantize_int8)


def project_adaptive_qp_map(
    raw_qg_map: np.ndarray, *, crf: float = 28.0,
    mask: np.ndarray | None = None, weights: np.ndarray | None = None,
    low: float = -4.0, high: float = 6.0,
    delta_rate_offset: float | None = None, quantize_int8: bool = False,
) -> np.ndarray:
    """Center spatial controls, compensate convex rate, then add a CRF margin.

    First solve the box-constrained area-zero-mean field b. Then solve for a
    nonnegative DC shift s with sum(w * 2**(-clip(b+s)/6))/sum(w) == 1.
    This compensates the Jensen penalty, including saturation at the bounds.
    Finally add delta_rate_offset, or a CRF-interpolated margin scaled by the
    positive area fraction. Higher CRFs get less extra background pressure.

    Supply qg_area_weights(width, height) to exclude CTU padding; zero-weight
    and unobserved cells stay zero. Output has the input QG shape and is C-order
    float32 (recommended) or int8. Rounding can change the rate proxy in int8
    mode; neither mode guarantees real codec savings or task preservation.
    """
    q, active, w = _inputs(raw_qg_map, mask, weights, low, high, quantize_int8)
    if not np.isfinite(crf) or not 0 <= crf <= 51:
        raise ValueError("CRF must be finite and in [0, 51]")
    if delta_rate_offset is not None and (
            not np.isfinite(delta_rate_offset) or delta_rate_offset < 0):
        raise ValueError("delta_rate_offset must be finite and nonnegative")
    active = active & (w > 0)
    if not active.any():
        return np.zeros(q.shape, dtype=np.int8 if quantize_int8 else np.float32)
    base = project_zero_mean_qp_map(q, mask=active, weights=w, low=low, high=high,
                                    quantize_int8=False).astype(np.float64)
    # Uniform predictions carry no spatial evidence. Remove numerical bisection dust.
    base[np.abs(base) < 1e-7] = 0.0
    valid, mass = base[active], w[active] / w[active].sum()
    lo_shift, hi_shift = 0., max(0., -float(valid.min()))
    for _ in range(48):
        mid = (lo_shift + hi_shift) / 2
        ratio = float(np.dot(mass, np.exp2(-np.clip(valid + mid, low, high) / 6)))
        if ratio > 1:
            lo_shift = mid
        else:
            hi_shift = mid
    if delta_rate_offset is None:
        knots, margins = np.asarray(RATE_OFFSET_SCHEDULE).T
        background_fraction = float(mass[valid > 0].sum())
        delta_rate_offset = float(np.interp(crf, knots, margins)) * background_fraction
    shift = (lo_shift + hi_shift) / 2 + delta_rate_offset
    result = np.where(active, np.clip(base + shift, low, high), 0.0)
    return _output(result, low, high, quantize_int8)
