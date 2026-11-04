"""Float-map diagnostics only; no codec, pixel-change, or performance claims."""
from itertools import combinations

import numpy as np

from prism.codec.qp_maps import (
    project_adaptive_qp_map, project_zero_mean_qp_map, qg_area_weights,
)


def summarize_numeric(values):
    """Population statistics of a finite, nonempty numeric array."""
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Require a finite nonempty numeric array")
    p01, p50, p99 = np.percentile(array, [1, 50, 99])
    return dict(min=float(array.min()), max=float(array.max()), mean=float(array.mean()),
                std=float(array.std()), p01=float(p01), p50=float(p50), p99=float(p99))


def bounded_output_summary(native_residual, cap):
    """Describe continuous residual maps, not rounded/applied luma differences."""
    if not np.isfinite(cap) or cap < 0:
        raise ValueError("cap must be finite and nonnegative")
    array = np.asarray(native_residual, dtype=np.float32)
    native = summarize_numeric(array)
    return dict(cap=float(cap), native=native,
                fraction_below_negative_cap=float(np.mean(array < -cap)),
                fraction_above_cap=float(np.mean(array > cap)),
                fraction_abs_ge_7_99=float(np.mean(np.abs(array) >= 7.99)),
                fraction_gt_half=float(np.mean(array > .5)),
                fraction_lt_negative_half=float(np.mean(array < -.5)),
                after_clip=summarize_numeric(np.clip(array, -cap, cap)))


def paired_qp_summary(raw, width, height):
    """Six fixed QG16 float32 controls and all 15 valid-area pair comparisons.

    OFF is the existing zero-mean projection; ON is the existing adaptive
    projection with delta_rate_offset=0 (no CRF-dependent margin). Padding is
    masked out. These are paired controls, not a cap search or a recommendation.
    """
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0
           for v in (width, height)):
        raise ValueError("Require positive integer width and height")
    weights = qg_area_weights(width, height)
    array = np.asarray(raw, dtype=np.float32)
    if array.shape != weights.shape:
        raise ValueError("raw must match the CTU64-padded QG16 geometry")
    summarize_numeric(array)
    valid = weights > 0
    mass = weights[valid].astype(np.float64)
    mass /= mass.sum()
    states, maps = {}, {}
    for name, low, high in (("sym6", -6., 6.), ("asym46", -4., 6.), ("sym5", -5., 5.)):
        for state, project in (("OFF", project_zero_mean_qp_map), ("ON", project_adaptive_qp_map)):
            kwargs = {"delta_rate_offset": 0.0} if state == "ON" else {}
            projected = project(array, mask=valid, weights=weights, low=low, high=high,
                                quantize_int8=False, **kwargs)
            key = f"{name}_{state}"
            values = projected[valid].astype(np.float64)
            mean = float(np.dot(mass, values))
            at_low, at_high = values <= low, values >= high
            states[key] = dict(low=low, high=high, dtype=str(projected.dtype),
                               area_weighted_mean=mean,
                               area_weighted_std=float(np.sqrt(np.dot(mass, (values - mean) ** 2))),
                               min=float(values.min()), max=float(values.max()),
                               fraction_at_low=float(np.dot(mass, at_low)),
                               fraction_at_high=float(np.dot(mass, at_high)),
                               fraction_at_cap=float(np.dot(mass, at_low | at_high)))
            maps[key] = values
    pairs = {}
    for left, right in combinations(maps, 2):
        difference = np.abs(maps[left] - maps[right])
        pairs[f"{left}__{right}"] = dict(area_weighted_mae=float(np.dot(mass, difference)),
                                       max_abs_difference=float(difference.max()),
                                       exact_equal=bool(np.array_equal(maps[left], maps[right])))
    return dict(shape=list(array.shape), delta_rate_offset=0.0,
                weighting="valid_pixel_area_excluding_padding", states=states, pairs=pairs)
