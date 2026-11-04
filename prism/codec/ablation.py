"""Plan B controlled QG16 ablations with identical float32 offset precision."""
from __future__ import annotations

import numpy as np

from .qp_maps import project_adaptive_qp_map, project_zero_mean_qp_map, qg_area_weights
from .x265_bridge import qg_shape


def generate_ablation_offsets(
    spatial_qp: np.ndarray,
    variant: str,
    crf: float,
    *,
    width: int = 1920,
    height: int = 1080,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, bool]:
    """Return ``(quant_offsets, apply_prefilter)`` for one controlled ablation.

    ``spatial_qp`` is the unrounded student field, already sampled on the exact
    QG16 grid covering the CTU64-padded frame (68x120 for 1920x1080). Maps are
    never resized or implicitly padded. Centering uses valid pixel area, so
    partial edge QGs contribute proportionally; fully padded and masked cells
    receive zero offsets and do not influence the projection.

    Variants:
      * anchor / M1: no offsets, no prefilter.
      * M2_zero_mean: box-constrained zero mean within [-6, +6].
      * M2_asym: box-constrained zero mean within [-4, +6], no rate shift.
      * M2_asym_ratecomp: M2_asym plus the nonnegative DC shift that makes the
        area-weighted rate proxy mean(2**(-offset/6)) equal to one, with clipping.
      * M2_filter_only: no offsets, prefilter enabled.
      * M2_asym_prefilter: identical offsets to M2_asym_ratecomp, prefilter enabled.

    The extra CRF-dependent margin is disabled in every variant, isolating rate
    compensation from that additional mechanism. The proxy does not guarantee
    actual codec rate neutrality. Every returned map is finite, C-contiguous
    float32; integer quantization is deliberately unavailable. This function
    only selects the prefilter flag; the caller applies background smoothing.
    ``spatial_qp`` and ``mask`` are unused in variants that return no offsets.
    """
    if variant not in (
        "anchor", "M1", "M2_zero_mean", "M2_asym", "M2_asym_ratecomp",
        "M2_filter_only", "M2_asym_prefilter",
    ):
        raise ValueError(f"Unknown ablation variant: {variant!r}")
    if not np.isfinite(crf) or not 0 <= crf <= 51:
        raise ValueError("CRF must be finite and in [0, 51]")
    if any(isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer))
           for d in (width, height)):
        raise ValueError("Frame dimensions must be positive even integers")
    expected_shape = qg_shape(width, height)
    apply_prefilter = variant in ("M2_filter_only", "M2_asym_prefilter")
    if variant in ("anchor", "M1", "M2_filter_only"):
        return None, apply_prefilter

    raw = np.asarray(spatial_qp)
    if raw.shape != expected_shape:
        raise ValueError(
            f"Expected QG16 shape {expected_shape} on the CTU64-padded domain, "
            f"got {raw.shape}")
    if (not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw)
            or not np.isfinite(raw).all()):
        raise ValueError("spatial_qp must contain finite real numeric offsets")
    weights = qg_area_weights(width, height)
    active = weights > 0
    if mask is not None:
        known = np.asarray(mask, dtype=bool)
        if known.shape != expected_shape:
            raise ValueError("mask must match the padded QG16 shape")
        active &= known

    low = -6.0 if variant == "M2_zero_mean" else -4.0
    if variant in ("M2_asym_ratecomp", "M2_asym_prefilter"):
        # Zero extra margin leaves the adaptive helper's Jensen compensation on.
        offsets = project_adaptive_qp_map(
            raw, crf=crf, mask=active, weights=weights, low=low, high=6.0,
            delta_rate_offset=0.0, quantize_int8=False)
    else:
        # Bypass the adaptive helper entirely: M2_asym has no rate compensation.
        offsets = project_zero_mean_qp_map(
            raw, mask=active, weights=weights, low=low, high=6.0,
            quantize_int8=False)

    offsets = np.ascontiguousarray(offsets, dtype=np.float32)
    if offsets.shape != expected_shape or not np.isfinite(offsets).all():
        raise ValueError("Projection must produce finite float32 padded QG16 offsets")
    return offsets, apply_prefilter
