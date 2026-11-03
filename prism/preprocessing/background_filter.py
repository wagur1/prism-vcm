"""Soft background Gaussian smoothing with exact foreground preservation.

No learned residual, color conversion, or data-dependent normalization is used.
QP maps refer to the original QG16 grid, including CTU64 padding. A saliency
map can be supplied explicitly with mask_kind='saliency' (1 = foreground).
"""
from __future__ import annotations

import numpy as np

from prism.codec.x265_bridge import qg_shape


def _soft_mask(grid, width, height, strength, mask_kind):
    grid = np.asarray(grid, dtype=np.float32)
    if grid.shape != qg_shape(width, height) or not np.isfinite(grid).all():
        raise ValueError("Expected finite QG16 map matching the CTU64-padded image")
    if mask_kind == "qp":
        # Fixed scale keeps +2 weak and +6 strong even on mostly foreground images.
        alpha = np.clip(grid / 6.0, 0.0, 1.0)
    elif mask_kind == "saliency":
        if np.any((grid < 0) | (grid > 1)):
            raise ValueError("Saliency must be in [0, 1], with 1 denoting foreground")
        alpha = 1.0 - grid
    else:
        raise ValueError("mask_kind must be 'qp' or 'saliency'")
    return np.clip(alpha * strength, 0, 1).repeat(16, 0).repeat(16, 1)[:height, :width]


def _smooth(plane, alpha, sigma):
    if not np.any(alpha):
        return plane.copy()
    radius = max(1, int(np.ceil(3 * sigma)))
    work_dtype = np.float64 if plane.dtype == np.float64 else np.float32
    x = np.arange(-radius, radius + 1, dtype=work_dtype)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    original = plane.astype(work_dtype)
    blurred = original
    for axis in (0, 1):
        pads = [(0, 0)] * plane.ndim
        pads[axis] = (radius, radius)
        padded = np.pad(blurred, pads, mode="edge")
        blurred = np.zeros_like(original)
        for index, value in enumerate(kernel):
            slices = [slice(None)] * plane.ndim
            slices[axis] = slice(index, index + plane.shape[axis])
            blurred += value * padded[tuple(slices)]
    mix = alpha if plane.ndim == 2 else alpha[..., None]
    result = original + mix * (blurred - original)
    result = np.clip(result, original.min(), original.max())
    if np.issubdtype(plane.dtype, np.integer):
        bounds = np.iinfo(plane.dtype)
        result = np.rint(result).clip(bounds.min, bounds.max)
    result = result.astype(plane.dtype)
    # Copy original values explicitly: protected pixels are bit-exact, even float64.
    result[alpha == 0] = plane[alpha == 0]
    return np.ascontiguousarray(result)


def apply_background_smoothing(yuv_or_rgb, qg_map, strength: float = 1.0, *,
                               sigma: float = 1.0, mask_kind: str = "qp"):
    """Smooth RGB/YUV444 [H,W,3], luma [H,W], or planar YUV420 (Y,U,V).

    Accept unsigned integer or floating arrays; preserve shape, dtype, range,
    and input ownership. Negative/zero QPs have alpha=0. A positive strength
    scales the soft blend, capped at one; sigma is the Gaussian radius scale
    in luma pixels. Bytes and packed I420 are intentionally rejected: unpack
    into planes first so geometry is never guessed. For YUV420, the chroma
    alpha is the minimum over its four luma pixels, protecting boundary color.
    """
    if not np.isfinite(strength) or strength < 0 or not np.isfinite(sigma) or not 0 < sigma <= 16:
        raise ValueError("Require finite strength >= 0 and sigma in (0, 16]")
    planar = isinstance(yuv_or_rgb, (tuple, list))
    planes = [np.asarray(p) for p in yuv_or_rgb] if planar else [np.asarray(yuv_or_rgb)]
    if (not planes or any(p.dtype.kind not in "uf" or not np.isfinite(p).all() for p in planes)):
        raise ValueError("Expected finite unsigned integer or floating image arrays")
    first = planes[0]
    if first.ndim not in (2, 3) or (first.ndim == 3 and first.shape[2] != 3):
        raise ValueError("Expected luma [H,W] or RGB/YUV444 [H,W,3]")
    height, width = first.shape[:2]
    alpha = _soft_mask(qg_map, width, height, strength, mask_kind)
    if planar:
        if (len(planes) != 3 or first.ndim != 2 or any(
                p.shape != (height // 2, width // 2) or p.dtype != first.dtype for p in planes[1:])):
            raise ValueError("Expected same-dtype planar YUV420 (Y[H,W],U[H/2,W/2],V[H/2,W/2])")
        chroma_alpha = alpha.reshape(height // 2, 2, width // 2, 2).min(axis=(1, 3))
        result = [_smooth(first, alpha, sigma)] + [
            _smooth(p, chroma_alpha, sigma / 2) for p in planes[1:]]
        return tuple(result) if isinstance(yuv_or_rgb, tuple) else result
    return _smooth(first, alpha, sigma)
