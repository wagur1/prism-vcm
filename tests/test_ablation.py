"""Controlled ablations share geometry and precision, changing one factor at a time."""
from __future__ import annotations

import numpy as np
import pytest

from prism.codec.ablation import generate_ablation_offsets
from prism.codec.qp_maps import qg_area_weights


OFFSET_VARIANTS = (
    "M2_zero_mean", "M2_asym", "M2_asym_ratecomp", "M2_asym_prefilter",
)
ALL_VARIANTS = ("anchor", "M1", *OFFSET_VARIANTS, "M2_filter_only")


@pytest.mark.parametrize("variant", OFFSET_VARIANTS)
@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
@pytest.mark.parametrize("width,height,shape", [
    (64, 64, (4, 4)), (66, 66, (8, 8)),
    (1280, 720, (48, 80)), (1920, 1080, (68, 120)),
])
def test_float32_exact_padded_shape_and_finite(variant, dtype, width, height, shape):
    raw = np.random.default_rng(10).normal(0, 8, shape).astype(dtype)[:, ::-1]
    offsets, prefilter = generate_ablation_offsets(
        raw, variant, 28, width=width, height=height)
    assert offsets.shape == shape
    assert offsets.dtype == np.float32
    assert offsets.flags.c_contiguous
    assert np.isfinite(offsets).all()
    assert prefilter is (variant == "M2_asym_prefilter")


@pytest.mark.parametrize("variant,prefilter", [
    ("anchor", False), ("M1", False), ("M2_filter_only", True),
])
def test_no_quant_offsets_for_anchor_or_filter_only(variant, prefilter):
    result = generate_ablation_offsets(np.ones((68, 120)), variant, 28)
    assert result == (None, prefilter)


@pytest.mark.parametrize("variant", OFFSET_VARIANTS)
def test_fractional_offsets_are_never_integer_rounded(variant):
    raw = np.tile([-1.75, -0.25, 0.25, 1.75], (4, 1))
    offsets, _ = generate_ablation_offsets(raw, variant, 28, width=64, height=64)
    assert np.any(np.abs(offsets - np.rint(offsets)) > 0.1)
    if variant in ("M2_zero_mean", "M2_asym"):
        np.testing.assert_allclose(offsets, raw, atol=1e-7, rtol=0)


@pytest.mark.parametrize("crf", [0, 20, 28, 36, 51])
def test_asym_has_no_rate_compensation_or_crf_margin(crf):
    # Already zero mean and inside asymmetric bounds: the DC offset must be zero.
    raw = np.tile([-4.0, -2.0, 0.0, 6.0], (4, 1))
    offsets, prefilter = generate_ablation_offsets(
        raw, "M2_asym", crf, width=64, height=64)
    np.testing.assert_allclose(offsets, raw, atol=1e-7, rtol=0)
    assert float(offsets.mean()) == pytest.approx(0, abs=1e-7)
    # A zero extra margin in the adaptive projector would still cancel this penalty.
    expected_ratio = float(np.exp2(-raw / 6).mean())
    assert expected_ratio > 1.0
    assert float(np.exp2(-offsets.astype(np.float64) / 6).mean()) == pytest.approx(
        expected_ratio, abs=1e-7)
    assert prefilter is False


@pytest.mark.parametrize("variant", ["M2_asym_ratecomp", "M2_asym_prefilter"])
@pytest.mark.parametrize("crf", [0, 20, 28, 36, 51])
def test_rate_compensation_matches_analytic_shift_without_extra_margin(variant, crf):
    raw = np.tile([-1.25, 1.25], (4, 2))
    expected_shift = 6 * np.log2(np.cosh(np.log(2) * 1.25 / 6))
    offsets, prefilter = generate_ablation_offsets(
        raw, variant, crf, width=64, height=64)
    np.testing.assert_allclose(offsets, raw + expected_shift, atol=2e-7, rtol=0)
    assert float(offsets.mean()) > 0
    assert float(np.exp2(-offsets.astype(np.float64) / 6).mean()) == pytest.approx(1, abs=1e-7)
    assert prefilter is (variant == "M2_asym_prefilter")


def test_prefilter_changes_only_the_flag():
    raw = np.random.default_rng(9).normal(0, 8, (68, 120))
    ratecomp, flag = generate_ablation_offsets(raw, "M2_asym_ratecomp", 28)
    filtered, filter_flag = generate_ablation_offsets(raw, "M2_asym_prefilter", 28)
    np.testing.assert_array_equal(filtered, ratecomp)
    assert flag is False
    assert filter_flag is True


@pytest.mark.parametrize("variant", OFFSET_VARIANTS)
def test_symmetric_and_asymmetric_bounds(variant):
    raw = np.tile([-20.0, -2.0, 2.0, 20.0], (4, 1))
    offsets, _ = generate_ablation_offsets(raw, variant, 28, width=64, height=64)
    low = -6.0 if variant == "M2_zero_mean" else -4.0
    assert np.all(offsets >= low)
    assert np.all(offsets <= 6.0)
    assert offsets.max() == 6.0
    if variant in ("M2_zero_mean", "M2_asym"):
        assert offsets.min() == low
        assert float(offsets.mean()) == pytest.approx(0, abs=1e-7)


@pytest.mark.parametrize("variant", OFFSET_VARIANTS)
def test_partial_edges_and_mask_exclude_padding_from_projection(variant):
    weights = qg_area_weights(66, 66)
    raw = np.arange(64, dtype=np.float64).reshape(8, 8) / 2
    mask = np.ones((8, 8), dtype=bool)
    mask[1:3, 1:3] = False
    active = mask & (weights > 0)
    offsets, _ = generate_ablation_offsets(
        raw, variant, 28, width=66, height=66, mask=mask)
    np.testing.assert_array_equal(offsets[~active], 0)
    changed = raw.copy()
    changed[~active] = 1e6
    repeated, _ = generate_ablation_offsets(
        changed, variant, 28, width=66, height=66, mask=mask)
    np.testing.assert_array_equal(offsets, repeated)
    mass = weights[active].astype(np.float64)
    if variant in ("M2_zero_mean", "M2_asym"):
        assert np.average(offsets[active], weights=mass) == pytest.approx(0, abs=2e-7)
    else:
        ratio = np.average(np.exp2(-offsets[active].astype(np.float64) / 6), weights=mass)
        assert ratio == pytest.approx(1, abs=1e-7)


@pytest.mark.parametrize("variant", OFFSET_VARIANTS)
def test_empty_mask_has_zero_offsets(variant):
    offsets, _ = generate_ablation_offsets(
        np.ones((4, 4)), variant, 28, width=64, height=64,
        mask=np.zeros((4, 4), dtype=bool))
    assert offsets.dtype == np.float32
    np.testing.assert_array_equal(offsets, np.zeros((4, 4), dtype=np.float32))


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_deterministic_and_does_not_mutate_input(variant):
    raw = np.random.default_rng(12).normal(0, 6, (68, 120))
    original = raw.copy()
    raw.flags.writeable = False
    first, flag = generate_ablation_offsets(raw, variant, 28)
    second, second_flag = generate_ablation_offsets(raw, variant, 28)
    assert flag is second_flag
    if first is None:
        assert second is None
    else:
        np.testing.assert_array_equal(first, second)
        assert not np.shares_memory(first, raw)
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("variant", OFFSET_VARIANTS)
@pytest.mark.parametrize("shape", [(5, 5), (4, 16), (64,), (1, 8, 8), (0, 8)])
def test_rejects_incorrect_qg_shape_including_visible_only_grid(variant, shape):
    with pytest.raises(ValueError, match="QG16"):
        generate_ablation_offsets(np.zeros(shape), variant, 28, width=66, height=66)


@pytest.mark.parametrize("variant", OFFSET_VARIANTS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_spatial_offsets(variant, value):
    raw = np.zeros((4, 4))
    raw[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        generate_ablation_offsets(raw, variant, 28, width=64, height=64)


@pytest.mark.parametrize("crf", [-1, 52, np.nan, np.inf])
@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_rejects_invalid_crf(variant, crf):
    with pytest.raises(ValueError, match="CRF"):
        generate_ablation_offsets(np.zeros((68, 120)), variant, crf)


def test_rejects_unknown_variant_and_mismatched_mask():
    raw = np.zeros((68, 120))
    with pytest.raises(ValueError, match="variant"):
        generate_ablation_offsets(raw, "M2_asym_typo", 28)
    with pytest.raises(ValueError, match="mask"):
        generate_ablation_offsets(raw, "M2_zero_mean", 28, mask=np.ones((68, 1)))


@pytest.mark.parametrize("width,height", [(0, 64), (64, -2), (65, 64), (64, 65), (64.0, 64)])
def test_rejects_invalid_frame_dimensions(width, height):
    with pytest.raises(ValueError, match="dimensions"):
        generate_ablation_offsets(np.zeros((4, 4)), "M2_asym", 28, width=width, height=height)
