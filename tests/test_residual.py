"""Bounded luma residuals require live calibration, not a nonzero model output."""
from __future__ import annotations

import json

import numpy as np
import pytest

from prism.training.calibration import CalibrationCell, CalibrationGate


def apply(*args, **kwargs):
    from prism.preprocessing.residual import apply_bounded_residual

    return apply_bounded_residual(*args, **kwargs)


def apply_pixels(*args, **kwargs):
    from prism.preprocessing.residual import _apply_residual_pixels

    return _apply_residual_pixels(*args, **kwargs)


def payload(luma):
    return np.asarray(luma, dtype=np.uint8).tobytes() + bytes([23, 231])


def calibration_cell():
    values = np.arange(1, 101, dtype=float)
    return CalibrationCell(32, "P", values, values, values, values,
                           np.full(100, -0.2), True, True, True, True)


@pytest.fixture
def verified_gate():
    gate = CalibrationGate({(32, "P")})
    assert gate.assess(0, [calibration_cell()], lambda: None)["passed"]
    return gate


def test_default_is_byte_identity_without_calibration():
    original = payload([[0, 1], [254, 255]])
    result, stats = apply(original, np.ones((2, 2)), 2, 2)
    assert result == original
    assert stats["applied"] is False
    assert stats["bypass_reason"] == "cap_zero"
    assert stats["changed_pixel_fraction"] == 0
    assert stats["max_abs_pixel_delta"] == 0
    json.dumps(stats, allow_nan=False)


def test_nonzero_cap_without_gate_does_not_apply_residual():
    original = payload([[40, 80], [120, 160]])
    result, stats = apply(original, np.full((2, 2), 3), 2, 2, cap=3, qp=32, frame_type="P")
    assert result == original
    assert stats["bypass_reason"] == "missing_calibration"
    assert stats["applied"] is False


@pytest.mark.parametrize("cap", [0, 1, 2, 3])
def test_integer_caps_bound_actual_signed_pixel_changes(cap, verified_gate):
    original = payload([[100, 100], [100, 100]])
    residual = np.array([[-100, -0.1], [0.1, 100]], dtype=np.float32)
    result, stats = apply(original, residual, 2, 2, cap=cap,
                          gate=verified_gate, qp=32, frame_type="P")
    assert result == payload([[100 - cap, 100], [100, 100 + cap]])
    delta = np.frombuffer(result[:4], np.uint8).astype(np.int16) - 100
    assert np.abs(delta).max() <= cap
    assert stats["min_pixel_delta"] == -cap
    assert stats["max_pixel_delta"] == cap
    assert stats["max_abs_pixel_delta"] == cap
    assert stats["changed_pixel_fraction"] == (0.5 if cap else 0)
    assert stats["mean_abs_pixel_delta"] == cap / 2
    assert stats["mean_pixel_delta"] == 0


def test_residual_is_in_luma_levels_not_normalized_rgb(verified_gate):
    original = payload([[100, 100], [100, 100]])
    result, stats = apply(original, np.array([[1, -1], [2, -2]], dtype=float), 2, 2,
                          cap=3, gate=verified_gate, qp=32, frame_type="P")
    assert result == payload([[101, 99], [102, 98]])
    assert stats["changed_pixel_fraction"] == 1
    assert stats["mean_abs_pixel_delta"] == 1.5
    assert stats["residual_clipped_fraction"] == 0


def test_sub_half_level_predictions_can_disappear(verified_gate):
    original = payload([[100, 100], [100, 100]])
    result, stats = apply(original, np.array([[-0.49, 0.49], [0.01, -0.01]]), 2, 2,
                          cap=3, gate=verified_gate, qp=32, frame_type="P")
    assert result == original
    assert stats["applied"] is True
    assert stats["changed_pixel_fraction"] == 0
    assert stats["max_abs_pixel_delta"] == 0


def test_rounding_matches_nearest_even_luma_convention(verified_gate):
    result, stats = apply(payload([[100, 101], [100, 101]]),
                          np.array([[0.5, 0.5], [-0.5, -0.5]]), 2, 2,
                          cap=3, gate=verified_gate, qp=32, frame_type="P")
    assert result == payload([[100, 102], [100, 100]])
    assert stats["changed_pixel_fraction"] == 0.5


def test_cap_clipping_and_luma_saturation_are_separate(verified_gate):
    result, stats = apply(payload([[0, 1], [254, 255]]),
                          np.array([[-9, -3], [3, 9]], dtype=float), 2, 2,
                          cap=3, gate=verified_gate, qp=32, frame_type="P")
    assert result == payload([[0, 0], [255, 255]])
    assert stats["changed_pixel_fraction"] == 0.5
    assert stats["residual_clipped_fraction"] == 0.5
    assert stats["saturation_fraction"] == 1
    assert stats["max_abs_pixel_delta"] == 1
    assert stats["mean_abs_pixel_delta"] == 0.5
    assert stats["mean_pixel_delta"] == 0
    assert stats["num_luma_pixels"] == 4
    json.dumps(stats, allow_nan=False)


def test_noncontiguous_readonly_residual_and_chroma_preserved(verified_gate):
    luma = bytes([100] * 8)
    chroma = bytes([0, 255, 39, 201])
    original = luma + chroma
    residual = np.arange(-4, 4, dtype=np.float32).reshape(2, 4)[:, ::-1]
    residual.flags.writeable = False
    before = residual.copy()
    result, _ = apply(original, residual, 4, 2, cap=3,
                       gate=verified_gate, qp=32, frame_type="P")
    assert result[:8] == bytes([99, 98, 97, 97, 103, 102, 101, 100])
    assert result[8:] == chroma
    assert original == luma + chroma
    np.testing.assert_array_equal(residual, before)


@pytest.mark.parametrize("state", ["fresh", "expired", "failed", "disabled"])
def test_unverified_gate_keeps_output_zero_even_for_nonfinite_predictions(state, verified_gate):
    gate, step = verified_gate, 0
    if state == "fresh":
        gate = CalibrationGate({(32, "P")})
    elif state == "expired":
        step = 1000
    elif state == "failed":
        gate.assess(1000, [], lambda: None)
        step = 1000
    elif state == "disabled":
        gate.assess(1000, [], lambda: None)
        gate.assess(2000, [], lambda: None)
        gate.assess(3000, [calibration_cell()], lambda: None)
        step = 3000
    original = payload([[0, 16], [235, 255]])
    raw = np.array([[np.nan, np.inf], [-np.inf, 3.0]])
    result, stats = apply(original, raw, 2, 2, cap=3, gate=gate,
                          qp=32, frame_type="P", step=step)
    assert result == original
    assert stats["applied"] is False
    assert stats["bypass_reason"] == "unverified_calibration"
    assert stats["changed_pixel_fraction"] == 0
    assert stats["saturation_fraction"] == 0
    json.dumps(stats, allow_nan=False)


@pytest.mark.parametrize("qp,frame_type", [(None, "P"), (27, "P"), (32, "I"), (32, "unknown")])
def test_verified_gate_does_not_authorize_another_calibration_cell(qp, frame_type, verified_gate):
    original = payload([[100, 100], [100, 100]])
    result, stats = apply(original, np.ones((2, 2)), 2, 2, cap=3,
                          gate=verified_gate, qp=qp, frame_type=frame_type)
    assert result == original
    assert stats["applied"] is False
    assert stats["bypass_reason"] == "uncalibrated_cell"


@pytest.mark.parametrize("step,changed", [(999, True), (1000, False)])
def test_gate_expiry_is_checked_on_every_application(step, changed, verified_gate):
    original = payload([[100, 100], [100, 100]])
    result, stats = apply(original, np.ones((2, 2)), 2, 2, cap=1,
                          gate=verified_gate, step=step, qp=32, frame_type="P")
    assert (result != original) is changed
    assert stats["applied"] is changed


def test_zero_cap_suppresses_nonfinite_prediction_even_with_verified_gate(verified_gate):
    original = payload([[100, 100], [100, 100]])
    result, stats = apply(original, np.full((2, 2), np.nan), 2, 2, cap=0,
                          gate=verified_gate, qp=32, frame_type="P")
    assert result == original
    assert stats["bypass_reason"] == "cap_zero"


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_active_nonfinite_prediction_is_rejected(value, verified_gate):
    residual = np.zeros((2, 2))
    residual[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        apply(payload([[100, 100], [100, 100]]), residual, 2, 2,
              cap=3, gate=verified_gate, qp=32, frame_type="P")


@pytest.mark.parametrize("transform", [apply, apply_pixels])
@pytest.mark.parametrize("width,height", [(0, 2), (-2, 2), (3, 2), (2, 3), (2, 0),
                                         (2.0, 2), (True, 2), (2, np.bool_(True))])
def test_invalid_yuv420_geometry_rejected(width, height, transform):
    with pytest.raises(ValueError, match="dimensions"):
        transform(bytes(6), np.zeros((2, 2)), width, height, cap=0)


@pytest.mark.parametrize("transform", [apply, apply_pixels])
@pytest.mark.parametrize("raw", [b"", bytes(5), bytes(7), bytearray(6), memoryview(bytes(6))])
def test_payload_requires_exact_immutable_yuv420_bytes(raw, transform):
    with pytest.raises(ValueError, match="payload"):
        transform(raw, np.zeros((2, 2)), 2, 2, cap=0)


@pytest.mark.parametrize("transform", [apply, apply_pixels])
@pytest.mark.parametrize("residual", [np.zeros((1, 2)), np.zeros((2, 2, 1)),
                                      np.zeros(4), np.zeros((256, 256)),
                                      np.ones((2, 2), dtype=bool),
                                      np.ones((2, 2), dtype=complex),
                                      np.full((2, 2), "1"),
                                      np.full((2, 2), None, dtype=object)])
def test_residual_requires_native_shape_and_real_numeric_dtype(residual, transform):
    with pytest.raises(ValueError, match="residual"):
        transform(bytes(6), residual, 2, 2, cap=0)


@pytest.mark.parametrize("transform", [apply, apply_pixels])
@pytest.mark.parametrize("cap", [-1, 4, 8, 0.5, 3.0, True, np.nan, np.inf, "3"])
def test_invalid_amplitude_cap_rejected(cap, transform):
    with pytest.raises(ValueError, match="cap"):
        transform(bytes(6), np.zeros((2, 2)), 2, 2, cap=cap)


@pytest.mark.parametrize("step", [-1, 1.5, True, None])
def test_invalid_training_step_rejected(step):
    with pytest.raises(ValueError, match="step"):
        apply(bytes(6), np.zeros((2, 2)), 2, 2, step=step)


@pytest.mark.parametrize("qp", [-1, 52, 32.0, True, "32", np.nan])
def test_invalid_calibration_qp_rejected(qp):
    with pytest.raises(ValueError, match="qp"):
        apply(bytes(6), np.zeros((2, 2)), 2, 2, qp=qp)


@pytest.mark.parametrize("frame_type", ["", " ", None, 1])
def test_invalid_frame_type_rejected(frame_type):
    with pytest.raises(ValueError, match="frame_type"):
        apply(bytes(6), np.zeros((2, 2)), 2, 2, frame_type=frame_type)


def test_boolean_or_report_dict_cannot_replace_calibration_gate():
    for gate in (True, {"passed": True}):
        with pytest.raises(ValueError, match="CalibrationGate"):
            apply(bytes(6), np.zeros((2, 2)), 2, 2, gate=gate)


def test_numpy_integer_options_and_uint_residual_are_supported(verified_gate):
    result, stats = apply(payload([[100, 100], [100, 100]]),
                          np.full((2, 2), 2, dtype=np.uint8), np.int64(2), np.int32(2),
                          cap=np.int64(1), step=np.int64(0), qp=np.int64(32),
                          gate=verified_gate, frame_type="P")
    assert result == payload([[101, 101], [101, 101]])
    assert stats["calibration_qp"] == 32
    assert stats["frame_type"] == "P"
    json.dumps(stats, allow_nan=False)


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint8, np.uint64])
@pytest.mark.parametrize("cap", [1, 2, 3])
def test_integer_extrema_are_clipped_before_arithmetic(dtype, cap, verified_gate):
    bounds = np.iinfo(dtype)
    raw = np.array([[bounds.min, bounds.max], [bounds.min, bounds.max]], dtype=dtype)
    negative = cap if np.issubdtype(dtype, np.signedinteger) else 0
    result, stats = apply(payload([[100, 100], [100, 100]]), raw, 2, 2,
                          cap=cap, gate=verified_gate, qp=32, frame_type="P")
    assert result == payload([[100 - negative, 100 + cap], [100 - negative, 100 + cap]])
    assert stats["max_abs_pixel_delta"] == cap
    assert stats["residual_clipped_fraction"] == (1 if negative else 0.5)


@pytest.mark.parametrize("missing_gate", [True, False])
def test_missing_or_unmatched_calibration_suppresses_nonfinite_values(missing_gate, verified_gate):
    original = payload([[0, 16], [235, 255]])
    result, stats = apply(original, np.full((2, 2), np.nan), 2, 2, cap=3,
                          gate=None if missing_gate else verified_gate,
                          qp=27, frame_type="P")
    assert result == original
    assert stats["applied"] is False
    json.dumps(stats, allow_nan=False)


def test_saturation_counts_clipping_after_rounding(verified_gate):
    original = payload([[0, 255], [0, 255]])
    result, stats = apply(original, np.array([[-0.49, 0.49], [-0.6, 0.6]]), 2, 2,
                          cap=1, gate=verified_gate, qp=32, frame_type="P")
    assert result == original
    assert stats["saturation_fraction"] == 0.5
    assert stats["changed_pixel_fraction"] == 0


@pytest.mark.parametrize("cap", [0, 1, 2, 3])
def test_pure_pixels_bounds_signed_changes_without_authorization_stats(cap):
    original = payload([[100, 100], [100, 100]])
    raw = np.array([[-9, 0], [0, 9]], dtype=np.float32)
    result, stats = apply_pixels(original, raw, 2, 2, cap=cap)
    assert result == payload([[100 - cap, 100], [100, 100 + cap]])
    assert stats == {
        "cap": cap, "num_luma_pixels": 4,
        "changed_pixel_fraction": 0.5 if cap else 0.0,
        "min_pixel_delta": -cap, "max_pixel_delta": cap,
        "max_abs_pixel_delta": cap, "mean_pixel_delta": 0.0,
        "mean_abs_pixel_delta": cap / 2,
        "residual_clipped_fraction": 0.5, "saturation_fraction": 0.0,
    }
    json.dumps(stats, allow_nan=False)


@pytest.mark.parametrize("cap", [0, 1, 2, 3])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_pure_pixels_rejects_nonfinite_even_with_zero_cap(cap, value):
    raw = np.array([[0, 0], [0, value]], dtype=float)
    with pytest.raises(ValueError, match="finite"):
        apply_pixels(bytes(6), raw, 2, 2, cap=cap)


@pytest.mark.parametrize("luma,raw,want,clipped,saturated,changed", [
    ([[100, 101], [100, 101]], [[0.5, 0.5], [-0.5, -0.5]],
     [[100, 102], [100, 100]], 0.0, 0.0, 0.5),
    ([[0, 1], [254, 255]], [[-9, -3], [3, 9]],
     [[0, 0], [255, 255]], 0.5, 1.0, 0.5),
    ([[0, 255], [0, 255]], [[-0.49, 0.49], [-0.6, 0.6]],
     [[0, 255], [0, 255]], 0.0, 0.5, 0.0),
])
def test_pure_pixels_rounds_luma_then_saturates(luma, raw, want, clipped, saturated, changed):
    result, stats = apply_pixels(payload(luma), np.array(raw), 2, 2, cap=3)
    assert result == payload(want)
    assert stats["residual_clipped_fraction"] == clipped
    assert stats["saturation_fraction"] == saturated
    assert stats["changed_pixel_fraction"] == changed


def test_pure_pixels_preserves_readonly_noncontiguous_residual_and_chroma():
    original = bytes([100] * 8 + [0, 255, 39, 201])
    raw = np.arange(-4, 4, dtype=np.float32).reshape(2, 4)[:, ::-1]
    raw.flags.writeable = False
    before = raw.copy()
    result, stats = apply_pixels(original, raw, np.int64(4), np.int32(2), cap=np.int64(3))
    assert result == bytes([99, 98, 97, 97, 103, 102, 101, 100, 0, 255, 39, 201])
    assert original == bytes([100] * 8 + [0, 255, 39, 201])
    np.testing.assert_array_equal(raw, before)
    assert stats["num_luma_pixels"] == 8
    assert stats["residual_clipped_fraction"] == 0.125
    json.dumps(stats, allow_nan=False)


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint8, np.uint64])
@pytest.mark.parametrize("cap", [1, 2, 3])
def test_pure_pixels_preserves_integer_extrema_clipping(dtype, cap):
    bounds = np.iinfo(dtype)
    raw = np.array([[bounds.min, bounds.max], [bounds.min, bounds.max]], dtype=dtype)
    negative = cap if np.issubdtype(dtype, np.signedinteger) else 0
    result, stats = apply_pixels(payload([[100, 100], [100, 100]]), raw, 2, 2, cap=cap)
    assert result == payload([[100 - negative, 100 + cap], [100 - negative, 100 + cap]])
    assert stats["max_abs_pixel_delta"] == cap
    assert stats["residual_clipped_fraction"] == (1.0 if negative else 0.5)


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_gated_transform_retains_all_pixel_stats(cap, verified_gate):
    original = payload([[0, 100], [101, 255]])
    raw = np.array([[-9, -0.5], [0.5, 9]])
    pure_result, pure_stats = apply_pixels(original, raw, 2, 2, cap=cap)
    result, stats = apply(original, raw, 2, 2, cap=cap, gate=verified_gate,
                          qp=32, frame_type="P", step=7)
    assert result == pure_result
    assert stats == {
        **pure_stats, "step": 7, "calibration_qp": 32, "frame_type": "P",
        "applied": True, "bypass_reason": None,
    }


@pytest.mark.parametrize("cap,gate_kind,qp,reason", [
    (0, "verified", 32, "cap_zero"),
    (3, "missing", 32, "missing_calibration"),
    (3, "fresh", 32, "unverified_calibration"),
    (3, "verified", 27, "uncalibrated_cell"),
])
def test_gated_nonfinite_bypass_preserves_exact_stats(cap, gate_kind, qp, reason, verified_gate):
    gate = {"verified": verified_gate, "missing": None,
            "fresh": CalibrationGate({(32, "P")})}[gate_kind]
    original = payload([[0, 100], [101, 255]])
    result, stats = apply(original, np.full((2, 2), np.nan), 2, 2,
                          cap=cap, gate=gate, qp=qp, frame_type="P")
    assert result == original
    assert stats == {
        "cap": cap, "step": 0, "calibration_qp": qp, "frame_type": "P",
        "applied": False, "bypass_reason": reason, "num_luma_pixels": 4,
        "changed_pixel_fraction": 0.0, "min_pixel_delta": 0, "max_pixel_delta": 0,
        "max_abs_pixel_delta": 0, "mean_pixel_delta": 0.0, "mean_abs_pixel_delta": 0.0,
        "residual_clipped_fraction": 0.0, "saturation_fraction": 0.0,
    }
    json.dumps(stats, allow_nan=False)
