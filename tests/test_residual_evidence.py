"""Synthetic raw-evidence tests; no native codec or measured-truth claims."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from prism.codec.x265_bridge import BuildPin, EncoderConfig
from prism.training.calibration import CalibrationGate


ARRAYS = ("actual_bits", "predicted_bits", "actual_delta_j", "predicted_delta_j",
          "direction_delta_j")
BOOLS = ("semantic_passed", "same_codec_state", "feasible_plus_minus_one",
         "dependent_frames_measured")
EVALUATION = {"source_group": "MOT17-02-FRCNN", "sha256": "e" * 64}


def runtime_context(step=0):
    return {
        "checkpoint_sha256": "a" * 64, "seed": 0, "checkpoint_step": step,
        # Deliberately different from CRF: CQP configuration is not an effective QP.
        "encoder_config": asdict(EncoderConfig(qp=32, rc_mode="CRF", crf=27.0,
                                               bframes=0, fps=25)),
        "build_pin": asdict(BuildPin("b" * 64, "c" * 64, "test-build", 210, 123)),
        "width": 64, "height": 48, "fps": 25, "cap": 1,
        "color_contract": "bt709_limited_yuv420p8_luma_clip_0_255",
        "rate_control_condition": {
            "mode": "CRF", "value": 27, "cell_qp": 27, "frame_type": "unknown",
            "interpretation": "crf_setting_not_effective_qp",
        },
    }


def raw_cell(passed=True):
    values = list(range(1, 101))
    return {
        "qp": 27, "frame_type": "unknown", "actual_bits": values,
        "predicted_bits": values.copy() if passed else [v * 1.2 for v in values],
        "actual_delta_j": values.copy(), "predicted_delta_j": values.copy(),
        "direction_delta_j": [-0.2] * 100,
        **{key: True for key in BOOLS}, "split": "calibration",
    }


def document(context=None, outcomes=(True,)):
    return {
        "schema_version": 1, "protocol": "bounded_residual_crf_unknown_v1",
        "context": deepcopy(context if context is not None else runtime_context()),
        "calibration_sources": [
            {"source_group": "MOT17-04-DPM", "sha256": "d" * 64, "split": "calibration"}
        ],
        "history": [{"step": i * 1000, "cells": [raw_cell(passed)]}
                    for i, passed in enumerate(outcomes)],
    }


def write_evidence(tmp_path, data):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def load(path, context=None, evaluation_source=None):
    from prism.evaluation.residual_evidence import load_residual_evidence
    return load_residual_evidence(
        path, context=runtime_context() if context is None else context,
        evaluation_source=EVALUATION if evaluation_source is None else evaluation_source,
    )


def reject(tmp_path, data, context=None, evaluation_source=None):
    with pytest.raises(ValueError) as exc:
        load(write_evidence(tmp_path, data), context, evaluation_source)
    assert str(exc.value).strip(), "Invalid evidence must have a readable explanation"


def change(mapping, dotted_key, value):
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        mapping = mapping[key]
    mapping[keys[-1]] = value


def test_valid_raw_evidence_returns_live_gate_and_raw_file_digest(tmp_path, capsys):
    context = runtime_context(999)
    path = write_evidence(tmp_path, document(context))
    before = path.read_bytes()
    result = load(path, context)
    assert isinstance(result.gate, CalibrationGate)
    assert result.step == 999
    assert result.cell_qp == 27 and result.frame_type == "unknown"
    assert result.gate.expected_cells == {(27, "unknown")}
    assert result.gate.allow_delta_x(999) and not result.gate.allow_delta_x(1000)
    assert not result.gate.fine_tune_disabled
    assert result.context == context and result.digest == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]
    assert capsys.readouterr() == ("", "")


def test_missing_file_is_not_authorization(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        load(tmp_path / "missing.json")


@pytest.mark.parametrize("key,value", [
    ("verified", True), ("passed", True), ("gate", {"verified": True}),
    ("schema_version", True), ("schema_version", 1.0), ("schema_version", 2),
    ("protocol", "qp_frame_type_v1"), ("history", []), ("history", {}),
    ("calibration_sources", []), ("calibration_sources", {}),
])
def test_rejects_summary_flags_and_invalid_top_level(tmp_path, key, value):
    data = document()
    data[key] = value
    reject(tmp_path, data)


@pytest.mark.parametrize("key", list(document()))
def test_rejects_missing_top_level_fields(tmp_path, key):
    data = document()
    del data[key]
    reject(tmp_path, data)


@pytest.mark.parametrize("key,value", [
    ("checkpoint_sha256", "f" * 64), ("seed", 1), ("checkpoint_step", 1),
    ("cap", 2), ("width", 128), ("height", 96), ("fps", 30),
    ("encoder_config.qp", 37), ("encoder_config.preset", "slow"),
    ("encoder_config.rc_mode", "CQP"), ("encoder_config.crf", 28.0),
    ("build_pin.library_sha256", "f" * 64), ("build_pin.shim_sha256", "f" * 64),
    ("build_pin.version", "other"), ("build_pin.enable_assembly", False),
    ("build_pin.abi", 211), ("build_pin.picture_size", 124),
    ("color_contract", "full_range"), ("rate_control_condition.frame_type", "P"),
])
def test_binds_entire_expected_runtime_context(tmp_path, key, value):
    expected = runtime_context()
    change(expected, key, value)
    reject(tmp_path, document(), expected)


@pytest.mark.parametrize("key,value", [
    ("checkpoint_sha256", "A" * 64), ("checkpoint_sha256", "a" * 63),
    ("seed", True), ("seed", -1), ("seed", 0.0),
    ("checkpoint_step", True), ("checkpoint_step", -1), ("checkpoint_step", 0.0),
    ("width", True), ("width", 63), ("width", 0), ("height", -2),
    ("height", 48.0), ("fps", True), ("fps", 0), ("fps", 25.0),
    ("cap", True), ("cap", 0), ("cap", 4), ("cap", 1.0),
    ("color_contract", "bt709_full"), ("encoder_config.rc_mode", "CQP"),
    ("encoder_config.crf", 32.0), ("encoder_config.crf", True),
    ("encoder_config.fps", 30), ("encoder_config.bframes", 1),
    ("encoder_config.bframes", False), ("encoder_config.cutree", 0),
    ("encoder_config.qp", True), ("encoder_config.bitrate_kbps", "100"),
    ("build_pin.enable_assembly", 1), ("build_pin.abi", True),
    ("build_pin.library_sha256", "bad"), ("build_pin.version", 210),
    ("rate_control_condition.mode", "CQP"), ("rate_control_condition.value", 27.0),
    ("rate_control_condition.value", True), ("rate_control_condition.value", -1),
    ("rate_control_condition.value", 52), ("rate_control_condition.cell_qp", 32),
    ("rate_control_condition.cell_qp", 27.0),
    ("rate_control_condition.frame_type", "I"),
    ("rate_control_condition.interpretation", "effective_qp"),
])
def test_invalid_context_is_rejected_even_when_expected_matches(tmp_path, key, value):
    context = runtime_context()
    change(context, key, value)
    reject(tmp_path, document(context), context)


@pytest.mark.parametrize("section", [None, "encoder_config", "build_pin", "rate_control_condition"])
@pytest.mark.parametrize("extra", [False, True])
def test_context_requires_exact_complete_fields(tmp_path, section, extra):
    context = runtime_context()
    target = context if section is None else context[section]
    if extra:
        target["verified"] = True
    else:
        del target[next(iter(target))]
    reject(tmp_path, document(context), context)


def test_canonical_context_comparison_does_not_equate_int_and_float(tmp_path):
    context = runtime_context()
    data = document(context)
    data["context"]["encoder_config"]["crf"] = 27
    reject(tmp_path, data, context)


@pytest.mark.parametrize("key", list(raw_cell()))
def test_every_cell_field_is_required(tmp_path, key):
    data = document()
    del data["history"][0]["cells"][0][key]
    reject(tmp_path, data)


@pytest.mark.parametrize("key,value", [
    ("verified", True), ("passed", True), ("qp", 32), ("qp", True), ("qp", 27.0),
    ("frame_type", "P"), ("frame_type", "I"), ("split", "test"), ("split", True),
] + [(key, value) for key in BOOLS for value in (1, "true", None)])
def test_rejects_forged_summary_wrong_cell_and_coerced_flags(tmp_path, key, value):
    data = document()
    data["history"][0]["cells"][0][key] = value
    reject(tmp_path, data)


@pytest.mark.parametrize("key", BOOLS)
def test_false_safety_flag_blocks_current_calibration(tmp_path, key):
    data = document()
    data["history"][0]["cells"][0][key] = False
    reject(tmp_path, data)


@pytest.mark.parametrize("key", ARRAYS)
@pytest.mark.parametrize("value", ["1", True, None, [1], float("nan"), float("inf"), -float("inf")])
def test_rejects_non_numeric_nonfinite_or_nested_array_elements(tmp_path, key, value):
    data = document()
    data["history"][0]["cells"][0][key][0] = value
    reject(tmp_path, data)


@pytest.mark.parametrize("value", [1, "1,2,3", {}, None])
def test_arrays_must_be_real_lists(tmp_path, value):
    data = document()
    data["history"][0]["cells"][0]["actual_bits"] = value
    reject(tmp_path, data)


def test_arrays_must_be_paired(tmp_path):
    data = document()
    data["history"][0]["cells"][0]["actual_bits"].pop()
    reject(tmp_path, data)


@pytest.mark.parametrize("failure", ["samples", "bits", "rate", "rank", "direction"])
def test_raw_gate_metrics_cannot_be_bypassed(tmp_path, failure):
    data = document()
    cell = data["history"][0]["cells"][0]
    if failure == "samples":
        for key in ARRAYS:
            cell[key] = cell[key][:99]
    elif failure == "bits":
        cell["actual_bits"][0] = 0
    elif failure == "rate":
        cell["predicted_bits"] = [x * 1.2 for x in cell["actual_bits"]]
    elif failure == "rank":
        cell["predicted_delta_j"].reverse()
    else:
        cell["direction_delta_j"] = [0.2] * 100
    reject(tmp_path, data)


@pytest.mark.parametrize("steps", [[1000], [0, 2000], [0, 0], [0, 500], [0, -1000], [True], [0.0], ["0"]])
def test_history_requires_initial_zero_and_consecutive_integer_boundaries(tmp_path, steps):
    context = runtime_context(3000)
    data = document(context)
    data["history"] = [{"step": step, "cells": [raw_cell()]} for step in steps]
    reject(tmp_path, data, context)


@pytest.mark.parametrize("entry", [
    {"step": 0}, {"cells": [raw_cell()]}, {"step": 0, "cells": [raw_cell()], "verified": True},
    {"step": 0, "cells": []}, {"step": 0, "cells": [raw_cell(), raw_cell()]},
    {"step": 0, "cells": {}}, {"step": 0, "cells": [True]}, None,
])
def test_history_requires_exact_entry_and_one_expected_cell(tmp_path, entry):
    data = document()
    data["history"] = [entry]
    reject(tmp_path, data)


@pytest.mark.parametrize("step,outcomes", [(1000, (True,)), (999, (True, True)),
                                            (0, (False,)), (2000, (False, False, True))])
def test_expired_future_failed_and_permanently_disabled_histories_block(tmp_path, step, outcomes):
    context = runtime_context(step)
    reject(tmp_path, document(context, outcomes), context)


@pytest.mark.parametrize("outcomes", [(False, True), (True, False, True)])
def test_single_historical_failure_can_recover(tmp_path, outcomes):
    context = runtime_context((len(outcomes) - 1) * 1000)
    result = load(write_evidence(tmp_path, document(context, outcomes)), context)
    assert result.gate.allow_delta_x(context["checkpoint_step"])
    assert result.gate.consecutive_failures == 0 and not result.gate.fine_tune_disabled


@pytest.mark.parametrize("group", ["MOT17-02", "MOT17-02-DPM", "MOT17-02-SDP", "MOT17-02-FRCNN"])
def test_evaluation_source_alias_overlap_is_blocked(tmp_path, group):
    data = document()
    data["calibration_sources"][0]["source_group"] = group
    reject(tmp_path, data)


def test_source_hash_overlap_is_blocked_even_with_different_group(tmp_path):
    data = document()
    data["calibration_sources"][0]["sha256"] = EVALUATION["sha256"]
    reject(tmp_path, data)


@pytest.mark.parametrize("group", ["MOT17-020", "MOT17-02-extra", "MOT17-02-DPM-extra"])
def test_arbitrary_source_prefixes_are_not_guessed(tmp_path, group):
    data = document()
    data["calibration_sources"][0]["source_group"] = group
    assert load(write_evidence(tmp_path, data)).gate.allow_delta_x(0)


@pytest.mark.parametrize("key,value", [
    ("source_group", "MOT17-04-SDP"), ("source_group", "MOT17-04-DPM"),
    ("sha256", "d" * 64),
])
def test_declared_source_groups_and_hashes_are_individually_unique(tmp_path, key, value):
    data = document()
    source = {"source_group": "MOT17-05", "sha256": "f" * 64, "split": "calibration"}
    source[key] = value
    data["calibration_sources"].append(source)
    reject(tmp_path, data)


@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.parametrize("key,value", [("source_group", ""), ("source_group", "   "),
    ("source_group", 2), ("sha256", "D" * 64), ("sha256", "invalid"), ("verified", True)])
def test_source_declarations_require_strict_identity_fields(tmp_path, evaluation, key, value):
    data = document()
    source = deepcopy(EVALUATION) if evaluation else data["calibration_sources"][0]
    source[key] = value
    reject(tmp_path, data, evaluation_source=source if evaluation else None)


def test_calibration_source_cannot_declare_an_evaluation_split(tmp_path):
    data = document()
    data["calibration_sources"][0]["split"] = "test"
    reject(tmp_path, data)


@pytest.mark.parametrize("text", [
    '{"schema_version": 1, "schema_version": 1}',
    json.dumps(document()).replace('"qp": 27', '"qp": 27, "qp": 27'),
    json.dumps(document()).replace('"seed": 0', '"seed": 0, "seed": 0'),
    json.dumps(document()).replace('"seed": 0', '"seed": NaN'),
    json.dumps(document()).replace('"seed": 0', '"seed": Infinity'),
    json.dumps(document()).replace('"seed": 0', '"seed": -Infinity'),
    '[1, 2]', 'null', '{broken',
])
def test_duplicate_keys_nonfinite_constants_and_invalid_json_block(tmp_path, text):
    path = tmp_path / "evidence.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load(path)
