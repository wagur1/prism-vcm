"""Replay raw calibration under an explicit CRF-setting/unknown-frame protocol.

Hashes bind declared identities, not measurement truth. This loader neither verifies
that samples were measured by a real codec nor equates CRF with effective QP; older
QP/frame-type evidence is not interchangeable with this protocol. Replay changes
only a fresh gate's state, never checkpoint weights.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from prism.codec.x265_bridge import BuildPin, EncoderConfig
from prism.training.calibration import CalibrationCell, CalibrationGate


@dataclass
class ResidualEvidence:
    gate: CalibrationGate
    step: int
    cell_qp: int
    frame_type: str
    digest: str
    context: dict


_CONTEXT_KEYS = {"checkpoint_sha256", "seed", "checkpoint_step", "encoder_config",
                 "build_pin", "width", "height", "fps", "cap", "color_contract",
                 "rate_control_condition"}
_ARRAYS = ("actual_bits", "predicted_bits", "actual_delta_j", "predicted_delta_j",
           "direction_delta_j")
_FLAGS = ("semantic_passed", "same_codec_state", "feasible_plus_minus_one",
          "dependent_frames_measured")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, names, label):
    _require(type(value) is dict and set(value) == set(names),
             f"{label} must contain exactly the required fields: {sorted(names)}")


def _integer(value, label, minimum=0, maximum=None):
    _require(type(value) is int and value >= minimum and
             (maximum is None or value <= maximum), f"{label} must be a valid integer")


def _hash(value, label):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{label} must be a lowercase SHA256 digest")


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _canonical(value):
    # JSON serialization preserves bool/int and int/float distinctions; == does not.
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("context must contain finite JSON-compatible primitives") from exc


def _validate_context(context):
    _keys(context, _CONTEXT_KEYS, "context")
    _hash(context["checkpoint_sha256"], "checkpoint_sha256")
    for name in ("seed", "checkpoint_step", "width", "height", "fps", "cap"):
        _integer(context[name], name, 0 if name in {"seed", "checkpoint_step"} else 1,
                 3 if name == "cap" else None)
    _require(context["width"] % 2 == context["height"] % 2 == 0,
             "context geometry must be positive even width and height")
    _require(context["color_contract"] == "bt709_limited_yuv420p8_luma_clip_0_255",
             "Unsupported color_contract")
    condition = context["rate_control_condition"]
    _keys(condition, {"mode", "value", "cell_qp", "frame_type", "interpretation"},
          "rate_control_condition")
    for name in ("value", "cell_qp"):
        _integer(condition[name], f"rate_control_condition.{name}", 0, 51)
    _require(condition == {"mode": "CRF", "value": condition["value"],
                           "cell_qp": condition["value"], "frame_type": "unknown",
                           "interpretation": "crf_setting_not_effective_qp"},
             "Evidence requires explicit CRF-setting/unknown-frame condition, not effective QP")
    config = context["encoder_config"]
    _keys(config, {field.name for field in fields(EncoderConfig)}, "encoder_config")
    for name in ("qp", "aq_mode", "qg_size", "gop", "bframes", "fps"):
        _integer(config[name], f"encoder_config.{name}")
    for name in ("cutree", "sao"):
        _require(type(config[name]) is bool, f"encoder_config.{name} must be boolean")
    for name in ("preset", "rc_mode"):
        _require(type(config[name]) is str, f"encoder_config.{name} must be a string")
    for name in ("bitrate_kbps", "vbv_maxrate_kbps", "vbv_bufsize_kbits"):
        if config[name] is not None:
            _integer(config[name], f"encoder_config.{name}", 1)
    _require(config["rc_mode"] == "CRF" and _finite_number(config["crf"]) and
             config["crf"] == condition["value"] and config["fps"] == context["fps"] and
             config["bframes"] == 0, "encoder_config must match CRF, fps and bframes=0")
    pin = context["build_pin"]
    _keys(pin, {field.name for field in fields(BuildPin)}, "build_pin")
    for name in ("library_sha256", "shim_sha256"):
        _hash(pin[name], f"build_pin.{name}")
    for name in ("abi", "picture_size"):
        _integer(pin[name], f"build_pin.{name}")
    _require(type(pin["version"]) is str and type(pin["enable_assembly"]) is bool,
             "build_pin requires a string version and boolean enable_assembly")
    return _canonical(context)


def _source(value, calibration=False):
    _keys(value, {"source_group", "sha256", "split"} if calibration else
          {"source_group", "sha256"}, "source")
    group = value["source_group"]
    _require(type(group) is str and bool(group.strip()), "source_group must be a nonempty string")
    _hash(value["sha256"], "source.sha256")
    if calibration:
        _require(value["split"] == "calibration", "Source split must be calibration")
    match = re.fullmatch(r"(MOT17-\d{2})(?:-(?:DPM|SDP|FRCNN))?", group)
    return (match.group(1) if match else group), value["sha256"]


def _cell(raw, cell_qp):
    _keys(raw, {field.name for field in fields(CalibrationCell)}, "CalibrationCell")
    _integer(raw["qp"], "cell.qp", 0, 51)
    _require((raw["qp"], raw["frame_type"]) == (cell_qp, "unknown"),
             "Cell must match the explicit CRF-setting/unknown-frame condition")
    _require(raw["split"] == "calibration", "Cell split must be calibration")
    for name in _FLAGS:
        _require(type(raw[name]) is bool, f"cell.{name} must be boolean")
    for name in _ARRAYS:
        _require(type(raw[name]) is list and all(_finite_number(v) for v in raw[name]),
                 f"cell.{name} must be a list of finite real numbers (not booleans)")
    _require(len({len(raw[name]) for name in _ARRAYS}) == 1,
             "Calibration arrays must contain equal-length paired samples")
    return CalibrationCell(**{**raw, **{name: np.asarray(raw[name], dtype=float) for name in _ARRAYS}})


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant is not permitted: {value}")


def load_residual_evidence(path: Path, *, context: dict, evaluation_source: dict) -> ResidualEvidence:
    """Validate identity bindings and replay raw metrics; no output, writes or training.

    The returned step is the checkpoint step, not necessarily the last assessment.
    A single historical failure may recover; two consecutive failures permanently
    block authorization. Caller-supplied identities cannot establish measurement truth.
    """
    expected = _validate_context(context)
    eval_group, eval_hash = _source(evaluation_source)
    raw_bytes = Path(path).read_bytes()
    data = json.loads(raw_bytes, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    _keys(data, {"schema_version", "protocol", "context", "calibration_sources", "history"},
          "residual evidence")
    _require(type(data["schema_version"]) is int and data["schema_version"] == 1 and
             data["protocol"] == "bounded_residual_crf_unknown_v1", "Unsupported evidence schema/protocol")
    _require(_validate_context(data["context"]) == expected, "Evidence context does not match runtime context")
    sources = data["calibration_sources"]
    _require(type(sources) is list and bool(sources), "calibration_sources must be a nonempty list")
    groups, hashes = {eval_group}, {eval_hash}
    for source in sources:
        group, digest = _source(source, calibration=True)
        _require(group not in groups and digest not in hashes,
                 "Calibration source duplicate or evaluation source overlap")
        groups.add(group)
        hashes.add(digest)
    history = data["history"]
    _require(type(history) is list and bool(history), "history must be a nonempty list")
    step, cell_qp = context["checkpoint_step"], context["rate_control_condition"]["cell_qp"]
    gate = CalibrationGate({(cell_qp, "unknown")})
    for index, assessment in enumerate(history):
        _keys(assessment, {"step", "cells"}, "history entry")
        _integer(assessment["step"], "history step")
        _require(assessment["step"] == index * gate.interval and assessment["step"] <= step,
                 "History must start at 0, advance by 1000 and not exceed checkpoint_step")
        cells = assessment["cells"]
        _require(type(cells) is list and len(cells) == 1, "History requires exactly the expected cell")
        gate.assess(assessment["step"], [_cell(cells[0], cell_qp)], lambda: None)
    _require(not gate.fine_tune_disabled and gate.allow_delta_x(step),
             "Residual evidence is failed, expired or permanently disabled at checkpoint_step")
    return ResidualEvidence(gate, step, cell_qp, "unknown", hashlib.sha256(raw_bytes).hexdigest(),
                            json.loads(expected))
