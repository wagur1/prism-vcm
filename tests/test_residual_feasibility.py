"""Research-only paired codec measurements must never authorize residual deployment."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from prism.codec.x265_bridge import BuildPin
from prism.video.mot_eval import MOTSequence
from tier1b.run_bounded_residual_mot17 import source_identity


def collector():
    return importlib.import_module("scripts.collect_residual_feasibility")


def test_separate_research_collector_exists():
    assert collector().build_parser().prog


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "MOT17-02-FRCNN"
    (root / "img1").mkdir(parents=True)
    (root / "gt").mkdir()
    (root / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-02-FRCNN\nimDir=img1\nimExt=.png\n"
        "imWidth=4\nimHeight=2\nframeRate=25\nseqLength=6\n", encoding="utf-8")
    for i in range(1, 7):
        Image.fromarray(np.full((2, 4, 3), 100, np.uint8)).save(root / "img1" / f"{i:06d}.png")
    (root / "gt/gt.txt").write_text("".join(f"{i},1,0,0,2,2,1,1,1\n" for i in range(1, 7)))
    checkpoint, library, shim = tmp_path / "student_seed_0.pt", tmp_path / "codec.dll", tmp_path / "shim.dll"
    weights = tmp_path / "yolo.pt"
    for p in (checkpoint, library, shim, weights):
        p.write_bytes(p.name.encode())
    pin = BuildPin(hashlib.sha256(library.read_bytes()).hexdigest(),
                   hashlib.sha256(shim.read_bytes()).hexdigest(), "test-codec", 1, 200)
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(asdict(pin)))
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps({"schema_version": 1, "source": source_identity(MOTSequence(root)),
                                    "role": "development"}))
    return SimpleNamespace(root=root, checkpoint=checkpoint, library=library, shim=shim, weights=weights,
                           pin=pin, pin_path=pin_path, manifest=manifest, output=tmp_path / "result")


def args_for(inputs):
    return collector().build_parser().parse_args([
        "--sequence-dir", str(inputs.root), "--checkpoint", str(inputs.checkpoint),
        "--library", str(inputs.library), "--shim", str(inputs.shim),
        "--build-pin", str(inputs.pin_path), "--source-manifest", str(inputs.manifest),
        "--output", str(inputs.output), "--crf", "32", "--cap", "2",
        "--intervention-start", "3", "--intervention-end", "4",
        "--suffix-frames", "2", "--max-frames", "6", "--max-encodes", "2",
        "--device", "cpu", "--research-only"])


@pytest.fixture
def runtime(monkeypatch):
    module = collector()
    state = SimpleNamespace(student_loads=0, predictions=0, native_loads=0, detector_loads=0,
                            codecs=[], encodes=[], detector_values=[], residual_value=2.0,
                            decode_delta=0, encode_delta=0, byte_delta=0, closed=0,
                            fail_second=False, decoded_value=90)

    class FakeStudent:
        def __init__(self, checkpoint, device):
            assert checkpoint.is_file() and device == "cpu"
            state.student_loads += 1
            self.model = SimpleNamespace(backbone_name="mobilenetv4_conv_small")

        def predict_residual(self, rgb, crf):
            assert rgb.shape == (2, 4, 3) and crf == 32
            state.predictions += 1
            return np.full((2, 4), state.residual_value, dtype=np.float32)

    class FakeLibrary:
        def __init__(self, *, library, shim, pin):
            state.native_loads += 1
            self.pin = pin

    class FakeCodec:
        def __init__(self, config, *, library):
            self.config = config
            state.codecs.append(self)

        def encode_sequence(self, frames, offset_generator, output, width, height, fps):
            assert offset_generator is None
            assert (width, height, fps) == (4, 2, 25)
            rows = list(frames)
            state.encodes.append(rows)
            if state.fail_second and len(state.encodes) == 2:
                raise RuntimeError("synthetic second encode failure")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"stream" * len(rows))
            self.count = len(rows)
            return {"num_frames": len(rows) + state.encode_delta,
                    "total_bytes": output.stat().st_size + state.byte_delta}

        def decode_bitstream(self, path, width, height):
            try:
                for _ in range(self.count + state.decode_delta):
                    yield np.full((height, width, 3), state.decoded_value, np.uint8)
            finally:
                state.closed += 1

    class FakeDetector:
        def __init__(self, weights, device):
            state.detector_loads += 1

        def detect(self, rgb):
            np.testing.assert_array_equal(rgb, np.full((2, 4, 3), state.decoded_value, np.uint8))
            state.detector_values.append(int(rgb[0, 0, 0]))
            return np.array([[0, 0, 2, 2, 0.99]])

    monkeypatch.setattr(module, "ResidualStudent", FakeStudent)
    monkeypatch.setattr(module, "X265Library", FakeLibrary)
    monkeypatch.setattr(module, "VideoCodecRunner", FakeCodec)
    monkeypatch.setattr(module, "YOLODetector", FakeDetector)
    return state


def test_missing_research_ack_blocks_all_runtime(inputs, runtime):
    module = collector()
    args = args_for(inputs)
    args.research_only = False
    with pytest.raises(ValueError, match="research-only"):
        module.run(args)
    assert runtime.native_loads == runtime.student_loads == 0
    assert not inputs.output.exists()


def test_validate_only_needs_no_research_execution_ack_and_writes_nothing(inputs, runtime):
    module = collector()
    args = args_for(inputs)
    args.validate_only, args.research_only = True, False
    report = module.run(args)
    assert report["status"] == "validated"
    assert report["research_only"] is True
    assert report["calibration_status"] == "not_established"
    assert report["gate_authorization"] is False
    assert report["planned_encodes"] == 2
    assert not inputs.output.exists()
    assert runtime.native_loads == runtime.student_loads == runtime.detector_loads == 0


def test_two_fresh_encoders_same_prefix_suffix_and_only_intervention_luma_changes(inputs, runtime, monkeypatch):
    from prism.training.calibration import CalibrationGate
    module = collector()

    def forbid_gate(*args, **kwargs):
        raise AssertionError("Research collector must not fabricate a gate")

    monkeypatch.setattr(CalibrationGate, "__init__", forbid_gate)
    path = module.run(args_for(inputs))
    report = json.loads(path.read_text())
    assert report["status"] == "complete"
    assert report["research_only"] and not report["gate_authorization"]
    assert report["calibration_status"] == "not_established"
    assert len(runtime.codecs) == len(runtime.encodes) == 2
    assert runtime.codecs[0] is not runtime.codecs[1]
    assert runtime.codecs[0].config == runtime.codecs[1].config
    assert runtime.predictions == 2
    original, changed = runtime.encodes
    for index, (before, after) in enumerate(zip(original, changed), 1):
        assert before[8:] == after[8:]
        expected = np.frombuffer(before[:8], np.uint8).astype(int) + (2 if index in (3, 4) else 0)
        np.testing.assert_array_equal(np.frombuffer(after[:8], np.uint8).astype(int), expected)
    assert original[:2] == changed[:2] and original[4:] == changed[4:]
    first, second = report["observations"]
    assert first["distortion_rgb"]["all"]["mse"] == 100
    for stage in ("prefix", "intervention", "suffix"):
        assert second["distortion_rgb"][stage]["frames"] == 2
        assert second["distortion_rgb"][stage]["mse"] == 100
    assert second["pixel_changes"]["changed_pixel_fraction"] == pytest.approx(1 / 3)
    assert second["pixel_changes"]["mean_abs_pixel_delta"] == pytest.approx(2 / 3)
    assert second["pixel_changes"]["max_abs_pixel_delta"] == 2
    assert second["intervention_pixel_changes"]["changed_pixel_fraction"] == 1
    assert first["metrics"]["evaluated"] is False
    assert first["metrics"]["reason"] == "detector_not_requested"
    assert not first["external_qp_offsets"] and not second["external_qp_offsets"]
    assert "history" not in report and "cells" not in report
    attempts = [json.loads(line) for line in (inputs.output / "attempts.jsonl").read_text().splitlines()]
    assert [row["status"] for row in attempts] == ["complete", "complete"]
    assert all(row["gate_authorization"] is False for row in attempts)
    assert runtime.closed == 2


def test_report_schema_is_rejected_by_calibration_parser(inputs, runtime):
    from prism.evaluation.residual_evidence import load_residual_evidence
    module = collector()
    path = module.run(args_for(inputs))
    report = json.loads(path.read_text())
    context = {"checkpoint_sha256": report["checkpoint_sha256"], "seed": 0, "checkpoint_step": 0,
               "encoder_config": report["encoder_config"], "build_pin": report["build_pin"],
               "width": 4, "height": 2, "fps": 25, "cap": 2,
               "color_contract": "bt709_limited_yuv420p8_luma_clip_0_255",
               "rate_control_condition": {"mode": "CRF", "value": 32, "cell_qp": 32,
                  "frame_type": "unknown", "interpretation": "crf_setting_not_effective_qp"}}
    with pytest.raises(ValueError, match="residual evidence"):
        load_residual_evidence(path, context=context, evaluation_source=report["source"])


@pytest.mark.parametrize("field,value", [("cap", 0), ("cap", 4), ("cap", True), ("crf", -1), ("crf", 52),
    ("crf", 32.0), ("intervention_start", 1), ("intervention_start", 5), ("intervention_end", 7),
    ("suffix_frames", 0), ("max_frames", 5), ("max_encodes", 1), ("max_encodes", 3)])
def test_invalid_window_or_budget_blocks(inputs, runtime, field, value):
    args = args_for(inputs)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        collector().run(args)
    assert not inputs.output.exists() and runtime.native_loads == 0


@pytest.mark.parametrize("change", ["test_role", "hash", "group", "extra_field", "missing"])
def test_source_declaration_failures_block(inputs, runtime, change):
    document = json.loads(inputs.manifest.read_text())
    if change == "test_role":
        document["role"] = "test"
    elif change == "hash":
        document["source"]["sha256"] = "a" * 64
    elif change == "group":
        document["source"]["source_group"] = "MOT17-04"
    elif change == "extra_field":
        document["passed"] = True
    if change == "missing":
        inputs.manifest.unlink()
    else:
        inputs.manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        collector().run(args_for(inputs))
    assert runtime.encodes == [] and not inputs.output.exists()


def test_missing_student_never_falls_back_to_random_weights(inputs, runtime):
    inputs.checkpoint.unlink()
    with pytest.raises(ValueError, match="checkpoint"):
        collector().run(args_for(inputs))
    assert runtime.student_loads == runtime.native_loads == 0


def test_optional_tracking_uses_decoded_frames_and_fresh_tracker(inputs, runtime):
    args = args_for(inputs)
    args.weights = inputs.weights
    report = json.loads(collector().run(args).read_text())
    assert runtime.detector_loads == 1 and len(runtime.detector_values) == 12
    a, b = report["observations"]
    assert a["metrics"]["evaluated"] and a["metrics"] == b["metrics"]
    for row in (a, b):
        ids = {line.split(",")[1] for line in (inputs.output / row["tracks_path"]).read_text().splitlines()}
        assert ids == {"1"}


def test_missing_gt_is_unavailable_and_does_not_load_detector(inputs, runtime):
    args = args_for(inputs)
    args.weights = inputs.weights
    (inputs.root / "gt/gt.txt").unlink()
    report = json.loads(collector().run(args).read_text())
    assert runtime.detector_loads == 0
    assert report["observations"][0]["metrics"]["reason"] == "ground_truth_missing"


@pytest.mark.parametrize("field,value", [("decode_delta", -1), ("decode_delta", 1),
                                         ("encode_delta", -1), ("byte_delta", 1)])
def test_bad_frame_or_byte_accounting_is_failed_not_complete(inputs, runtime, field, value):
    setattr(runtime, field, value)
    with pytest.raises(RuntimeError):
        collector().run(args_for(inputs))
    report = json.loads((inputs.output / "feasibility_report.json").read_text())
    assert report["status"] == "failed"
    assert report["observations"] == []
    assert report["gate_authorization"] is False


def test_second_failure_retains_anchor_and_no_retry(inputs, runtime):
    runtime.fail_second = True
    with pytest.raises(RuntimeError, match="second encode"):
        collector().run(args_for(inputs))
    report = json.loads((inputs.output / "feasibility_report.json").read_text())
    assert report["status"] == "failed"
    assert [o["variant"] for o in report["observations"]] == ["anchor"]
    assert len(runtime.encodes) == 2
    attempts = [json.loads(line) for line in (inputs.output / "attempts.jsonl").read_text().splitlines()]
    assert [a["status"] for a in attempts] == ["complete", "failed"]


def test_nonfinite_residual_fails_without_altering_completed_anchor(inputs, runtime):
    runtime.residual_value = np.nan
    with pytest.raises(ValueError, match="finite"):
        collector().run(args_for(inputs))
    report = json.loads((inputs.output / "feasibility_report.json").read_text())
    assert len(report["observations"]) == 1 and report["status"] == "failed"


def test_output_cannot_overlap_source_or_existing_artifacts(inputs, runtime):
    args = args_for(inputs)
    args.output = inputs.root / "results"
    with pytest.raises(ValueError, match="output"):
        collector().run(args)
    args.output = inputs.output
    inputs.output.mkdir()
    keep = inputs.output / "keep.txt"
    keep.write_text("keep")
    with pytest.raises(ValueError, match="output"):
        collector().run(args)
    assert keep.read_text() == "keep" and runtime.native_loads == 0


def test_cli_blocked_is_json_nonzero_and_not_a_fake_observation(inputs, runtime, capsys):
    args = args_for(inputs)
    module = collector()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "build_parser", lambda: SimpleNamespace(parse_args=lambda argv: args))
        args.research_only = False
        assert module.main([]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked"
    assert report["research_only"] and report["gate_authorization"] is False
    assert not inputs.output.exists()


def test_smoke_backbone_checkpoint_cannot_produce_student_research_claim(inputs, runtime, monkeypatch):
    module = collector()
    original_init = module.ResidualStudent.__init__

    def smoke_init(self, checkpoint, device):
        original_init(self, checkpoint, device)
        self.model.backbone_name = "small_conv_smoke_only"

    monkeypatch.setattr(module.ResidualStudent, "__init__", smoke_init)
    with pytest.raises(ValueError, match="backbone"):
        module.run(args_for(inputs))
    assert runtime.encodes == []
    report = json.loads((inputs.output / "feasibility_report.json").read_text())
    assert report["status"] == "failed"


def test_source_changes_between_arms_fail_before_second_encode(inputs, runtime, monkeypatch):
    module = collector()
    evaluate = module.evaluate_pair_arm

    def change_source(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        if result["variant"] == "anchor":
            Image.fromarray(np.zeros((2, 4, 3), np.uint8)).save(inputs.root / "img1/000006.png")
        return result

    monkeypatch.setattr(module, "evaluate_pair_arm", change_source)
    with pytest.raises(ValueError, match="Source changed"):
        module.run(args_for(inputs))
    report = json.loads((inputs.output / "feasibility_report.json").read_text())
    assert report["status"] == "failed" and len(report["observations"]) == 1
    assert len(runtime.encodes) == 1
