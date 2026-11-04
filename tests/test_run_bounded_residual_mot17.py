"""Exercise the gated residual video runner without model weights or native codecs."""
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

from prism.codec.x265_bridge import BuildPin, EncoderConfig


def runner_module():
    return importlib.import_module("tier1b.run_bounded_residual_mot17")


def test_runner_exposes_separate_cli():
    runner = runner_module()
    assert runner.build_parser().prog


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "MOT17-02-FRCNN"
    (root / "img1").mkdir(parents=True)
    (root / "gt").mkdir()
    (root / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-02-FRCNN\nimDir=img1\nimExt=.png\n"
        "imWidth=4\nimHeight=2\nframeRate=25\nseqLength=3\n", encoding="utf-8")
    for index in range(1, 4):
        Image.fromarray(np.full((2, 4, 3), 100, dtype=np.uint8)).save(root / "img1" / f"{index:06d}.png")
    (root / "gt/gt.txt").write_text(
        "".join(f"{i},7,0,0,2,2,1,1,1\n" for i in range(1, 4)), encoding="utf-8")
    checkpoint, weights = tmp_path / "student_seed_0.pt", tmp_path / "yolo.pt"
    library, shim, pin_path = tmp_path / "x265.dll", tmp_path / "shim.dll", tmp_path / "pin.json"
    for path in (checkpoint, weights, library, shim):
        path.write_bytes(path.name.encode())
    pin = BuildPin(hashlib.sha256(library.read_bytes()).hexdigest(),
                   hashlib.sha256(shim.read_bytes()).hexdigest(), "test-build", 1, 200)
    pin_path.write_text(json.dumps(asdict(pin)), encoding="utf-8")
    return SimpleNamespace(root=root, checkpoint=checkpoint, weights=weights,
                           library=library, shim=shim, pin=pin, pin_path=pin_path,
                           output=tmp_path / "output", evidence=tmp_path / "evidence.json")


def options(runner, inputs, *, cap=0, variants=("anchor", "residual")):
    argv = ["--sequence-dir", str(inputs.root), "--weights", str(inputs.weights),
            "--library", str(inputs.library), "--shim", str(inputs.shim),
            "--build-pin", str(inputs.pin_path), "--output", str(inputs.output),
            "--crfs", "32", "--cap", str(cap), "--device", "cpu",
            "--variants", *variants]
    if cap:
        argv += ["--checkpoint", str(inputs.checkpoint), "--checkpoint-step", "0",
                 "--evidence", f"32={inputs.evidence}"]
    return runner.build_parser().parse_args(argv)


def write_evidence(runner, inputs, *, cap=3, crf=32):
    values = list(range(1, 101))
    config = EncoderConfig(rc_mode="CRF", crf=crf, gop=32, bframes=0, fps=25)
    context = {
        "checkpoint_sha256": hashlib.sha256(inputs.checkpoint.read_bytes()).hexdigest(),
        "seed": 0, "checkpoint_step": 0, "encoder_config": asdict(config),
        "build_pin": asdict(inputs.pin), "width": 4, "height": 2, "fps": 25, "cap": cap,
        "color_contract": "bt709_limited_yuv420p8_luma_clip_0_255",
        "rate_control_condition": {"mode": "CRF", "value": crf, "cell_qp": crf,
                                   "frame_type": "unknown", "interpretation": "crf_setting_not_effective_qp"},
    }
    cell = dict(qp=crf, frame_type="unknown", actual_bits=values, predicted_bits=values,
                actual_delta_j=values, predicted_delta_j=values, direction_delta_j=[-0.2] * 100,
                semantic_passed=True, same_codec_state=True, feasible_plus_minus_one=True,
                dependent_frames_measured=True, split="calibration")
    document = {"schema_version": 1, "protocol": "bounded_residual_crf_unknown_v1",
                "context": context, "calibration_sources": [{"source_group": "MOT17-04",
                "sha256": "c" * 64, "split": "calibration"}],
                "history": [{"step": 0, "cells": [cell]}]}
    inputs.evidence.write_text(json.dumps(document), encoding="utf-8")
    return document


@pytest.fixture
def fake_runtime(monkeypatch):
    runner = runner_module()
    state = SimpleNamespace(student_loads=0, library_loads=0, detector_loads=0,
                            predictions=0, encodes=[], detections=[], decoded_delta=0,
                            encoded_delta=0, byte_delta=0, expire_after_first=False,
                            bypass_after_first=False, closed=0)

    class FakeStudent:
        def __init__(self, checkpoint, device):
            state.student_loads += 1
            assert checkpoint.is_file() and device == "cpu"

        def predict_residual(self, rgb, crf):
            state.predictions += 1
            assert rgb.shape == (2, 4, 3) and crf == 32
            return np.full((2, 4), 2.0, dtype=np.float32)

    class FakeLibrary:
        def __init__(self, *, library, shim, pin):
            state.library_loads += 1
            self.pin = pin

    class FakeDetector:
        def __init__(self, weights, device):
            state.detector_loads += 1

        def detect(self, rgb):
            index = int(rgb[0, 0, 0])
            assert index in (1, 2, 3, 4)
            np.testing.assert_array_equal(rgb, np.full((2, 4, 3), index, np.uint8))
            state.detections.append(index)
            return np.array([[0, 0, 2, 2, 0.99]])

    class FakeCodec:
        def __init__(self, config, *, library):
            self.config = config

        def encode_sequence(self, frames_yuv, offset_generator, output_bitstream, width, height, fps):
            assert offset_generator is None
            assert (width, height, fps) == (4, 2, 25)
            rows = []
            for frame in frames_yuv:
                rows.append(frame)
                if state.expire_after_first and len(rows) == 1:
                    state.evidence.gate.verified = False
            state.encodes.append(rows)
            output_bitstream.parent.mkdir(parents=True, exist_ok=True)
            output_bitstream.write_bytes(b"bitstream" * len(rows))
            self.count = len(rows)
            return {"num_frames": len(rows) + state.encoded_delta,
                    "total_bytes": output_bitstream.stat().st_size + state.byte_delta}

        def decode_bitstream(self, path, width, height):
            try:
                for index in range(1, self.count + state.decoded_delta + 1):
                    yield np.full((height, width, 3), index, np.uint8)
            finally:
                state.closed += 1

    monkeypatch.setattr(runner, "ResidualStudent", FakeStudent)
    monkeypatch.setattr(runner, "X265Library", FakeLibrary)
    monkeypatch.setattr(runner, "YOLODetector", FakeDetector)
    monkeypatch.setattr(runner, "VideoCodecRunner", FakeCodec)
    return state


def test_validate_only_does_not_load_models_native_or_write_output(inputs, fake_runtime):
    runner = runner_module()
    args = options(runner, inputs)
    args.validate_only = True
    report = runner.run(args)
    assert report["status"] == "validated"
    assert report["runtime_build_verified"] is False
    assert report["source"]["source_group"] == "MOT17-02"
    assert not inputs.output.exists()
    assert fake_runtime.student_loads == fake_runtime.detector_loads == fake_runtime.library_loads == 0


def test_nonzero_cap_missing_evidence_blocks_before_runtime(inputs, fake_runtime):
    runner = runner_module()
    args = options(runner, inputs, cap=3)
    with pytest.raises((ValueError, FileNotFoundError), match="evidence"):
        runner.run(args)
    assert not inputs.output.exists()
    assert fake_runtime.student_loads == fake_runtime.detector_loads == fake_runtime.library_loads == 0


def test_cli_reports_blocked_without_encoding(inputs, fake_runtime, capsys):
    runner = runner_module()
    argv = ["--sequence-dir", str(inputs.root), "--weights", str(inputs.weights),
            "--library", str(inputs.library), "--shim", str(inputs.shim),
            "--build-pin", str(inputs.pin_path), "--output", str(inputs.output),
            "--cap", "1", "--crfs", "32", "--validate-only"]
    assert runner.main(argv) != 0
    message = json.loads(capsys.readouterr().out)
    assert message["status"] == "blocked"
    assert fake_runtime.encodes == []


def test_cap_zero_is_identity_without_student_and_scores_decoded_frames(inputs, fake_runtime):
    runner = runner_module()
    path = runner.run(options(runner, inputs))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["runtime_build_verified"] is True
    assert fake_runtime.student_loads == fake_runtime.predictions == 0
    assert fake_runtime.encodes[0] == fake_runtime.encodes[1]
    assert fake_runtime.detections == [1, 2, 3, 1, 2, 3]
    assert fake_runtime.closed == 2
    first, second = report["observations"]
    assert first["metrics"] == second["metrics"]
    assert second["residual_stats"]["changed_pixel_fraction"] == 0
    assert second["residual_stats"]["bypass_counts"] == {"cap_zero": 3}
    assert len((inputs.output / "attempts.jsonl").read_text().splitlines()) == 2


def test_calibrated_residual_applies_once_per_frame_and_preserves_chroma(inputs, fake_runtime):
    runner = runner_module()
    write_evidence(runner, inputs)
    path = runner.run(options(runner, inputs, cap=3))
    report = json.loads(path.read_text())
    assert report["status"] == "complete"
    assert fake_runtime.predictions == 3 and fake_runtime.student_loads == 1
    original, processed = fake_runtime.encodes
    for a, b in zip(original, processed):
        np.testing.assert_array_equal(np.frombuffer(b[:8], np.uint8).astype(int),
                                      np.frombuffer(a[:8], np.uint8).astype(int) + 2)
        assert a[8:] == b[8:]
    stats = report["observations"][1]["residual_stats"]
    assert stats["changed_pixel_fraction"] == 1
    assert stats["mean_abs_pixel_delta"] == 2
    assert stats["max_abs_pixel_delta"] == 2
    assert stats["authorized_frames"] == 3 and stats["bypass_counts"] == {}
    assert report["observations"][1]["evidence_sha256"] == hashlib.sha256(inputs.evidence.read_bytes()).hexdigest()
    assert report["observations"][1]["student_seconds"] >= 0
    assert report["observations"][1]["preprocessing_seconds"] >= 0
    assert report["observations"][1]["metrics"]["evaluated"] is True


@pytest.mark.parametrize("field", ["checkpoint_sha256", "cap", "width", "checkpoint_step"])
def test_evidence_context_mismatch_blocks_runtime(inputs, fake_runtime, field):
    runner = runner_module()
    doc = write_evidence(runner, inputs)
    doc["context"][field] = "d" * 64 if field == "checkpoint_sha256" else 999
    inputs.evidence.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        runner.run(options(runner, inputs, cap=3))
    assert fake_runtime.library_loads == fake_runtime.student_loads == 0
    assert not inputs.output.exists()


def test_declared_passing_summary_cannot_enable_residual(inputs, fake_runtime):
    runner = runner_module()
    inputs.evidence.write_text(json.dumps({"passed": True, "verified": True}))
    with pytest.raises(ValueError):
        runner.run(options(runner, inputs, cap=3))
    assert fake_runtime.encodes == []


def test_build_hash_mismatch_fails_without_native_loading(inputs, fake_runtime):
    runner = runner_module()
    inputs.library.write_bytes(b"different build")
    with pytest.raises(ValueError, match="hash"):
        runner.run(options(runner, inputs))
    assert fake_runtime.library_loads == 0


def test_existing_output_is_not_overwritten(inputs, fake_runtime):
    runner = runner_module()
    inputs.output.mkdir()
    sentinel = inputs.output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises((ValueError, FileExistsError)):
        runner.run(options(runner, inputs))
    assert sentinel.read_text() == "keep"
    assert fake_runtime.encodes == []


def test_output_inside_source_is_rejected(inputs, fake_runtime):
    runner = runner_module()
    args = options(runner, inputs)
    args.output = inputs.root / "new_results"
    with pytest.raises(ValueError, match="output"):
        runner.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize("field,delta", [("decoded_delta", -1), ("decoded_delta", 1),
                                         ("encoded_delta", -1), ("byte_delta", 1)])
def test_frame_and_byte_failures_remain_failed_reports(inputs, fake_runtime, field, delta):
    runner = runner_module()
    setattr(fake_runtime, field, delta)
    with pytest.raises(RuntimeError):
        runner.run(options(runner, inputs))
    report = json.loads((inputs.output / "residual_seq_report.json").read_text())
    assert report["status"] == "failed"
    assert report["observations"] == []
    attempts = [json.loads(line) for line in (inputs.output / "attempts.jsonl").read_text().splitlines()]
    assert attempts[-1]["status"] == "failed"


def test_missing_ground_truth_is_explicit_not_fake_score(inputs, fake_runtime):
    runner = runner_module()
    (inputs.root / "gt/gt.txt").unlink()
    path = runner.run(options(runner, inputs, variants=("anchor",)))
    row = json.loads(path.read_text())["observations"][0]
    assert row["metrics"]["evaluated"] is False
    assert row["metrics"]["HOTA"] is None


def test_gate_revocation_mid_sequence_fails_not_identity_success(inputs, fake_runtime, monkeypatch):
    runner = runner_module()
    write_evidence(runner, inputs)
    load = runner.load_residual_evidence

    def retain_gate(*args, **kwargs):
        evidence = load(*args, **kwargs)
        fake_runtime.evidence = evidence
        return evidence

    monkeypatch.setattr(runner, "load_residual_evidence", retain_gate)
    fake_runtime.expire_after_first = True
    with pytest.raises(RuntimeError, match="calibration"):
        runner.run(options(runner, inputs, cap=3, variants=("residual",)))
    report = json.loads((inputs.output / "residual_seq_report.json").read_text())
    assert report["status"] == "failed" and not report["observations"]
    assert fake_runtime.predictions == 1


def test_native_pin_failure_keeps_run_failure_record(inputs, fake_runtime, monkeypatch):
    runner = runner_module()

    def wrong_build(**kwargs):
        raise ValueError("runtime pin mismatch")

    monkeypatch.setattr(runner, "X265Library", wrong_build)
    with pytest.raises(ValueError, match="pin"):
        runner.run(options(runner, inputs))
    report = json.loads((inputs.output / "residual_seq_report.json").read_text())
    assert report["status"] == "failed"
    assert report["runtime_build_verified"] is False


def test_source_hash_and_group_do_not_depend_on_detector_suffix(inputs):
    runner = runner_module()
    first = runner.source_identity(runner.MOTSequence(inputs.root))
    info = inputs.root / "seqinfo.ini"
    info.write_text(info.read_text().replace("MOT17-02-FRCNN", "MOT17-02-DPM"))
    second = runner.source_identity(runner.MOTSequence(inputs.root))
    assert first == second
    assert first["source_group"] == "MOT17-02"
    Image.fromarray(np.zeros((2, 4, 3), np.uint8)).save(inputs.root / "img1/000003.png")
    third = runner.source_identity(runner.MOTSequence(inputs.root))
    assert third["sha256"] != first["sha256"]


def test_residual_adapter_one_forward_native_resize_and_sample_units(monkeypatch):
    torch = pytest.importorskip("torch")
    runner = runner_module()
    calls = []

    class TinyModel(torch.nn.Module):
        def forward(self, x, q, *, for_encoding):
            calls.append((x.shape, q.clone(), for_encoding, torch.is_grad_enabled()))
            torch.testing.assert_close(x, torch.ones((1, 3, 256, 256)))
            return {"residual": torch.full((1, 1, 256, 256), 2.0),
                    "delta_qp": torch.full((1, 1, 17, 30), 99.0)}

    def parent_init(self, checkpoint, device):
        self.model, self.device = TinyModel(), device

    monkeypatch.setattr(runner.Student, "__init__", parent_init)
    student = runner.ResidualStudent(Path("unused.pt"), "cpu")
    result = student.predict_residual(np.full((6, 10, 3), 255, np.uint8), 32)
    np.testing.assert_array_equal(result, np.full((6, 10), 2.0, np.float32))
    assert result.flags.c_contiguous and result.dtype == np.float32
    assert len(calls) == 1
    assert calls[0][0] == (1, 3, 256, 256)
    assert float(calls[0][1][0, 0]) == pytest.approx(32 / 51)
    assert calls[0][2:] == (False, False)


def test_statistics_are_pixel_weighted_not_frame_weighted():
    from prism.preprocessing.residual import apply_bounded_residual
    from prism.training.calibration import CalibrationCell, CalibrationGate

    runner = runner_module()
    gate = CalibrationGate({(32, "unknown")})
    values = np.arange(1, 101, dtype=float)
    cell = CalibrationCell(32, "unknown", values, values, values, values,
                           np.full(100, -0.2), True, True, True, True)
    gate.assess(0, [cell], lambda: None)
    totals = runner.ResidualTotals()
    _, changed = apply_bounded_residual(bytes([100] * 6), np.ones((2, 2)), 2, 2,
                                        cap=1, gate=gate, qp=32)
    _, unchanged = apply_bounded_residual(bytes([100] * 12), np.zeros((2, 4)), 4, 2)
    totals.add(changed)
    totals.add(unchanged)
    result = totals.report()
    assert result["changed_pixel_fraction"] == pytest.approx(1 / 3)
    assert result["mean_pixel_delta"] == pytest.approx(1 / 3)
    assert result["authorized_frames"] == 1
    assert result["bypass_counts"] == {"cap_zero": 1}


def test_partial_report_retains_completed_anchor_when_residual_fails(inputs, fake_runtime, monkeypatch):
    runner = runner_module()
    write_evidence(runner, inputs)

    def bad_prediction(self, rgb, crf):
        return np.full((2, 4), np.nan)

    monkeypatch.setattr(runner.ResidualStudent, "predict_residual", bad_prediction)
    with pytest.raises(ValueError, match="finite"):
        runner.run(options(runner, inputs, cap=3))
    report = json.loads((inputs.output / "residual_seq_report.json").read_text())
    assert report["status"] == "failed"
    assert [row["variant"] for row in report["observations"]] == ["anchor"]
    attempts = [json.loads(line) for line in (inputs.output / "attempts.jsonl").read_text().splitlines()]
    assert [row["status"] for row in attempts] == ["complete", "failed"]


def test_validated_active_evidence_does_not_load_models_or_native(inputs, fake_runtime):
    runner = runner_module()
    write_evidence(runner, inputs)
    args = options(runner, inputs, cap=3)
    args.validate_only = True
    report = runner.run(args)
    assert report["status"] == "validated"
    assert report["evidence_sha256"]["32"]
    assert fake_runtime.student_loads == fake_runtime.library_loads == fake_runtime.detector_loads == 0
    assert not inputs.output.exists()


def test_max_frame_prefix_retains_whole_source_identity(inputs, fake_runtime):
    runner = runner_module()
    args = options(runner, inputs)
    args.validate_only = True
    whole = runner.run(args)
    args.max_frames = 1
    prefix = runner.run(args)
    assert whole["source"] == prefix["source"]
    assert prefix["frame_interval"] == [1, 1]


def test_existing_checkpoint_options_cannot_be_silently_ignored(inputs, fake_runtime):
    runner = runner_module()
    args = options(runner, inputs)
    args.checkpoint = inputs.checkpoint
    with pytest.raises(ValueError, match="only valid"):
        runner.run(args)


def test_duplicate_evidence_crf_rejected_before_loading_runtime(inputs, fake_runtime):
    runner = runner_module()
    args = options(runner, inputs, cap=3)
    args.evidence.append(args.evidence[0])
    with pytest.raises(ValueError, match="distinct"):
        runner.run(args)


def test_validation_rejects_source_frame_path_escape(inputs, fake_runtime):
    runner = runner_module()
    info = inputs.root / "seqinfo.ini"
    escaped = inputs.root.parent / "outside"
    escaped.mkdir()
    for index in range(1, 4):
        (escaped / f"{index:06d}.png").write_bytes((inputs.root / "img1" / f"{index:06d}.png").read_bytes())
    info.write_text(info.read_text().replace("imDir=img1", "imDir=../outside"))
    with pytest.raises(ValueError, match="outside"):
        runner.run(options(runner, inputs))
