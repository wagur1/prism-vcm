"""Exercise real frame/ablation/tracking/metric plumbing without GPU or Kaggle."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest

from tier1b import run_tier1b_mot17 as runner
from scripts import dispatch_pool_mot17 as dispatch


@pytest.fixture
def sequence(tmp_path):
    root = tmp_path / "MOT17-02-FRCNN"
    (root / "img1").mkdir(parents=True)
    (root / "gt").mkdir()
    (root / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-02-FRCNN\nimDir=img1\nimExt=.png\n"
        "imWidth=66\nimHeight=70\nframeRate=25\nseqLength=3\n", encoding="utf-8")
    rng = np.random.default_rng(7)
    for index in range(1, 4):
        Image.fromarray(rng.integers(0, 256, (70, 66, 3), dtype=np.uint8)).save(
            root / "img1" / f"{index:06d}.png")
    (root / "gt/gt.txt").write_text(
        "".join(f"{i},7,10,12,20,30,1,1,1\n" for i in range(1, 4)), encoding="utf-8")
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    (checkpoint / "student_seed_0.pt").write_bytes(b"mock student")
    weights = tmp_path / "yolo11l.pt"
    weights.write_bytes(b"mock detector")
    return runner.build_parser().parse_args([
        "--sequence-dir", str(root), "--checkpoints-dir", str(checkpoint),
        "--weights", str(weights), "--output", str(tmp_path / "output"),
        "--device", "cpu", "--crfs", "22", "37",
    ])


@pytest.fixture
def models(monkeypatch):
    state = SimpleNamespace(encodes=[], student_calls=[], detector_calls=[],
                            decoded_delta=0, empty_frame=None, closed=0)

    class MockStudent:
        def __init__(self, path, device):
            assert path.name == "student_seed_0.pt"
            assert device == "cpu"

        def spatial_qp(self, rgb, crf):
            assert rgb.shape == (70, 66, 3)
            state.student_calls.append((rgb.copy(), crf))
            # Nonuniform, fractional map spanning foreground/background.
            return np.tile(np.linspace(-3.5, 4.5, 8, dtype=np.float32), (8, 1))

    class MockDetector:
        def __init__(self, weights, device):
            assert weights.name == "yolo11l.pt"
            assert device == "cpu"

        def detect(self, rgb):
            index = int(rgb[0, 0, 0])
            # The codec emits these distinct frames; never score source images.
            np.testing.assert_array_equal(rgb, np.full((70, 66, 3), index, np.uint8))
            state.detector_calls.append(index)
            if state.empty_frame == index:
                return np.empty((0, 5))
            return np.array([[10, 12, 30, 42, .2 if index == 2 else .95]])

    class MockCodec:
        def __init__(self, config):
            self.config = config

        def encode_sequence(self, frames_yuv, offset_generator, output_bitstream, width, height, fps):
            rows = []
            assert (width, height, fps) == (66, 70, 25)
            for index, frame in enumerate(frames_yuv):
                assert len(frame) == width * height * 3 // 2
                offsets = offset_generator(index, width, height)
                if offsets is not None:
                    assert offsets.shape == (8, 8)
                    assert offsets.dtype == np.float32
                    assert offsets.flags.c_contiguous
                rows.append((frame, None if offsets is None else offsets.copy()))
            output_bitstream.parent.mkdir(parents=True, exist_ok=True)
            output_bitstream.write_bytes(b"HEVC" * (self.config.crf + len(rows)))
            state.encodes.append({"path": output_bitstream, "rows": rows, "config": self.config})
            self.count = len(rows)
            return {"total_bytes": output_bitstream.stat().st_size, "num_frames": self.count,
                    "bitrate_kbps": output_bitstream.stat().st_size * 8 * fps / self.count / 1000,
                    "bitstream_path": output_bitstream}

        def decode_bitstream(self, path, width, height):
            assert path == state.encodes[-1]["path"]
            try:
                for index in range(1, self.count + state.decoded_delta + 1):
                    yield np.full((height, width, 3), index, np.uint8)
            finally:
                state.closed += 1

    monkeypatch.setattr(runner, "Student", MockStudent)
    monkeypatch.setattr(runner, "YOLODetector", MockDetector)
    monkeypatch.setattr(runner, "VideoCodecRunner", MockCodec)
    return state


@pytest.mark.parametrize("bframes", [0, 3])
def test_all_variants_end_to_end(sequence, models, bframes):
    sequence.bframes = bframes
    report_path = runner.run(sequence)
    report = json.loads(report_path.read_text())
    assert report_path == sequence.output / "mot17_seq_report.json"
    assert report["status"] == "complete"
    assert report["sequence"] == "MOT17-02-FRCNN"
    assert report["prediction_structure"] == ("LDP" if bframes == 0 else "RA")
    assert report["frame_interval"] == [1, 3]
    assert report["duration_seconds"] == pytest.approx(3 / 25)
    assert report["metric_scale"] == "percent"
    assert len(report["checkpoint_sha256"]) == len(report["weights_sha256"]) == 64
    assert len(report["observations"]) == len(models.encodes) == 12
    assert len(models.student_calls) == len(models.detector_calls) == 36
    assert models.closed == 12
    for row in report["observations"]:
        assert row["num_frames"] == row["track_rows"] == 3
        assert row["bitrate_kbps"] == pytest.approx(row["bits"] * 25 / 3 / 1000)
        assert row["bits"] == row["total_bytes"] * 8
        assert row["encoder_config"]["bframes"] == bframes
        assert row["encoder_config"]["gop"] == 32
        assert row["encoder_config"]["rc_mode"] == "CRF"
        assert row["encoder_config"]["crf"] == row["crf"]
        assert row["metrics"]["evaluated"]
        for metric in ("HOTA", "MOTA", "IDF1", "DetA", "AssA"):
            assert row["metrics"][metric] == pytest.approx(100)
        lines = (sequence.output / row["tracks_path"]).read_text().splitlines()
        # Fresh IDs per operating point; low-confidence frame 2 rescues ID 1.
        assert [line.split(",")[:2] for line in lines] == [[str(i), "1"] for i in range(1, 4)]
        assert all(len(line.split(",")) == 10 for line in lines)
        assert (sequence.output / row["bitstream_path"]).is_file()
    assert not (sequence.output / ".mot17.lock").exists()


def test_prefilter_is_identical_and_does_not_send_offsets_for_filter_only(sequence, models):
    sequence.crfs = [22]
    runner.run(sequence)
    points = {row["path"].stem.removesuffix("_crf22"): row["rows"] for row in models.encodes}
    for i in range(3):
        anchor, no_map = points["anchor"][i]
        filtered, no_map_filter = points["M2_filter_only"][i]
        combined, combined_offsets = points["M2_asym_prefilter"][i]
        assert no_map is no_map_filter is None
        assert filtered == combined
        assert anchor != filtered
        np.testing.assert_array_equal(combined_offsets, points["M2_asym_ratecomp"][i][1])
        assert points["M2_asym"][i][0] == anchor
        assert points["M2_zero_mean"][i][0] == anchor
        assert np.any(np.abs(combined_offsets - np.rint(combined_offsets)) > .01)
        # Cells on the left have negative centered QPs and must stay bit-exact.
        before = np.frombuffer(anchor, np.uint8)[:66 * 70].reshape(70, 66)
        after = np.frombuffer(filtered, np.uint8)[:66 * 70].reshape(70, 66)
        np.testing.assert_array_equal(before[:, :16], after[:, :16])


def test_max_frames_trims_gt_and_never_reads_later_frames(sequence, models):
    sequence.max_frames = 2
    sequence.variants = ["anchor"]
    sequence.crfs = [22]
    (sequence.sequence_dir / "img1/000003.png").unlink()
    report = json.loads(runner.run(sequence).read_text())
    assert report["num_frames"] == 2
    assert report["sequence_frames"] == 3
    assert report["frame_interval"] == [1, 2]
    assert len(models.student_calls) == 2
    assert len((sequence.output / "evaluated_gt.txt").read_text().splitlines()) == 2
    assert len((sequence.sequence_dir / "gt/gt.txt").read_text().splitlines()) == 3
    assert report["observations"][0]["metrics"]["MOTA"] == 100
    assert report["observations"][0]["bitrate_kbps"] == pytest.approx(24 * 4 * 8 * 25 / 2 / 1000)


def test_no_ground_truth_keeps_tracks_and_null_metrics(sequence, models):
    (sequence.sequence_dir / "gt/gt.txt").unlink()
    sequence.variants = ["anchor"]
    sequence.crfs = [22]
    sequence.max_frames = 100
    report = json.loads(runner.run(sequence).read_text())
    assert report["ground_truth_available"] is False
    assert report["num_frames"] == 3
    metrics = report["observations"][0]["metrics"]
    assert not metrics["evaluated"]
    assert all(metrics[key] is None for key in ("HOTA", "MOTA", "IDF1", "DetA", "AssA"))
    assert report["observations"][0]["track_rows"] == 3


def test_empty_detections_advance_tracker_and_preserve_frame_numbers(sequence, models):
    sequence.variants = ["anchor"]
    sequence.crfs = [22]
    models.empty_frame = 2
    report = json.loads(runner.run(sequence).read_text())
    row = report["observations"][0]
    lines = (sequence.output / row["tracks_path"]).read_text().splitlines()
    assert [line.split(",")[:2] for line in lines] == [["1", "1"], ["3", "1"]]
    assert row["metrics"]["MOTA"] == pytest.approx(200 / 3)


@pytest.mark.parametrize("delta", [-1, 1])
def test_decode_count_mismatch_fails_without_reporting_scores(sequence, models, delta):
    sequence.variants = ["anchor"]
    sequence.crfs = [22]
    models.decoded_delta = delta
    with pytest.raises(RuntimeError, match="Decoded frame count"):
        runner.run(sequence)
    report = json.loads((sequence.output / "mot17_seq_report.json").read_text())
    assert report["status"] == "failed"
    assert report["observations"] == []
    assert models.closed == 1
    assert not (sequence.output / ".mot17.lock").exists()


def test_missing_frame_fails(sequence, models):
    (sequence.sequence_dir / "img1/000002.png").unlink()
    with pytest.raises(FileNotFoundError, match="Missing frame 2"):
        runner.run(sequence)
    assert json.loads((sequence.output / "mot17_seq_report.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("field,value", [
    ("crfs", []), ("crfs", [22, 22]), ("crfs", [-1]), ("crfs", [52]),
    ("variants", []), ("variants", ["bogus"]), ("variants", ["anchor", "anchor"]),
    ("gop", 0), ("bframes", 1), ("max_frames", 0), ("max_frames", -1), ("device", "tpu"),
])
def test_invalid_options_rejected_before_loading_models(sequence, models, field, value):
    setattr(sequence, field, value)
    with pytest.raises(ValueError):
        runner.run(sequence)
    assert not models.student_calls


def test_cli_defaults_and_standalone_help(tmp_path):
    args = runner.build_parser().parse_args([
        "--sequence-dir", "seq", "--checkpoints-dir", "ckpt", "--weights", "yolo11l.pt",
        "--output", "output",
    ])
    assert args.crfs == [22, 27, 32, 37]
    assert args.variants == list(runner.VARIANTS)
    assert (args.gop, args.bframes, args.max_frames, args.device) == (32, 0, None, "cuda")
    result = subprocess.run([sys.executable, str(Path(runner.__file__)), "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--sequence-dir" in result.stdout


def test_yolo_adapter_uses_rgb_person_filter_and_tracker_low_threshold(monkeypatch, tmp_path):
    class Tensor:
        def __init__(self, data):
            self.data = np.asarray(data)
        def cpu(self):
            return self
        def numpy(self):
            return self.data

    class YOLO:
        def __init__(self, path):
            assert path.endswith("yolo11l.pt")
        def predict(self, image, **kwargs):
            assert isinstance(image, Image.Image) and image.mode == "RGB"
            assert image.getpixel((0, 0)) == (250, 10, 20)
            assert kwargs["classes"] == [0] and kwargs["conf"] == .1
            boxes = SimpleNamespace(cls=Tensor([0, 2]), xyxy=Tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
                                    conf=Tensor([.2, .9]))
            return [SimpleNamespace(boxes=boxes)]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=YOLO))
    detector = runner.YOLODetector(tmp_path / "yolo11l.pt", "cpu")
    result = detector.detect(np.full((8, 8, 3), [250, 10, 20], np.uint8))
    np.testing.assert_allclose(result, [[1, 2, 3, 4, .2]])


def account(index):
    return {"username": f"user{index}", "key": "secret-not-for-kernels", "dir": Path(f"pool/user{index}")}


@pytest.mark.parametrize("status,idle", [("complete", True), ("error", True), ("cancelled", True),
                                        ("running", False), ("queued", False), ("unknown", False)])
def test_account_activity_checks_kernel_status(monkeypatch, status, idle):
    def fake_cli(acc, args):
        if args[1] == "list":
            assert "--mine" in args and "--csv" in args
            return "ref,title\nuser0/existing,existing\n"
        assert args == ["kernels", "status", "user0/existing"]
        return f'user0/existing has status "KernelWorkerStatus.{status.upper()}"'
    monkeypatch.setattr(dispatch, "kaggle", fake_cli)
    assert dispatch.account_activity(account(0))["idle"] is idle


def test_api_failure_is_not_idle(monkeypatch):
    def fail(*args):
        raise RuntimeError("authentication failed")
    monkeypatch.setattr(dispatch, "kaggle", fail)
    result = dispatch.account_activity(account(0))
    assert not result["idle"]
    assert "authentication failed" in result["error"]


@pytest.mark.parametrize("listing", ["Not found\n", "No kernels found\n", "ref,title\n"])
def test_account_without_kernels_is_idle(monkeypatch, listing):
    monkeypatch.setattr(dispatch, "kaggle", lambda *args: listing)
    assert dispatch.account_activity(account(0))["idle"]


def test_paginated_listing_detects_busy_kernel_on_later_page(monkeypatch):
    pages = []
    def cli(acc, args):
        if args[1] == "list":
            page = int(args[-1])
            pages.append(page)
            if page == 1:
                return "Next Page Token = opaque\nref,title\n" + "".join(
                    f"user0/job{i},job\n" for i in range(100))
            return "ref,title\nuser0/busy,busy\n"
        status = "running" if args[-1] == "user0/busy" else "complete"
        return f'{args[-1]} has status "{status}"'
    monkeypatch.setattr(dispatch, "kaggle", cli)
    assert not dispatch.account_activity(account(0))["idle"]
    assert pages == [1, 2]


def test_selection_skips_busy_and_duplicate_accounts(monkeypatch):
    monkeypatch.setattr(dispatch, "account_activity", lambda a: {"idle": a["username"] != "user0"})
    accounts = [account(0), account(1), account(1), *[account(i) for i in range(2, 8)]]
    assert [a["username"] for a in dispatch.select_idle_accounts(accounts)] == [f"user{i}" for i in range(1, 8)]
    with pytest.raises(RuntimeError, match="Need 7 distinct idle"):
        dispatch.select_idle_accounts(accounts[:-1])


def test_bundle_kernel_metadata_and_generated_python(tmp_path):
    bundle = dispatch.source_bundle()
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(bundle[0]))) as archive:
        assert "tier1b/run_tier1b_mot17.py" in archive.namelist()
        assert "prism/video/video_codec.py" in archive.namelist()
        assert "prism/codec/native/CMakeLists.txt" in archive.namelist()
        assert b"\r\n" not in archive.read("gate0/build_gate0_stack.sh")
        assert not any("kaggle.json" in name for name in archive.namelist())
    options = dict(crfs=[22, 37], variants=list(runner.VARIANTS), gop=32, bframes=3, max_frames=10)
    row = dispatch.generate_sequence_kernel(dispatch.SEQUENCES[0], "user0", tmp_path, bundle, options)
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert row["kernel_id"] == "user0/prism-tier1b-mot17-02-frcnn"
    assert metadata["dataset_sources"] == ["ahmedsamir1598/mot17challenge"]
    assert metadata["kernel_sources"] == ["htran123456/prism-tier1a-step3"]
    assert metadata["enable_gpu"] and metadata["is_private"]
    code = (tmp_path / metadata["code_file"]).read_text()
    compile(code, metadata["code_file"], "exec")
    assert "PRISM_X265_LIBRARY" in code and "PRISM_X265_SHIM" in code
    assert "--sequence-dir" in code and "--max-frames" in code
    assert "secret-not-for-kernels" not in code


def test_dispatch_dry_run_push_and_saved_status(tmp_path, monkeypatch, capsys):
    accounts = [account(i) for i in range(7)]
    monkeypatch.setattr(dispatch, "discover_pool_accounts", lambda p: accounts)
    monkeypatch.setattr(dispatch, "account_activity", lambda a: {"idle": True})
    monkeypatch.setattr(dispatch, "source_bundle", lambda: ("", "digest", "revision"))
    calls = []
    monkeypatch.setattr(dispatch, "kaggle", lambda a, args: calls.append((a["username"], args)) or "")
    args = ["--kernels-dir", str(tmp_path), "--push", "--dry-run"]
    assert dispatch.main(args) == 0
    assert calls == []
    manifest = json.loads((tmp_path / "dispatch_manifest.json").read_text())
    rows = manifest["assignments"]
    assert len(rows) == len({row["username"] for row in rows}) == 7
    assert [row["sequence"] for row in rows] == list(dispatch.SEQUENCES)
    assert "secret-not-for-kernels" not in (tmp_path / "dispatch_manifest.json").read_text()
    assert dispatch.main(args[:-1]) == 0
    assert len(calls) == 7 and all(c[1][:2] == ["kernels", "push"] for c in calls)
    assert dispatch.main(args[:-1]) == 0  # No duplicate submissions.
    assert len(calls) == 7
    monkeypatch.setattr(dispatch, "select_idle_accounts", lambda a: pytest.fail("status remapped accounts"))
    monkeypatch.setattr(dispatch, "check_kernel_status", lambda a, k: "running")
    assert dispatch.main(["--kernels-dir", str(tmp_path), "--status"]) == 0
    assert "running" in capsys.readouterr().out


def test_push_rechecks_idle_before_submission(tmp_path, monkeypatch):
    accounts = [account(i) for i in range(7)]
    monkeypatch.setattr(dispatch, "discover_pool_accounts", lambda p: accounts)
    monkeypatch.setattr(dispatch, "select_idle_accounts", lambda a: accounts)
    monkeypatch.setattr(dispatch, "source_bundle", lambda: ("", "digest", "revision"))
    monkeypatch.setattr(dispatch, "account_activity", lambda a: {"idle": False})
    monkeypatch.setattr(dispatch, "kaggle", lambda *a: pytest.fail("pushed onto busy account"))
    assert dispatch.main(["--kernels-dir", str(tmp_path), "--push"]) == 1


def test_kaggle_credentials_are_scoped_and_not_printed(monkeypatch):
    monkeypatch.setenv("KAGGLE_API_TOKEN", "ambient-token")
    monkeypatch.setenv("KAGGLE_USERNAME", "ambient-user")
    monkeypatch.setenv("KAGGLE_KEY", "ambient-key")
    def fake_run(cmd, **kwargs):
        assert kwargs["env"]["KAGGLE_USERNAME"] == "user0"
        assert kwargs["env"]["KAGGLE_KEY"] == "secret-not-for-kernels"
        assert "KAGGLE_API_TOKEN" not in kwargs["env"]
        assert "secret-not-for-kernels" not in " ".join(cmd)
        return SimpleNamespace(returncode=1, stdout="", stderr="secret-not-for-kernels")
    monkeypatch.setattr(dispatch.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as exc:
        dispatch.kaggle(account(0), ["kernels", "list"])
    assert "secret-not-for-kernels" not in str(exc.value)
