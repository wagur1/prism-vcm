"""Standalone diagnostics: numeric controls, selected-source identity, lazy CLI."""
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_diagnostics_modules_exist():
    for name in ("prism.evaluation.controller_diagnostics", "scripts.diagnose_controller_outputs"):
        assert importlib.util.find_spec(name) is not None, f"{name} is not implemented"


@pytest.fixture
def core():
    name = "prism.evaluation.controller_diagnostics"
    assert importlib.util.find_spec(name) is not None, "diagnostic core is not implemented"
    return importlib.import_module(name)


@pytest.fixture
def cli():
    name = "scripts.diagnose_controller_outputs"
    assert importlib.util.find_spec(name) is not None, "diagnostic CLI is not implemented"
    return importlib.import_module(name)


def test_numeric_statistics(core):
    got = core.summarize_numeric(np.array([0, 1, 2, 3, 4], dtype=np.float32))
    assert got == pytest.approx(dict(min=0, max=4, mean=2, std=np.sqrt(2),
                                    p01=.04, p50=2, p99=3.96))


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf], [-np.inf]])
def test_numeric_rejects_nonfinite_or_empty(core, bad):
    with pytest.raises(ValueError):
        core.summarize_numeric(bad)


def test_bounded_is_float_map_not_quantized_pixel_claim(core):
    raw = np.array([[-8, -1, -.5, 0, .5, 1, 8]], dtype=np.float32)
    got = core.bounded_output_summary(raw, 1)
    assert got["fraction_below_negative_cap"] == pytest.approx(1 / 7)
    assert got["fraction_above_cap"] == pytest.approx(1 / 7)
    assert got["fraction_abs_ge_7_99"] == pytest.approx(2 / 7)
    assert got["fraction_gt_half"] == pytest.approx(2 / 7)
    assert got["fraction_lt_negative_half"] == pytest.approx(2 / 7)
    assert got["native"]["mean"] == 0
    assert got["after_clip"]["std"] == pytest.approx(np.sqrt(4.5 / 7))
    assert got["after_clip"]["min"] == -1
    assert got["after_clip"]["max"] == 1
    assert core.bounded_output_summary(raw, 0)["after_clip"]["std"] == 0


def test_tanh_fullscale_statistics(core):
    residual = (8 * np.tanh(np.array([[-20, 0, 20]], dtype=np.float32)))
    got = core.bounded_output_summary(residual, 1)
    assert got["native"]["min"] == -8
    assert got["native"]["max"] == 8
    assert got["fraction_abs_ge_7_99"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("cap", [-1, np.nan, np.inf])
def test_invalid_caps(core, cap):
    with pytest.raises(ValueError):
        core.bounded_output_summary(np.zeros((2, 3)), cap)


def test_qp_factorial_projection_and_all_pairs(core):
    raw = np.tile([-20, -3, 2, 20], (4, 1)).astype(np.float32)
    before = raw.copy()
    got = core.paired_qp_summary(raw, 64, 64)
    assert set(got["states"]) == {f"{cap}_{state}" for cap in ("sym6", "asym46", "sym5")
                                  for state in ("OFF", "ON")}
    assert len(got["pairs"]) == 15
    for cap, low, high in (("sym6", -6, 6), ("asym46", -4, 6), ("sym5", -5, 5)):
        off, on = (got["states"][f"{cap}_{state}"] for state in ("OFF", "ON"))
        assert off["dtype"] == on["dtype"] == "float32"
        assert off["area_weighted_mean"] == pytest.approx(0, abs=1e-6)
        assert on["area_weighted_mean"] > 0
        assert off["min"] >= low and off["max"] <= high
        assert off["fraction_at_cap"] > 0
        pair = got["pairs"][f"{cap}_OFF__{cap}_ON"]
        assert pair["area_weighted_mae"] > 0
        assert pair["max_abs_difference"] >= pair["area_weighted_mae"]
        assert pair["exact_equal"] is False
    np.testing.assert_array_equal(raw, before)


def test_qp_padding_is_excluded_and_fractional_area_used(core):
    # Valid area is 16x16 plus 2x16, all remaining CTU-padded cells are invisible.
    raw = np.full((4, 4), 1e6, dtype=np.float32)
    raw[0, :2] = [-1, 4]
    got = core.paired_qp_summary(raw, 18, 16)
    off = got["states"]["sym6_OFF"]
    assert off["area_weighted_mean"] == pytest.approx(0, abs=1e-6)
    assert off["min"] == pytest.approx(-5 / 9)
    assert off["max"] == pytest.approx(40 / 9)
    assert off["area_weighted_std"] == pytest.approx(np.sqrt(200 / 81))
    assert off["fraction_at_cap"] == 0
    uniform = core.paired_qp_summary(np.full((4, 4), 7, np.float32), 18, 16)
    assert uniform["states"]["sym6_ON"]["area_weighted_mean"] == 0


@pytest.mark.parametrize("width,height,shape", [(0, 16, (4, 4)), (16, -1, (4, 4)),
                                               (18, 16, (2, 3)), (16.5, 16, (4, 4))])
def test_qp_invalid_geometry(core, width, height, shape):
    with pytest.raises(ValueError):
        core.paired_qp_summary(np.zeros(shape), width, height)


def test_help_and_core_import_do_not_import_torch():
    root = Path(__file__).resolve().parents[1]
    code = ("import sys,runpy; "
            "import prism.evaluation.controller_diagnostics; "
            "assert 'torch' not in sys.modules and 'timm' not in sys.modules; "
            "sys.modules['torch']=None; sys.modules['timm']=None; "
            "sys.argv=['diagnose_controller_outputs.py','--help']; "
            "runpy.run_path('scripts/diagnose_controller_outputs.py',run_name='__main__')")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--frame-indices" in result.stdout
    assert "--checkpoint" in result.stdout


def test_parser_defaults(cli):
    args = cli.parse_args(["--sequence-dir", "s", "--checkpoint", "c", "--output", "o"])
    assert args.frame_indices == [1, 16, 33, 40, 48, 64]
    assert args.crfs == [22, 32, 37]
    assert args.cap == 1 and args.device == "cpu"


@pytest.mark.parametrize("option", [["--cap", "-1"], ["--cap", "nan"], ["--cap", "inf"],
                                    ["--frame-indices", "0"], ["--frame-indices", "1", "1"],
                                    ["--crfs", "52"], ["--crfs", "22", "22"],
                                    ["--crfs", "1.5"], ["--device", "bogus"]])
def test_parser_rejects_invalid_without_gpu(cli, option):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["--sequence-dir", "s", "--checkpoint", "c", "--output", "o"] + option)
    assert exc.value.code == 2


def make_sequence(tmp_path, size=(18, 16)):
    from PIL import Image
    sequence = tmp_path / "sequence"
    (sequence / "img1").mkdir(parents=True)
    (sequence / "seqinfo.ini").write_text(
        "[Sequence]\nname=fixture\nimWidth=18\nimHeight=16\nseqLength=600\n"
        "frameRate=30\nimDir=img1\nimExt=.png\n", encoding="utf-8")
    Image.new("RGB", size, (237, 11, 67)).save(sequence / "img1/000040.png")
    return sequence


def test_selected_frames_reuses_original_color_without_prefix(cli, tmp_path, monkeypatch):
    import hashlib
    from prism.video.mot_eval import MOTSequence
    from tier1a.oracle_codec import yuv_to_rgb
    sequence = make_sequence(tmp_path)
    path = sequence / "img1/000040.png"
    monkeypatch.setattr(MOTSequence, "iter_frame_paths", lambda self: iter([path]))
    expected = yuv_to_rgb(next(MOTSequence(sequence).iter_yuv_frames(18, 16)), 18, 16)
    # Prohibit full-sequence enumeration in the diagnostic.
    monkeypatch.setattr(MOTSequence, "iter_frame_paths", lambda self: pytest.fail("full scan"))
    index, actual_path, digest, rgb = next(cli.selected_frames(sequence, [40]))
    assert index == 40 and actual_path == path
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    np.testing.assert_array_equal(rgb, expected)


@pytest.mark.parametrize("indices,error", [([1], FileNotFoundError), ([601], ValueError)])
def test_selected_source_errors(cli, tmp_path, indices, error):
    with pytest.raises(error):
        list(cli.selected_frames(make_sequence(tmp_path), indices))


def test_selected_source_dimension_mismatch(cli, tmp_path):
    with pytest.raises(ValueError, match="dimensions"):
        list(cli.selected_frames(make_sequence(tmp_path, (16, 16)), [40]))


def test_atomic_report_rejects_nan_and_preserves_previous(cli, tmp_path):
    path = tmp_path / "report.json"
    cli.write_report(path, {"status": "running"})
    with pytest.raises(ValueError):
        cli.write_report(path, {"bad": float("nan")})
    assert json.loads(path.read_text())["status"] == "running"
    assert list(tmp_path.iterdir()) == [path]


def test_run_writes_failure_status_without_model(cli, tmp_path, monkeypatch):
    args = cli.parse_args(["--sequence-dir", str(tmp_path / "missing"), "--checkpoint",
                           str(tmp_path / "missing.pt"), "--output", str(tmp_path / "out")])
    monkeypatch.setattr(cli, "load_student", lambda *a: pytest.fail("unexpected model load"))
    assert cli.run(args) == 1
    report = json.loads((args.output / "controller_diagnostics.json").read_text())
    assert report["status"] == "failed" and report["error"]
    assert report["scope"] == "selected_frames_only_not_full_benchmark"
    assert report["records"] == []


def test_one_forward_hook_captures_logits_and_closes_on_error(cli):
    torch = pytest.importorskip("torch")
    from torch.nn import functional as F
    from types import SimpleNamespace

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.residual_head = torch.nn.Identity()
            self.calls = 0
            self.fail = False

        def forward(self, x, q, *, for_encoding):
            self.calls += 1
            assert x.shape == (1, 3, 256, 256) and not for_encoding
            assert float(x.min()) == float(x.max()) == 1
            assert float(q[0, 0]) == pytest.approx(32 / 51)
            raw = self.residual_head(torch.tensor([[[[-2., 2.], [0., 1.]]]]))
            if self.fail:
                raise RuntimeError("fixture forward failure")
            return {"residual": 8 * torch.tanh(F.interpolate(raw, (256, 256),
                     mode="bilinear", align_corners=False)), "delta_qp": torch.full((1, 1, 17, 30), 6.)}

    model = Tiny().eval()
    student = SimpleNamespace(model=model, device="cpu")
    got = cli.infer_frame(student, np.full((16, 18, 3), 255, np.uint8), 32, 1)
    assert model.calls == 1 and not model.residual_head._forward_hooks
    assert got["residual_head_pre_resize_pre_tanh"]["min"] == -2
    assert got["residual_head_pre_resize_pre_tanh"]["mean"] == .25
    assert got["conditioned_features"]["mean"] == .25
    assert got["tanh_reconstruction_max_abs_error"] == 0
    assert got["residual_native"]["after_clip"]["min"] == -1
    assert got["residual_native"]["after_clip"]["max"] == 1
    assert got["delta_qp_17x30"]["fraction_abs_ge_5_99"] == 1
    assert got["delta_qp_17x30"]["mean"] == 6
    assert got["qp_controls"]["shape"] == [4, 4]
    model.fail = True
    with pytest.raises(RuntimeError, match="fixture forward failure"):
        cli.infer_frame(student, np.full((16, 18, 3), 255, np.uint8), 32, 1)
    assert model.calls == 2 and not model.residual_head._forward_hooks


@pytest.mark.parametrize("field,value", [("cap", -1), ("cap", float("nan")),
                                          ("frame_indices", [0]), ("frame_indices", [1, 1]),
                                          ("frame_indices", [True]), ("frame_indices", []),
                                          ("crfs", [52]), ("crfs", [22.5]), ("crfs", []),
                                          ("device", "invalid")])
def test_run_validates_direct_arguments_before_mkdir(cli, tmp_path, field, value):
    args = cli.parse_args(["--sequence-dir", str(tmp_path / "seq"), "--checkpoint", "c",
                           "--output", str(tmp_path / "out")])
    setattr(args, field, value)
    with pytest.raises(ValueError):
        cli.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize("relationship", ["sequence_child", "sequence_parent", "checkpoint_child",
                                         "checkpoint_parent", "sequence_equal", "checkpoint_equal"])
def test_run_rejects_overlapping_output_before_mkdir(cli, tmp_path, relationship):
    seq, checkpoint, out = tmp_path / "seq", tmp_path / "weights.pt", tmp_path / "out"
    if relationship == "sequence_child":
        out = seq / "diagnostics"
    elif relationship == "sequence_parent":
        seq = out / "seq"
    elif relationship == "checkpoint_child":
        out = checkpoint / "diagnostics"
    elif relationship == "checkpoint_parent":
        checkpoint = out / "weights.pt"
    elif relationship == "sequence_equal":
        out = seq
    else:
        out = checkpoint
    args = cli.parse_args(["--sequence-dir", str(seq), "--checkpoint", str(checkpoint),
                           "--output", str(tmp_path / "initial")])
    args.output = out
    with pytest.raises(ValueError, match="overlap"):
        cli.run(args)
    assert not out.exists()


def test_selected_path_cannot_escape_source(cli, tmp_path):
    sequence = make_sequence(tmp_path)
    info = sequence / "seqinfo.ini"
    info.write_text(info.read_text().replace("imDir=img1", "imDir=../outside"))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "000040.png").write_bytes((sequence / "img1/000040.png").read_bytes())
    with pytest.raises(ValueError, match="outside"):
        list(cli.selected_frames(sequence, [40]))


def test_run_mocked_inference_reports_each_selected_frame_crf(cli, tmp_path, monkeypatch):
    from types import SimpleNamespace
    sequence = make_sequence(tmp_path)
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"test identity, not a model")
    args = cli.parse_args(["--sequence-dir", str(sequence), "--checkpoint", str(checkpoint),
                           "--output", str(tmp_path / "out"), "--frame-indices", "40",
                           "--crfs", "22", "32"])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="fake"))
    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(__version__="fake"))
    monkeypatch.setattr(cli, "load_student", lambda *a: SimpleNamespace(
        model=SimpleNamespace(backbone_name="mobilenetv4_conv_small")))
    calls = []

    def fake_infer(student, rgb, crf, cap):
        calls.append(crf)
        return {"qp_controls": {"states": {f"{bounds}_{state}": {} for bounds in
                  ("sym6", "asym46", "sym5") for state in ("OFF", "ON")}}}

    monkeypatch.setattr(cli, "infer_frame", fake_infer)
    assert cli.run(args) == 0
    report = json.loads((args.output / "controller_diagnostics.json").read_text())
    assert report["status"] == "completed"
    assert report["source"]["metadata_seq_length"] == 600
    assert report["source"]["unselected_frames_validated"] is False
    assert calls == [22, 32]
    assert [row["frame_index"] for row in report["records"]] == [40, 40]
    assert all(row["status"] == "completed" and len(row["qp_controls"]["states"]) == 6
               for row in report["records"])
    assert report["encoder_run"] is report["performance_claims"] is False


def test_run_refuses_existing_output(cli, tmp_path):
    args = cli.parse_args(["--sequence-dir", "s", "--checkpoint", "c", "--output", str(tmp_path)])
    with pytest.raises(FileExistsError):
        cli.run(args)
