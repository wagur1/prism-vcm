"""Offline orchestration tests; codecs/numerical evidence are explicitly mocked."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gate0 import run_gate0_kaggle as runner

ROOT = Path(__file__).resolve().parents[1]


def load_controller():
    # Only test control flow here. Restoring sys.modules prevents this minimal
    # stand-in from affecting real NumPy tests in the same test suite.
    with patch.dict(sys.modules, {"numpy": SimpleNamespace(isfinite=math.isfinite)}):
        spec = importlib.util.spec_from_file_location("_gate0_controller_test", ROOT / "scripts/spike_gate0.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


class Gate0ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.spike = load_controller()
        self.pin = self.spike.BuildPin("a" * 64, "b" * 64, "test-only", 199, 123)
        self.codec = self.spike.CodecBuild(self.output / "shim.so", self.output / "libx265.so", self.pin)

    def exercise(self, *, quick=True, encoder_error=False, decoder_error=False, identity_error=False):
        # Council esc#4: run_spike now requires >=60 valid residual CUs per frame
        # before the spatial-effect verification, so the stub trace must be dense.
        def dense_trace():
            return {(f, (i % 30) * 16, (i // 30) * 16): 32.0
                    for f in range(10) for i in range(60)}
        # numpy is stubbed in the loaded controller; content hashing is numeric work.
        self.np_stub_fixed = patch.object(self.spike, "synthetic_content_sha256",
                                          return_value="0" * 64)
        self.np_stub_fixed.start()
        self.addCleanup(self.np_stub_fixed.stop)

        def encode(prefix, name, *unused):
            if encoder_error:
                raise RuntimeError("test encode error")
            prefix.with_suffix(".hevc").write_bytes(b"different" if identity_error and name == "stock" else b"test")
            prefix.with_suffix(".csv").write_text("Bits\n" + "8\n" * 10)
        with patch.object(self.spike, "offset_maps", return_value={name: None for name in (
                "zero", "plus6", "minus6", "checkerboard")}), \
                patch.object(self.spike, "encode_condition", side_effect=encode) as encodes, \
                patch.object(self.spike, "residual_trace", side_effect=lambda *a, **k: dense_trace()), \
                patch.object(self.spike, "verify_spatial_effect", return_value={"1": 6}), \
                patch.object(self.spike.subprocess, "run", side_effect=(
                    subprocess.CalledProcessError(9, ["decoder"]) if decoder_error else None)) as decodes, \
                redirect_stdout(io.StringIO()):
            report = self.spike.run_spike(self.output, self.codec, "ffmpeg", ["HM", "-b", "{bitstream}"],
                                         stock=self.codec, quick=quick)
        return report, encodes, decodes

    def test_quick_is_exactly_two_sequences_and_cannot_pass_gate(self):
        report, encodes, decodes = self.exercise()
        self.assertEqual(report["status"], "SMOKE_PASS")
        self.assertFalse(report["gate0_complete"])
        self.assertEqual((report["encodes"], report["encoded_frames"]), (2, 20))
        self.assertEqual([call.args[1] for call in encodes.call_args_list], ["zero", "plus6"])
        self.assertEqual(decodes.call_count, 4)
        self.assertFalse(list(self.output.glob("*.yuv")))

    def test_full_matrix_still_has_all_sixty_encodes(self):
        report, encodes, decodes = self.exercise(quick=False)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual((report["encodes"], report["encoded_frames"]), (60, 600))
        self.assertEqual(encodes.call_count, 60)
        self.assertEqual(decodes.call_count, 120)

    def test_failures_never_become_smoke_success(self):
        report, _, _ = self.exercise(encoder_error=True)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("test encode error", report["reason"])

    def test_decoder_error_fails_gate(self):
        report, _, _ = self.exercise(decoder_error=True)
        self.assertEqual(report["status"], "FAIL")

    def test_stock_identity_is_required(self):
        report, _, _ = self.exercise(quick=False, identity_error=True)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("identity", report["reason"])

    def test_full_cannot_substitute_patched_for_missing_stock(self):
        with self.assertRaises(self.spike.X265Unavailable):
            self.spike.run_spike(self.output, self.codec, "ffmpeg", ["HM", "-b", "{bitstream}"])

    def test_each_worker_gets_its_own_library_and_trace_environment(self):
        for name in ("stock", "zero"):
            with patch.object(self.spike.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
                self.spike.encode_condition(self.output / name, name, self.codec,
                                            self.spike.EncoderConfig(), 64, 64)
            env = run.call_args.kwargs["env"]
            self.assertEqual(env["PRISM_X265_LIBRARY"], str(self.codec.library_path.resolve()))
            self.assertEqual(env["PRISM_X265_SHIM"], str(self.codec.shim_path.resolve()))
            self.assertEqual(env["LD_LIBRARY_PATH"], str(self.codec.library_path.resolve().parent))
            self.assertEqual("PRISM_X265_QP_TRACE" in env, name != "stock")
            self.assertIn("--encode-job-json", run.call_args.args[0])

    def test_duplicate_trace_rows_are_rejected(self):
        trace = self.output / "duplicate.csv"
        trace.write_text("frame,x,y,qp,has_residual,skip\n0,0,0,32,1,0\n0,0,0,33,1,0\n")
        with self.assertRaisesRegex(ValueError, "Duplicate CU"):
            self.spike.residual_trace(trace, 64, 64)


class Gate0RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.gate = root / "repo/gate0"
        self.gate.mkdir(parents=True)
        self.output = root / "output"
        for name in ("build-pin.json", "stock-pin.json", "x265_residual_patch.patch", "PATCH_AUDIT.md"):
            (self.gate / name).write_text("{}")
        stack = {key: str(self.gate / key) for key in (
            "shim", "library", "stock_shim", "stock_library", "reference_decoder")}
        (self.gate / "stack.json").write_text(json.dumps(stack))

    def invoke(self, fake_run, quick=False):
        argv = ["runner", "--output", str(self.output)] + (["--quick"] if quick else [])
        with patch.object(runner, "GATE", self.gate), patch.object(runner, "REPO", self.gate.parent), \
                patch.object(runner, "run_logged", side_effect=fake_run) as calls, \
                patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            code = runner.main()
        return code, json.loads((self.output / "gate0.json").read_text()), calls

    def test_build_failure_reports_exact_step_and_preserves_exit(self):
        def failure(command, env, log):
            log.write_text("[GATE0 STEP] build_HM_TLibDecoder\ncompiler diagnostic\n"
                           "[GATE0 FAIL] step=build_HM_TLibDecoder exit=7 line=100\n")
            return 7
        code, report, calls = self.invoke(failure)
        self.assertEqual(code, 7)
        self.assertEqual(report["status"], "SKIP")
        self.assertEqual(report["failing_step"], "build_HM_TLibDecoder")
        self.assertIn("compiler diagnostic", report["log_tail"])
        self.assertEqual(calls.call_count, 1)
        self.assertFalse((self.output / "build-pin.json").exists())

    def test_missing_bash_is_skip(self):
        code, report, _ = self.invoke(FileNotFoundError("bash missing"))
        self.assertNotEqual(code, 0)
        self.assertEqual(report["status"], "SKIP")
        self.assertIn("bash missing", report["reason"])

    def test_quick_rejects_a_full_pass_report(self):
        def fake(command, env, log):
            log.write_text("test log\n")
            if "--output" in command:
                evidence = Path(command[command.index("--output") + 1])
                evidence.mkdir()
                (evidence / "gate0.json").write_text('{"status": "PASS"}')
            return 0
        code, report, calls = self.invoke(fake, quick=True)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("--quick", calls.call_args.args[0])

    def test_spike_exit_code_propagates_without_report(self):
        def fake(command, env, log):
            log.write_text("test log\n")
            return 42 if "--output" in command else 0
        code, report, _ = self.invoke(fake)
        self.assertEqual(code, 42)
        self.assertEqual(report["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
