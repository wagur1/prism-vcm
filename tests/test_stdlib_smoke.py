"""Dependency-free checks, runnable with unittest when pytest is unavailable."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from prism.codec.ffmpeg_stream import frame_bytes, memory_yuv, stream_process
from prism.data.splits import SplitEntry, SplitLedger
from prism.training.gates import require_training_ready
from prism.training.manifest import EncodeManifest, validate_budget


class StdlibSmoke(unittest.TestCase):
    def test_pipe_roundtrip_and_bad_frame(self) -> None:
        frames = [bytes([index]) * frame_bytes(4, 4) for index in range(3)]
        self.assertEqual(memory_yuv(frames, 4, 4).read(), b"".join(frames))
        output = stream_process([sys.executable, "-c",
            "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"], frames, 4, 4)
        self.assertEqual(output, b"".join(frames))
        with self.assertRaises(ValueError):
            memory_yuv([b"bad"], 4, 4)

    def test_split_freeze_tampering_and_val_seal(self) -> None:
        entries = [SplitEntry("a", "a" * 64, "train"),
                   SplitEntry("b", "b" * 64, "test", "val2017")]
        ledger = SplitLedger(entries)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            ledger.freeze(path)
            frozen = SplitLedger.read(path)
            self.assertEqual(frozen.require("a", "train"), entries[0])
            with self.assertRaises(RuntimeError):
                frozen.require("b", "test")
            data = json.loads(path.read_text())
            data["entries"][0]["sha256"] = "c" * 64
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                SplitLedger.read(path)

    def test_training_gate_rejects_skip(self) -> None:
        with self.assertRaises(RuntimeError):
            require_training_ready({"status": "SKIP"}, SplitLedger([]), {})

    def test_encode_accounting_and_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = EncodeManifest(Path(directory) / "manifest.jsonl", repo_sha="test",
                                      config_sha="test", build={}, seed=1)
            for purpose in ("oracle_initial", "oracle_refresh", "calibration_probe", "evaluation"):
                manifest.record(purpose, "a")
            self.assertEqual(sum(manifest.counts().values()), 4)
        validate_budget(gpu_hours=60, cpu_hours=100, job_hours=6,
                        working_gb=15.9, measured_intra_images=200)
        for storage in (16, float("nan")):
            with self.assertRaises((ValueError, RuntimeError)):
                validate_budget(gpu_hours=60, cpu_hours=100, job_hours=6,
                                working_gb=storage, measured_intra_images=200)


if __name__ == "__main__":
    unittest.main()
