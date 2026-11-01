"""GATE-0 shim workers and FFmpeg decode; raw YUV exists only in memory/pipes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from prism.codec.ffmpeg_stream import frame_bytes
from prism.codec.gate0 import frame_bits
from prism.codec.x265_bridge import BuildPin, EncoderConfig, X265Encoder, X265Library
from tier1a.common import read_json, write_json


@dataclass(frozen=True)
class ClaimConfig(EncoderConfig):
    rc_mode: str = "CRF"
    crf: float = 20.
    gop: int = 1
    bframes: int = 0

    def options(self, csv_path: Path) -> str:
        return super().options(csv_path) + "\naq-strength=1.0\nhevc-aq=0\nrc-lookahead=0\n"


def worker(job_path: Path) -> int:
    job = read_json(job_path)
    config = ClaimConfig(**job["config"])
    library = X265Library(job["shim"], job["library"], BuildPin(**job["pin"]))
    payload = sys.stdin.buffer.read(frame_bytes(job["width"], job["height"]) + 1)
    offsets = None if job["offsets"] is None else np.asarray(job["offsets"], dtype=np.float32)
    started = time.monotonic()
    with X265Encoder(library, job["width"], job["height"], config, Path(job["csv"])) as encoder:
        encoded = encoder.encode(payload, offsets) + encoder.flush()
    seconds = time.monotonic() - started
    if not encoded:
        raise ValueError("Encoder produced an empty stream")
    Path(job["bitstream"]).write_bytes(encoded)
    write_json(Path(job["result"]), {"whole_bitstream_bytes": len(encoded),
        "bits": len(encoded) * 8, "frame_bits": frame_bits(Path(job["csv"]), expected=1)[0],
        "bitstream_sha256": hashlib.sha256(encoded).hexdigest(), "encode_seconds": seconds,
        "encoder_parameters": config.options(Path(job["csv"])), "build_pin": asdict(library.pin)})
    return 0


def yuv_to_rgb(payload: bytes, width: int, height: int) -> np.ndarray:
    """BT.709 limited range, nearest 4:2:0 chroma expansion, round/clamp RGB8."""
    if len(payload) != frame_bytes(width, height):
        raise ValueError("FFmpeg must decode exactly one YUV420p8 frame")
    n = width * height
    raw = np.frombuffer(payload, dtype=np.uint8).astype(np.float32)
    y = (raw[:n].reshape(height, width) - 16) * (255 / 219)
    u, v = [(p.reshape(height // 2, width // 2).repeat(2, 0).repeat(2, 1) - 128) * (255 / 224)
            for p in (raw[n:n * 5 // 4], raw[n * 5 // 4:])]
    return np.rint(np.stack([y + 1.5748 * v, y - .187324 * u - .468124 * v,
                            y + 1.8556 * u], axis=-1)).clip(0, 255).astype(np.uint8)


class Gate0Encoder:
    def __init__(self, gate: Path, *, ffmpeg: str = "ffmpeg", deadline: float):
        self.stack = read_json(gate / "stack.json")
        self.pin = read_json(gate / "build-pin.json")
        self.stock_pin = read_json(gate / "stock-pin.json")
        self.ffmpeg, self.deadline = ffmpeg, deadline
        self.config = ClaimConfig()

    def __call__(self, payload: bytes, offsets: np.ndarray | None, frame: dict,
                 directory: Path, *, stock: bool = False) -> dict:
        directory.mkdir(parents=True, exist_ok=True)
        prefix = "stock_" if stock else ""
        job = {"width": frame["geometry"]["width"], "height": frame["geometry"]["height"],
               "library": self.stack[prefix + "library"], "shim": self.stack[prefix + "shim"],
               "pin": self.stock_pin if stock else self.pin, "config": asdict(self.config),
               "offsets": None if offsets is None else offsets.tolist(),
               "csv": str(directory / "frame.csv"), "bitstream": str(directory / "frame.hevc"),
               "result": str(directory / "encode.json")}
        write_json(directory / "job.json", job)
        env = dict(os.environ, PRISM_X265_LIBRARY=job["library"], PRISM_X265_SHIM=job["shim"])
        env["LD_LIBRARY_PATH"] = str(Path(job["library"]).parent)
        env.pop("PRISM_X265_RC_TELEMETRY", None)
        env.pop("PRISM_X265_RESIDUAL_TELEMETRY", None)
        started = time.monotonic()
        with (directory / "encode.log").open("wb") as log:
            subprocess.run([sys.executable, str(Path(__file__).with_name("run_oracle_sweep.py")),
                            "--encode-worker", str(directory / "job.json")], input=payload,
                           stdout=log, stderr=subprocess.STDOUT, env=env, check=True,
                           timeout=max(.01, min(900., self.deadline - time.monotonic())))
        result = read_json(directory / "encode.json")
        result["encode_worker_seconds"] = time.monotonic() - started
        result["bitstream_path"] = str(directory / "frame.hevc")
        return result

    def decode(self, result: dict, frame: dict) -> np.ndarray:
        geometry = frame["geometry"]
        command = [self.ffmpeg, "-v", "error", "-xerror", "-threads", "1", "-i",
                   result["bitstream_path"], "-threads", "1", "-f", "rawvideo",
                   "-pix_fmt", "yuv420p", "pipe:1"]
        decoded = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True, timeout=max(.01, min(120., self.deadline - time.monotonic())))
        return yuv_to_rgb(decoded.stdout, geometry["width"], geometry["height"])
