"""Offline ten-frame CQP spike. SKIP is not PASS and never unlocks training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from prism.codec.gate0 import frame_bits, residual_trace, verify_spatial_effect
from prism.codec.x265_bridge import (BuildPin, EncoderConfig, X265Encoder,
                                    X265Library, X265Unavailable, qg_shape)


def synthetic_frames(width: int, height: int) -> list[bytes]:
    rng = np.random.default_rng(0)
    texture = rng.integers(16, 236, (height, width), dtype=np.uint8)
    chroma = bytes([128]) * (width * height // 2)
    return [np.roll(texture, shift=index * 2, axis=1).tobytes() + chroma for index in range(10)]


def run_spike(output: Path, library: X265Library, ffmpeg: str,
              reference_command: list[str], width: int = 1920, height: int = 1080,
              stock: X265Library | None = None) -> dict:
    if not reference_command or not any("{bitstream}" in arg for arg in reference_command):
        raise ValueError("Reference decoder command must contain {bitstream} and write no raw disk output")
    output.mkdir(parents=True, exist_ok=True)
    frames = synthetic_frames(width, height)
    shape = qg_shape(width, height)
    yy, xx = np.indices(shape)
    maps = {"zero": np.zeros(shape, np.float32), "plus6": np.full(shape, 6, np.float32),
            "minus6": np.full(shape, -6, np.float32),
            "checkerboard": np.where((yy // 4 + xx // 4) % 2, 6, -6).astype(np.float32)}
    report: dict = {"status": "RUNNING", "instrument": "x265_prism_residual_v1",
                    "frames_per_encode": 10, "encodes": 0, "runs": [], "rc": "CQP"}
    previous_trace = os.environ.get("PRISM_X265_QP_TRACE")
    try:
        for qp in (27, 32, 37, 42, 47):
            for gop in (1, 10):
                cfg = EncoderConfig(qp=qp, gop=gop, bframes=0 if gop == 1 else 3)
                traces, streams = {}, {}
                for name, offsets in {"stock": None, "same_binary_no_map": None, **maps}.items():
                    prefix = output / f"qp{qp}_gop{gop}_{name}"
                    trace_path = prefix.with_suffix(".trace.csv")
                    if trace_path.exists():
                        raise ValueError("Use a fresh output directory; stale traces are not evidence")
                    os.environ["PRISM_X265_QP_TRACE"] = str(trace_path.resolve())
                    codec = (stock or library) if name == "stock" else library
                    report["encodes"] += 10
                    with X265Encoder(codec, width, height, cfg, prefix.with_suffix(".csv")) as enc:
                        stream = b"".join(enc.encode(frame, offsets) for frame in frames) + enc.flush()
                    streams[name] = stream
                    bitstream = prefix.with_suffix(".hevc")
                    bitstream.write_bytes(stream)  # Only compressed bitstreams may touch disk.
                    subprocess.run([ffmpeg, "-v", "error", "-xerror", "-i", str(bitstream),
                                    "-f", "null", "-"], check=True, stdout=subprocess.DEVNULL)
                    subprocess.run([arg.replace("{bitstream}", str(bitstream.resolve()))
                                    for arg in reference_command], check=True,
                                   stdout=subprocess.DEVNULL)
                    bits = frame_bits(prefix.with_suffix(".csv"))
                    if name in maps:
                        traces[name] = residual_trace(trace_path, width, height)
                    report["runs"].append({"qp": qp, "gop": gop, "map": name,
                        "whole_bitstream_bytes": len(stream), "per_frame_bits": bits})
                if streams["stock"] != streams["zero"]:
                    raise ValueError("Zero-delta bit identity with stock failed")
                if streams["same_binary_no_map"] != streams["zero"]:
                    raise ValueError("Zero-map changed the same-binary bitstream")
                for name in ("plus6", "minus6", "checkerboard"):
                    evidence = verify_spatial_effect(traces["zero"], traces[name], maps[name])
                    report.setdefault("spatial_evidence", []).append({"qp": qp, "gop": gop,
                        "map": name, "signed_median_qp_change": evidence,
                        "bit_change_percent": 100 * (len(streams[name]) / len(streams["zero"]) - 1)})
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["reason"] = str(exc)
        raise
    finally:
        if previous_trace is None:
            os.environ.pop("PRISM_X265_QP_TRACE", None)
        else:
            os.environ["PRISM_X265_QP_TRACE"] = previous_trace
        (output / "encode_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["pin"] = library.pin.__dict__
    report["stock_pin"] = (stock or library).pin.__dict__
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/gate0"))
    parser.add_argument("--pin", type=Path, help="Previously reviewed BuildPin JSON")
    parser.add_argument("--stock-pin", type=Path)
    parser.add_argument("--stock-shim", type=Path)
    parser.add_argument("--stock-library", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--reference-command-json", help='HEVC reference argv JSON with {bitstream}')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        library = X265Library(pin=BuildPin(**json.loads(args.pin.read_text())) if args.pin else None)
        if args.pin is None or not shutil.which(args.ffmpeg) or not args.reference_command_json:
            raise X265Unavailable("Pinned build and both independent HEVC decoders are required")
        stock = None
        if args.stock_library or args.stock_shim:
            if not (args.stock_library and args.stock_shim and args.stock_pin):
                raise ValueError("Separate stock builds require library, shim, and pin")
            stock = X265Library(args.stock_shim, args.stock_library,
                                BuildPin(**json.loads(args.stock_pin.read_text())))
        report = run_spike(args.output, library, args.ffmpeg,
                           json.loads(args.reference_command_json), stock=stock)
    except X265Unavailable as exc:
        report = {"status": "SKIP", "reason": str(exc)}
    except Exception as exc:
        report = {"status": "FAIL", "reason": str(exc)}
    (args.output / "gate0.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
