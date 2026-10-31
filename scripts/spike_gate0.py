"""Ten-frame CQP spike. SKIP and SMOKE_PASS never unlock training."""
from __future__ import annotations

import argparse
import hashlib
import csv
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class CodecBuild:
    """Paths and expected pin; do not dlopen two builds with the same SONAME."""

    shim_path: Path
    library_path: Path
    pin: BuildPin


def synthetic_frames(width: int, height: int) -> list[bytes]:
    """Council esc#4: independent seeded noise per frame so every P-frame carries
    residual CUs (the rolled single-texture design made P-frames skip-dominant)."""
    seeds = np.random.SeedSequence(0).spawn(10)
    chroma = bytes([128]) * (width * height // 2)
    return [
        np.random.Generator(np.random.PCG64(seed)).integers(
            16, 236, size=(height, width), dtype=np.uint8
        ).tobytes() + chroma
        for seed in seeds
    ]


def synthetic_content_sha256(width: int, height: int) -> str:
    digest = hashlib.sha256()
    for frame in synthetic_frames(width, height):
        digest.update(frame)
    return digest.hexdigest()


def offset_maps(width: int, height: int) -> dict[str, np.ndarray]:
    shape = qg_shape(width, height)
    yy, xx = np.indices(shape)
    return {"zero": np.zeros(shape, np.float32), "plus6": np.full(shape, 6, np.float32),
            "minus6": np.full(shape, -6, np.float32),
            "checkerboard": np.where((yy // 4 + xx // 4) % 2, 6, -6).astype(np.float32)}


def encode_worker(job: dict) -> None:
    """One library pair per fresh process; raw YUV exists only in Python memory."""
    prefix = Path(job["prefix"])
    width, height = job["width"], job["height"]
    codec = X265Library(pin=BuildPin(**job["pin"]))
    offsets = offset_maps(width, height).get(job["map"])
    cfg = EncoderConfig(**job["config"])
    with X265Encoder(codec, width, height, cfg, prefix.with_suffix(".csv")) as enc:
        stream = b"".join(enc.encode(frame, offsets) for frame in synthetic_frames(width, height))
        stream += enc.flush()
    prefix.with_suffix(".hevc").write_bytes(stream)
    bits = frame_bits(prefix.with_suffix(".csv"))
    with prefix.with_suffix(".bits.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["encode_order", "bits"])
        writer.writerows(enumerate(bits))


def encode_condition(prefix: Path, name: str, codec: CodecBuild | X265Library,
                     cfg: EncoderConfig, width: int, height: int) -> None:
    job = {"prefix": str(prefix.resolve()), "map": name, "pin": asdict(codec.pin),
           "config": asdict(cfg), "width": width, "height": height}
    job_path = prefix.with_suffix(".job.json")
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env["PRISM_X265_SHIM"] = str(codec.shim_path.resolve())
    env["PRISM_X265_LIBRARY"] = str(codec.library_path.resolve())
    # Each worker starts with its own loader search path, including the stock worker.
    env["LD_LIBRARY_PATH"] = str(codec.library_path.resolve().parent)
    env.pop("PRISM_X265_QP_TRACE", None)
    if name != "stock":
        env["PRISM_X265_QP_TRACE"] = str(prefix.with_suffix(".trace.csv").resolve())
    log_path = prefix.with_suffix(".encode.log")
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                 "--encode-job-json", str(job_path.resolve())],
                                env=env, stdout=log, stderr=subprocess.STDOUT)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if result.returncode or "x265_prism_residual_v1:" in log_text:
        tail = log_text[-6000:]
        raise RuntimeError(f"Encode {prefix.name} exited {result.returncode}:\n{tail}")


def run_spike(output: Path, library: CodecBuild | X265Library, ffmpeg: str,
              reference_command: list[str], width: int = 1920, height: int = 1080,
              stock: CodecBuild | X265Library | None = None, quick: bool = False) -> dict:
    if (not isinstance(reference_command, list) or not reference_command
            or not all(isinstance(arg, str) for arg in reference_command)
            or not any("{bitstream}" in arg for arg in reference_command)):
        raise ValueError("Reference decoder argv must contain {bitstream} and write no raw disk output")
    if stock is None and not quick:
        raise X265Unavailable("Full GATE 0 requires a separately pinned stock build")
    output.mkdir(parents=True, exist_ok=True)
    maps = offset_maps(width, height)
    # ZC-3 (esc#5): producer-side guard - a silently-drifted zero map would fake
    # a no-effect result; drift here means the harness is broken, fail loudly.
    assert maps["zero"].max() == 0.0 and maps["zero"].min() == 0.0, "zero map drifted"
    assert maps["plus6"].min() == maps["plus6"].max() == 6.0, "plus6 map drifted"
    assert sorted({float(v) for v in maps["checkerboard"].flatten()}) == [-6.0, 6.0], "checkerboard drifted"
    report: dict = {"status": "RUNNING", "instrument": "x265_prism_residual_v1",
                    "mode": "quick" if quick else "full", "gate0_complete": False,
                    "frames_per_encode": 10, "encode_attempts": 0,
                    "encodes": 0, "encoded_frames": 0,
                    "runs": [], "rc": "CQP", "pin": asdict(library.pin),
                    "stock_pin": asdict(stock.pin) if stock else None}
    try:
        for qp in ((32,) if quick else (27, 32, 37, 42, 47)):
            for gop in ((10,) if quick else (1, 10)):
                rc_mode = os.environ.get("GATE0_RC", "CQP")
                crf = float(os.environ.get("GATE0_CRF", "20"))
                cfg = EncoderConfig(qp=qp, crf=crf, rc_mode=rc_mode,
                                    gop=gop, bframes=0 if gop == 1 else 3)
                report.setdefault("rc_modes", []).append(rc_mode)
                traces, streams = {}, {}
                conditions = ("zero", "plus6") if quick else (
                    "stock", "same_binary_no_map", "zero", "plus6", "minus6", "checkerboard")
                for name in conditions:
                    prefix = output / f"qp{qp}_gop{gop}_{name}"
                    if any(prefix.parent.glob(prefix.name + ".*")):
                        raise ValueError("Use a fresh output directory; stale artifacts are not evidence")
                    codec = stock if name == "stock" else library
                    assert codec is not None
                    print(f"GATE 0: QP={qp} GOP={gop} map={name}", flush=True)
                    report["encode_attempts"] += 1
                    encode_condition(prefix, name, codec, cfg, width, height)
                    report["encodes"] += 1
                    report["encoded_frames"] += 10
                    bitstream = prefix.with_suffix(".hevc")
                    streams[name] = bitstream.read_bytes()
                    with prefix.with_suffix(".decode.log").open("w", encoding="utf-8") as log:
                        subprocess.run([ffmpeg, "-v", "error", "-xerror", "-c:v", "hevc",
                                        "-i", str(bitstream.resolve()), "-f", "null", "-"],
                                       check=True, stdout=log, stderr=subprocess.STDOUT)
                        subprocess.run([arg.replace("{bitstream}", str(bitstream.resolve()))
                                        for arg in reference_command], check=True,
                                       stdout=log, stderr=subprocess.STDOUT)
                    bits = frame_bits(prefix.with_suffix(".csv"))
                    if name in maps:
                        coverage: dict[str, dict[str, int]] = {}
                        traces[name] = residual_trace(prefix.with_suffix(".trace.csv"), width, height,
                                                      stats_out=coverage)
                        report.setdefault("frame_coverage", {})[name] = coverage
                        valid_per_frame: dict[int, int] = {}
                        for key in traces[name]:
                            valid_per_frame[key[0]] = valid_per_frame.get(key[0], 0) + 1
                        thin = {f: n for f, n in sorted(valid_per_frame.items()) if n < 60}
                        if thin:
                            raise ValueError(
                                f"Per-frame residual-block count below N=60 threshold: {thin}")
                    report["runs"].append({"qp": qp, "gop": gop, "map": name,
                        "whole_bitstream_bytes": len(streams[name]), "per_frame_bits": bits})
                if not quick:
                    if streams["stock"] != streams["zero"]:
                        raise ValueError("Zero-delta bit identity with stock failed")
                    if streams["same_binary_no_map"] != streams["zero"]:
                        raise ValueError("Zero-map changed the same-binary bitstream")
                for name in (("plus6",) if quick else ("plus6", "minus6", "checkerboard")):
                    evidence = verify_spatial_effect(traces["zero"], traces[name], maps[name])
                    report.setdefault("spatial_evidence", []).append({"qp": qp, "gop": gop,
                        "map": name, "signed_median_qp_change": evidence,
                        "bit_change_percent": 100 * (len(streams[name]) / len(streams["zero"]) - 1)})
        report["synthetic_content_sha256"] = synthetic_content_sha256(width, height)
        report["synthetic_generator"] = {"seed_root": 0, "numpy": getattr(np, "__version__", "unknown"),
                                         "frames": 10, "independent_per_frame": True}
        report["status"] = "SMOKE_PASS" if quick else "PASS"
        report["gate0_complete"] = not quick
    except Exception as exc:
        report["status"] = "FAIL"
        report["reason"] = str(exc)
    finally:
        (output / "encode_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def read_build(shim: Path | None, library: Path | None, pin: Path | None) -> CodecBuild:
    if not shim or not library or not pin or not all(p.is_file() for p in (shim, library, pin)):
        raise X265Unavailable("Pinned build requires existing library, shim, and pin files")
    return CodecBuild(shim.resolve(), library.resolve(), BuildPin(**json.loads(pin.read_text())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/gate0"))
    parser.add_argument("--pin", type=Path, help="BuildPin JSON; freshly built pins still require review")
    parser.add_argument("--stock-pin", type=Path)
    parser.add_argument("--stock-shim", type=Path)
    parser.add_argument("--stock-library", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--reference-command-json", help='HEVC reference argv JSON with {bitstream}')
    parser.add_argument("--quick", action="store_true", default=os.environ.get("GATE0_QUICK") == "1")
    parser.add_argument("--encode-job-json", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.encode_job_json:
        encode_worker(json.loads(args.encode_job_json.read_text(encoding="utf-8")))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        shim, library_path = os.environ.get("PRISM_X265_SHIM"), os.environ.get("PRISM_X265_LIBRARY")
        library = read_build(Path(shim) if shim else None,
                             Path(library_path) if library_path else None, args.pin)
        if not shutil.which(args.ffmpeg) or not args.reference_command_json:
            raise X265Unavailable("Both independent HEVC decoders are required")
        reference = json.loads(args.reference_command_json)
        if not isinstance(reference, list) or not reference or not isinstance(reference[0], str):
            raise ValueError("Reference decoder must be a nonempty argv array")
        if not shutil.which(reference[0]):
            raise X265Unavailable(f"Reference decoder unavailable: {reference[0]}")
        stock = None
        if args.stock_library or args.stock_shim or args.stock_pin:
            stock = read_build(args.stock_shim, args.stock_library, args.stock_pin)
        report = run_spike(args.output, library, args.ffmpeg, reference, stock=stock, quick=args.quick)
    except X265Unavailable as exc:
        report = {"status": "SKIP", "gate0_complete": False, "reason": str(exc)}
    except Exception as exc:
        report = {"status": "FAIL", "gate0_complete": False, "reason": str(exc)}
    (args.output / "gate0.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in {"PASS", "SMOKE_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
