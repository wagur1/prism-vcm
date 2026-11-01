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

from prism.codec.gate0 import (
    QGInvalidEvidence, frame_bits, residual_trace, verify_net_effect,
    verify_net_effect_qg, verify_spatial_effect)
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
    env.pop("PRISM_X265_RC_TELEMETRY", None)
    if name != "stock":
        env["PRISM_X265_QP_TRACE"] = str(prefix.with_suffix(".trace.csv").resolve())
    if name in ("zero", "plus6", "minus6", "checkerboard"):
        env["PRISM_X265_RC_TELEMETRY"] = str(prefix.with_suffix(".telemetry.csv").resolve())
    log_path = prefix.with_suffix(".encode.log")
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                 "--encode-job-json", str(job_path.resolve())],
                                env=env, stdout=log, stderr=subprocess.STDOUT)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if result.returncode or "x265_prism_residual_v1:" in log_text:
        tail = log_text[-6000:]
        raise RuntimeError(f"Encode {prefix.name} exited {result.returncode}:\n{tail}")


def collect_remaining_telemetry(output: Path, library: CodecBuild | X265Library,
                                width: int = 1920, height: int = 1080) -> dict:
    """Finish measurement after the frozen verifier stops; never rerun its verdict.

    Existing artifacts remain intact, including incomplete failed workers. Missing
    conditions get fresh workers; all errors are recorded in a separate manifest.
    """
    collected, existing, errors = [], [], []
    for qp in (27, 32, 37, 42, 47):
        for gop in (1, 10):
            cfg = EncoderConfig(qp=qp, crf=float(os.environ.get("GATE0_CRF", "20")),
                                rc_mode=os.environ.get("GATE0_RC", "CQP"),
                                gop=gop, bframes=0 if gop == 1 else 3)
            for name in ("zero", "plus6", "minus6", "checkerboard"):
                prefix = output / f"qp{qp}_gop{gop}_{name}"
                if any(output.glob(prefix.name + ".*")):
                    existing.append(prefix.name)
                    continue
                try:
                    encode_condition(prefix, name, library, cfg, width, height)
                    collected.append(prefix.name)
                except Exception as exc:
                    errors.append({"condition": prefix.name, "error": str(exc)})
    result = {"measurement_only": True, "collected": collected, "existing": existing, "errors": errors}
    (output / "telemetry_collection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_spike(output: Path, library: CodecBuild | X265Library, ffmpeg: str,
              reference_command: list[str], width: int = 1920, height: int = 1080,
              stock: CodecBuild | X265Library | None = None, quick: bool = False,
              qg_criteria: dict | None = None) -> dict:
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
        if not quick:
            if (qg_criteria is None or qg_criteria.get("status") != "RATIFIED"
                    or qg_criteria.get("metric") != "net-effect-qg16"):
                raise ValueError("Checkerboard requires council-RATIFIED QG16 v3 criteria; no proposal or legacy fallback")
            qg_profile = qg_criteria["bands_magnitude"]["checkerboard"]
            # Validate presence before any encoding; never substitute v1.5 bands.
            for kind in ("first_frame", "compensated"):
                for sign in ("-1", "1"):
                    if sign not in qg_profile[kind]:
                        raise ValueError(f"Missing ratified QG16 band: {kind}/{sign}")
            qg_fidelity = qg_profile["fidelity"]
            qg_floor = qg_criteria["aggregate_floor_magnitude"]
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
                # Legacy uniform-only fallback when deep telemetry is absent.
                band_profiles = {
                    "plus6": ({1: (2.5, 4.5)}, 1.0, None),
                    "minus6": ({-1: (3.5, 6.0)}, 1.0, None),
                }
                # CONFIG v1.5 (AG-2 ratified esc#10): NET-EFFECT criteria from
                # deep telemetry (CX-1 drift-cancelling metric). Falls back to the
                # frozen raw verifier only when telemetry is absent.
                band_profiles_v15 = {
                    "plus6": ({1: (4.9, 7.0)}, 1.0),
                    "minus6": ({-1: (4.9, 7.0)}, 1.0),
                }
                i_frames_v15 = set(range(10)) if gop == 1 else {0}
                for name in (("plus6",) if quick else ("plus6", "minus6", "checkerboard")):
                    # esc#11: attribution is per VERIFIED condition, never the
                    # final encode loop's stale prefix (usually checkerboard).
                    prefix = output / f"qp{qp}_gop{gop}_{name}"
                    telemetry_path = prefix.with_suffix(".telemetry.csv")
                    zero_prefix = output / f"qp{qp}_gop{gop}_zero"
                    zero_telemetry_path = zero_prefix.with_suffix(".telemetry.csv")
                    if name == "checkerboard":
                        evidence = verify_net_effect_qg(
                            zero_telemetry_path, telemetry_path, maps[name],
                            bands={kind: qg_profile[kind] for kind in ("first_frame", "compensated")},
                            min_fidelity=qg_fidelity, min_aggregate_median=qg_floor,
                            i_frames=i_frames_v15, width=width, height=height)
                    elif telemetry_path.is_file() and zero_telemetry_path.is_file():
                        bands_v15, min_fid = band_profiles_v15[name]
                        evidence = verify_net_effect(zero_telemetry_path, telemetry_path,
                                                     maps[name], bands=bands_v15,
                                                     min_fidelity=min_fid,
                                                     i_frames=i_frames_v15)
                    else:
                        bands_v14, min_fid_v14, min_sep = band_profiles[name]
                        evidence = verify_spatial_effect(traces["zero"], traces[name], maps[name],
                                                         bands=bands_v14, min_fidelity=min_fid_v14,
                                                         min_separation=min_sep)
                    report.setdefault("spatial_evidence", []).append({"qp": qp, "gop": gop,
                        "map": name, "signed_median_qp_change": evidence,
                        "bit_change_percent": 100 * (len(streams[name]) / len(streams["zero"]) - 1)})
        zero_bits_by_group = {(r["qp"], r["gop"]): r["per_frame_bits"]
                              for r in report["runs"] if r["map"] == "zero"}
        for r in report["runs"]:
            zb = zero_bits_by_group.get((r["qp"], r["gop"]))
            if zb and r["map"] != "zero" and sum(zb) > 0:
                r["bit_ratio_vs_zero_whole"] = sum(r["per_frame_bits"]) / sum(zb)
        report["synthetic_content_sha256"] = synthetic_content_sha256(width, height)
        report["synthetic_generator"] = {"seed_root": 0, "numpy": getattr(np, "__version__", "unknown"),
                                         "frames": 10, "independent_per_frame": True}
        report["status"] = "SMOKE_PASS" if quick else "PASS"
        report["gate0_complete"] = not quick
    except QGInvalidEvidence as exc:
        report["status"] = "INVALID"
        report["reason"] = str(exc)
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
    parser.add_argument("--qg-criteria", type=Path,
                        default=Path(__file__).resolve().parents[1] / "gate0/criteria_frozen_v3.json",
                        help="Council-RATIFIED QG16 bands; PROPOSAL is never accepted")
    parser.add_argument("--pin", type=Path, help="BuildPin JSON; freshly built pins still require review")
    parser.add_argument("--stock-pin", type=Path)
    parser.add_argument("--stock-shim", type=Path)
    parser.add_argument("--stock-library", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--reference-command-json", help='HEVC reference argv JSON with {bitstream}')
    parser.add_argument("--quick", action="store_true", default=os.environ.get("GATE0_QUICK") == "1")
    parser.add_argument("--encode-job-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--collect-telemetry-only", action="store_true",
                        help="Fill the full telemetry matrix without reading or writing the GATE verdict")
    args = parser.parse_args()
    if args.encode_job_json:
        encode_worker(json.loads(args.encode_job_json.read_text(encoding="utf-8")))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        shim, library_path = os.environ.get("PRISM_X265_SHIM"), os.environ.get("PRISM_X265_LIBRARY")
        library = read_build(Path(shim) if shim else None,
                             Path(library_path) if library_path else None, args.pin)
        if args.collect_telemetry_only:
            result = collect_remaining_telemetry(args.output, library)
            return int(bool(result["errors"]))
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
        qg_criteria = json.loads(args.qg_criteria.read_text(encoding="utf-8")) if args.qg_criteria.is_file() else None
        report = run_spike(args.output, library, args.ffmpeg, reference, stock=stock, quick=args.quick,
                           qg_criteria=qg_criteria)
    except X265Unavailable as exc:
        if args.collect_telemetry_only:
            raise
        report = {"status": "SKIP", "gate0_complete": False, "reason": str(exc)}
    except Exception as exc:
        if args.collect_telemetry_only:
            raise
        report = {"status": "FAIL", "gate0_complete": False, "reason": str(exc)}
    (args.output / "gate0.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in {"PASS", "SMOKE_PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
