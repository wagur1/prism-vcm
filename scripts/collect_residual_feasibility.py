#!/usr/bin/env python3
"""Collect two research-only codec observations; never authorize a residual gate.

A real checkpoint and pinned codec are required for execution. This collector
measures one CRF/cap on one declared non-test source, not calibration or BD-rate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from itertools import islice
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prism.codec.x265_bridge import EncoderConfig, X265Library, qg_shape
from prism.preprocessing.residual import _apply_residual_pixels
from prism.video.mot_eval import MOTSequence, SimpleByteTracker, compute_mot_metrics, format_mot_line
from prism.video.video_codec import VideoCodecRunner
from tier1a.common import sha256, write_json
from tier1a.oracle_codec import yuv_to_rgb
from tier1b.run_bounded_residual_mot17 import COLOR_CONTRACT, ResidualStudent, _read_pin, source_identity
from tier1b.run_tier1b_mot17 import YOLODetector, evaluation_gt

RESEARCH_LABELS = {"research_only": True, "calibration_status": "not_established",
                   "gate_authorization": False}


class BlockedCollection(ValueError):
    """Preflight failure, before model/native loading or output creation."""


def _source_declaration(path, actual):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate source-manifest key: {key}")
            result[key] = value
        return result

    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if (type(document) is not dict or set(document) != {"schema_version", "source", "role"}
            or type(document["schema_version"]) is not int or document["schema_version"] != 1):
        raise ValueError("Source manifest must declare schema_version, source and role only")
    if document["role"] not in ("development", "calibration"):
        raise ValueError("Research source role must be development or calibration, never test")
    source = document["source"]
    if (type(source) is not dict or set(source) != {"source_group", "sha256"}
            or type(source["source_group"]) is not str or type(source["sha256"]) is not str):
        raise ValueError("Source declaration must contain source_group and sha256 strings")
    group = source["source_group"]
    match = re.fullmatch(r"(MOT17-\d{2})(?:-(?:DPM|SDP|FRCNN))?", group)
    normalized = {"source_group": match.group(1) if match else group, "sha256": source["sha256"]}
    if normalized != actual:
        raise ValueError("Source manifest group/hash does not match the complete source")
    return document["role"]


def prepare(args):
    if not args.validate_only and not args.research_only:
        raise ValueError("Execution requires --research-only; this collector cannot authorize deployment")
    for name in ("crf", "cap", "gop", "intervention_start", "intervention_end",
                 "suffix_frames", "max_frames", "max_encodes"):
        if type(getattr(args, name)) is not int:
            raise ValueError(f"{name} must be an integer")
    if not 0 <= args.crf <= 51 or not 1 <= args.cap <= 3 or args.gop < 1:
        raise ValueError("Require CRF in [0,51], cap in [1,3], and positive GOP")
    if args.intervention_start < 2 or args.intervention_end < args.intervention_start:
        raise ValueError("Require a nonempty original prefix and ordered intervention bounds")
    if args.suffix_frames < 1 or args.max_frames < 1 or args.max_encodes != 2:
        raise ValueError("Require positive suffix/frame budget and exactly two planned encodes")
    count = args.intervention_end + args.suffix_frames
    if count > args.max_frames:
        raise ValueError("Intervention plus suffix exceeds the explicit per-encode frame budget")
    if args.device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    sequence = MOTSequence(args.sequence_dir.resolve())
    qg_shape(sequence.im_width, sequence.im_height)
    if count > sequence.seq_length:
        raise ValueError("Requested intervention and suffix exceed source length")
    output = args.output.resolve()
    files = {"checkpoint": args.checkpoint.resolve(), "library": args.library.resolve(),
             "shim": args.shim.resolve(), "build_pin": args.build_pin.resolve(),
             "source_manifest": args.source_manifest.resolve()}
    if args.weights is not None:
        files["weights"] = args.weights.resolve()
    if output.exists():
        raise ValueError("output must be a new directory; refusing overwrite")
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output)
           for p in (sequence.sequence_dir.resolve(), *files.values())):
        raise ValueError("output must not overlap source/checkpoint/weights/build/manifest")
    for name, path in files.items():
        if not path.is_file():
            raise ValueError(f"Missing {name} file: {path}")
    pin = _read_pin(files["build_pin"])
    if (sha256(files["library"]) != pin.library_sha256
            or sha256(files["shim"]) != pin.shim_sha256):
        raise ValueError("Codec library/shim hash does not match build pin")
    source = source_identity(sequence)
    role = _source_declaration(files["source_manifest"], source)
    gt_source = sequence.sequence_dir / "gt" / "gt.txt"
    if gt_source.exists() and not gt_source.resolve().is_relative_to(sequence.sequence_dir.resolve()):
        raise ValueError("Ground truth resolves outside the source directory")
    config = EncoderConfig(rc_mode="CRF", crf=args.crf, gop=args.gop, bframes=0, fps=sequence.frame_rate)
    config.options(output / "validation.csv")
    hashes = {name: sha256(path) for name, path in files.items()}
    source_files = (Path(__file__), REPO / "prism/preprocessing/residual.py",
                    REPO / "tier1b/run_bounded_residual_mot17.py", REPO / "prism/video/video_codec.py",
                    REPO / "prism/codec/x265_bridge.py", REPO / "prism/models/student.py",
                    REPO / "prism/video/mot_eval.py", REPO / "tier1b/run_tier1b_mot17.py")
    code_hashes = {p.relative_to(REPO).as_posix(): sha256(p) for p in source_files}
    spec = {"source": source, "source_role": role, "encoder_config": asdict(config),
            "cap": args.cap, "frame_interval": [1, count],
            "intervention": [args.intervention_start, args.intervention_end],
            "suffix_frames": args.suffix_frames, "max_frames": args.max_frames,
            "max_encodes": args.max_encodes, "device": args.device,
            "input_hashes": hashes, "code_hashes": code_hashes, "build_pin": asdict(pin)}
    config_digest = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"),
                                            allow_nan=False).encode()).hexdigest()
    report = {"schema_version": 1, "kind": "residual_codec_feasibility", "status": "validated",
              **RESEARCH_LABELS, **spec, "config_sha256": config_digest,
              "checkpoint_sha256": hashes["checkpoint"], "seed": 0,
              "width": sequence.im_width, "height": sequence.im_height,
              "fps": sequence.frame_rate, "num_frames": count, "planned_encodes": 2,
              "planned_encoded_frames": 2 * count, "runtime_build_verified": False,
              "color_contract": COLOR_CONTRACT, "proxy_calibration": "unavailable",
              "task_metrics_requested": args.weights is not None,
              "ground_truth_available": gt_source.is_file(),
              "ground_truth_sha256": sha256(gt_source) if gt_source.is_file() else None,
              "observations": []}
    return dict(sequence=sequence, count=count, output=output, files=files, pin=pin,
                config=config, report=report)


class PixelChanges:
    """Aggregate pure pixel measurements without calibration/authorization fields."""

    averages = ("changed_pixel_fraction", "mean_pixel_delta", "mean_abs_pixel_delta",
                "residual_clipped_fraction", "saturation_fraction")

    def __init__(self):
        self.frames = self.pixels = 0
        self.sums = dict.fromkeys(self.averages, 0.0)
        self.minimum = self.maximum = None
        self.max_abs = 0

    def add(self, stats):
        self.frames += 1
        self.pixels += stats["num_luma_pixels"]
        for key in self.averages:
            self.sums[key] += stats[key] * stats["num_luma_pixels"]
        self.minimum = stats["min_pixel_delta"] if self.minimum is None else min(self.minimum, stats["min_pixel_delta"])
        self.maximum = stats["max_pixel_delta"] if self.maximum is None else max(self.maximum, stats["max_pixel_delta"])
        self.max_abs = max(self.max_abs, stats["max_abs_pixel_delta"])

    def report(self):
        return {"frames": self.frames, "num_luma_pixels": self.pixels,
                "min_pixel_delta": self.minimum, "max_pixel_delta": self.maximum,
                "max_abs_pixel_delta": self.max_abs,
                **{key: value / self.pixels if self.pixels else None for key, value in self.sums.items()}}


def evaluate_pair_arm(sequence, count, student, detector, codec, variant, args, output, gt):
    width, height = sequence.im_width, sequence.im_height
    bitstream = output / "bitstreams" / f"{variant}.hevc"
    totals, intervention = PixelChanges(), PixelChanges()
    student_seconds = preprocessing_seconds = 0.0
    no_residual = np.zeros((height, width), dtype=np.float32)

    def frames():
        nonlocal student_seconds, preprocessing_seconds
        for index, original in enumerate(islice(sequence.iter_yuv_frames(width, height), count), 1):
            active = variant == "student_residual" and args.intervention_start <= index <= args.intervention_end
            if active:
                started = time.perf_counter()
                raw = student.predict_residual(yuv_to_rgb(original, width, height), args.crf)
                student_seconds += time.perf_counter() - started
            else:
                raw = no_residual
            started = time.perf_counter()
            changed, stats = _apply_residual_pixels(original, raw, width, height, cap=args.cap if active else 0)
            preprocessing_seconds += time.perf_counter() - started
            totals.add(stats)
            if args.intervention_start <= index <= args.intervention_end:
                intervention.add(stats)
            yield changed

    source = frames()
    started = time.perf_counter()
    try:
        encoded = codec.encode_sequence(source, None, bitstream, width, height, fps=sequence.frame_rate)
    finally:
        source.close()
    encode_pipeline_seconds = time.perf_counter() - started
    if encoded["num_frames"] != count:
        raise RuntimeError("Encoded frame count mismatch")
    if encoded["total_bytes"] <= 0 or bitstream.stat().st_size != encoded["total_bytes"]:
        raise RuntimeError("Bitstream byte accounting mismatch")

    from PIL import Image

    regions = {key: {"frames": 0, "samples": 0, "squared_error": 0.0}
               for key in ("all", "prefix", "intervention", "suffix")}
    tracker = SimpleByteTracker() if detector is not None else None
    tracks = output / "tracks" / f"{variant}.txt"
    if tracker is not None:
        tracks.parent.mkdir(parents=True, exist_ok=True)
    stream = tracks.open("x", encoding="utf-8") if tracker is not None else None
    decoded, originals = codec.decode_bitstream(bitstream, width, height), sequence.iter_frame_paths()
    decoded_count = track_rows = 0
    started = time.perf_counter()
    try:
        for index, rgb in enumerate(decoded, 1):
            if index > count:
                raise RuntimeError("Decoded frame count exceeds encoded frames")
            if rgb.dtype != np.uint8 or rgb.shape != (height, width, 3):
                raise RuntimeError("Decoded RGB frame has unexpected dtype/geometry")
            path = next(originals)
            with Image.open(path) as image:
                if image.size != (width, height):
                    raise ValueError("Source image geometry differs from sequence metadata")
                reference = np.asarray(image.convert("RGB"), dtype=np.uint8)
            difference = rgb.astype(np.float64) - reference.astype(np.float64)
            squared_error = float(np.square(difference).sum())
            phase = ("prefix" if index < args.intervention_start else
                     "intervention" if index <= args.intervention_end else "suffix")
            for region in (regions["all"], regions[phase]):
                region["frames"] += 1
                region["samples"] += difference.size
                region["squared_error"] += squared_error
            if tracker is not None:
                for track in tracker.update(detector.detect(rgb)):
                    stream.write(format_mot_line(index, int(track[4]), track[:4]) + "\n")
                    track_rows += 1
            decoded_count = index
    finally:
        if hasattr(decoded, "close"):
            decoded.close()
        originals.close()
        if stream is not None:
            stream.close()
    if decoded_count != count:
        raise RuntimeError("Decoded frame count differs from encoded frames")
    distortion = {key: {"frames": value["frames"], "samples": value["samples"],
                         "mse": value["squared_error"] / value["samples"] if value["samples"] else None}
                  for key, value in regions.items()}
    if tracker is not None:
        metrics = compute_mot_metrics(tracks, gt)
    else:
        metrics = {"evaluated": False, "HOTA": None, "MOTA": None, "IDF1": None,
                   "reason": "detector_not_requested" if args.weights is None else "ground_truth_missing"}
    return {"variant": variant, **RESEARCH_LABELS, "crf": args.crf,
            "cap": args.cap if variant == "student_residual" else 0,
            "num_frames": count, "total_bytes": encoded["total_bytes"],
            "bits": encoded["total_bytes"] * 8,
            "bitrate_kbps": encoded["total_bytes"] * 8 * sequence.frame_rate / count / 1000,
            "bitstream_path": bitstream.relative_to(output).as_posix(), "bitstream_sha256": sha256(bitstream),
            "external_qp_offsets": False, "pixel_changes": totals.report(),
            "intervention_pixel_changes": intervention.report(),
            "distortion_rgb": distortion,
            "distortion_reference": "original RGB; includes source color conversion and codec distortion",
            "metrics": metrics, "tracks_path": tracks.relative_to(output).as_posix() if tracker else None,
            "track_rows": track_rows, "student_seconds": student_seconds,
            "preprocessing_seconds": preprocessing_seconds, "encode_pipeline_seconds": encode_pipeline_seconds,
            "decode_measure_seconds": time.perf_counter() - started}


def run(args):
    try:
        prepared = prepare(args)
    except (OSError, ValueError) as exc:
        raise BlockedCollection(str(exc)) from exc
    report = prepared["report"]
    if args.validate_only:
        return report
    output, sequence = prepared["output"], prepared["sequence"]
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "feasibility_report.json"
    report["status"] = "running"
    write_json(report_path, report)
    try:
        for name, path in prepared["files"].items():
            if sha256(path) != report["input_hashes"][name]:
                raise ValueError(f"Input changed after preflight: {name}")
        library = X265Library(library=args.library, shim=args.shim, pin=prepared["pin"])
        if library.pin != prepared["pin"]:
            raise ValueError("Runtime build pin mismatch")
        report["runtime_build_verified"] = True
        write_json(report_path, report)
        student = ResidualStudent(args.checkpoint, args.device)
        if getattr(student.model, "backbone_name", None) != "mobilenetv4_conv_small":
            raise ValueError("Research collector requires the trained MobileNetV4 backbone, not a smoke fallback")
        report["backbone"] = student.model.backbone_name
        gt = evaluation_gt(sequence, prepared["count"], output)
        detector = YOLODetector(args.weights, args.device) if args.weights is not None and gt else None
        report["detector"] = "YOLOv11-L" if detector else None
        report["tracker"] = "SimpleByteTracker" if detector else None
        for variant in ("anchor", "student_residual"):
            attempt = {"variant": variant, **RESEARCH_LABELS, "status": "failed", "started": time.time()}
            try:
                if source_identity(sequence) != report["source"]:
                    raise ValueError("Source changed after validation or between arms")
                codec = VideoCodecRunner(prepared["config"], library=library)
                row = evaluate_pair_arm(sequence, prepared["count"], student, detector, codec,
                                        variant, args, output, gt)
                report["observations"].append(row)
                attempt["status"] = "complete"
                write_json(report_path, report)
            except Exception as exc:
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                attempt["finished"] = time.time()
                with (output / "attempts.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(attempt, allow_nan=False) + "\n")
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(report_path, report)
    return report_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sequence-dir", "checkpoint", "library", "shim", "build-pin", "source-manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--weights", type=Path, help="Optional detector weights; GT is also required for task metrics")
    parser.add_argument("--crf", type=int, required=True)
    parser.add_argument("--cap", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--gop", type=int, default=32)
    parser.add_argument("--intervention-start", type=int, required=True)
    parser.add_argument("--intervention-end", type=int, required=True)
    parser.add_argument("--suffix-frames", type=int, required=True)
    parser.add_argument("--max-frames", type=int, required=True, help="Maximum frames per encode, including original prefix")
    parser.add_argument("--max-encodes", type=int, required=True, help="Must be 2; no automatic retries or sweep")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--research-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except BlockedCollection as exc:
        print(json.dumps({"status": "blocked", **RESEARCH_LABELS, "error": str(exc)}, ensure_ascii=False))
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", **RESEARCH_LABELS, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result if isinstance(result, dict) else
                     {"status": "complete", **RESEARCH_LABELS, "report_path": str(result)},
                     ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
