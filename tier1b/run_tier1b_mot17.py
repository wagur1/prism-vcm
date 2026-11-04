#!/usr/bin/env python3
"""Evaluate one MOT17 sequence with Plan B spatial-QP ablations.

Requires torch/timm, Pillow, ultralytics, FFmpeg, and a built x265 shim.
Set PRISM_X265_LIBRARY and PRISM_X265_SHIM to the GATE-0 build artifacts.
Frames retain their native (even) dimensions; metrics use MOTChallenge percent.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import islice
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prism.codec.ablation import generate_ablation_offsets
from prism.codec.qp_maps import project_zero_mean_qp_map, qg_area_weights
from prism.codec.x265_bridge import EncoderConfig, qg_shape
from prism.preprocessing.background_filter import apply_background_smoothing
from prism.video.mot_eval import (
    MOTSequence, SimpleByteTracker, compute_mot_metrics, format_mot_line,
)
from prism.video.video_codec import VideoCodecRunner
from tier1a.common import exclusive, sha256, write_json
from tier1a.oracle_codec import yuv_to_rgb

CRFS = (22, 27, 32, 37)
VARIANTS = (
    "anchor", "M2_zero_mean", "M2_asym", "M2_asym_ratecomp",
    "M2_filter_only", "M2_asym_prefilter",
)


class Student:
    """Seed-0 controller adapter; keep optional GPU dependencies lazy."""

    def __init__(self, checkpoint_path: Path, device: str):
        import torch
        from prism.models.student import MobileNetV4HybridController

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu for CPU runs")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("seed") != 0:
            raise ValueError("Expected a seed-0 student checkpoint")
        backbone = {"mobilenetv4_conv_small": "timm", "small_conv_smoke_only": "small"}.get(
            checkpoint.get("backbone_name"))
        if backbone is None:
            raise ValueError("Unknown student checkpoint backbone")
        self.model = MobileNetV4HybridController(backbone=backbone, allow_fallback=False)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(device).eval()
        self.device = device

    def spatial_qp(self, rgb: np.ndarray, crf: int) -> np.ndarray:
        import torch
        from torch.nn import functional as F
        from prism.models.student import upsample_qp_map

        height, width = rgb.shape[:2]
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None]
            x = F.interpolate(x.to(self.device, dtype=torch.float32) / 255.0,
                              size=(256, 256), mode="bilinear", align_corners=False)
            qp = torch.full((1, 1), crf / 51.0, device=self.device)
            delta = self.model(x, qp, for_encoding=False)["delta_qp"]
            # The trained head is 17x30. Resample CTU centers for native MOT geometry
            # before the existing CTU-to-QG interpolation (no integer rounding).
            delta = F.interpolate(delta, size=((height + 63) // 64, (width + 63) // 64),
                                  mode="bilinear", align_corners=False)
            return np.ascontiguousarray(
                upsample_qp_map(delta, width, height)[0, 0].cpu().numpy(), dtype=np.float32)


class YOLODetector:
    def __init__(self, weights: Path, device: str):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device

    def detect(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image

        # PIL explicitly denotes RGB; Ultralytics treats ndarray input as BGR.
        result = self.model.predict(Image.fromarray(rgb), device=self.device,
                                    classes=[0], conf=0.1, iou=0.7, imgsz=640,
                                    max_det=300, verbose=False)[0]
        if result.boxes is None:
            return np.empty((0, 5), dtype=float)
        boxes = result.boxes
        people = boxes.cls.cpu().numpy() == 0
        return np.column_stack((boxes.xyxy.cpu().numpy()[people],
                                boxes.conf.cpu().numpy()[people]))


def smooth_yuv(payload: bytes, spatial_qp: np.ndarray, width: int, height: int) -> bytes:
    """Use the same centered background mask in both prefilter ablations.

    Positive centered asymmetric QPs indicate background. The rate-compensation
    DC shift must not turn foreground cells into background smoothing targets.
    """
    mask = project_zero_mean_qp_map(
        spatial_qp, weights=qg_area_weights(width, height), low=-4, high=6,
        quantize_int8=False)
    raw, n = np.frombuffer(payload, np.uint8), width * height
    planes = (raw[:n].reshape(height, width),
              raw[n:n * 5 // 4].reshape(height // 2, width // 2),
              raw[n * 5 // 4:].reshape(height // 2, width // 2))
    return b"".join(p.tobytes() for p in apply_background_smoothing(planes, mask))


def validate_options(args) -> None:
    if (not args.crfs or len(set(args.crfs)) != len(args.crfs)
            or any(type(c) is not int or not 0 <= c <= 51 for c in args.crfs)):
        raise ValueError("CRFs must be distinct integers in [0, 51]")
    if (not args.variants or len(set(args.variants)) != len(args.variants)
            or any(v not in VARIANTS for v in args.variants)):
        raise ValueError("Variants must be distinct supported ablations")
    if type(args.gop) is not int or args.gop < 1 or args.bframes not in (0, 3):
        raise ValueError("Require GOP >= 1 and bframes 0 (LDP) or 3 (RA)")
    if args.max_frames is not None and (type(args.max_frames) is not int or args.max_frames < 1):
        raise ValueError("max_frames must be a positive integer")
    if args.device not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu")


def evaluation_gt(sequence: MOTSequence, count: int, output: Path) -> Path | None:
    source = sequence.sequence_dir / "gt" / "gt.txt"
    if not source.is_file():
        return None
    if count == sequence.seq_length:
        return source
    target = output / "evaluated_gt.txt"
    with source.open(encoding="utf-8-sig") as src, target.open("w", encoding="utf-8") as dst:
        for line in src:
            if line.strip() and 1 <= float(line.split(",", 1)[0]) <= count:
                dst.write(line.rstrip() + "\n")
    return target


def evaluate_point(sequence, count, student, detector, codec, variant, crf, output):
    width, height = sequence.im_width, sequence.im_height
    bitstream = output / "bitstreams" / f"{variant}_crf{crf}.hevc"
    tracks_path = output / "tracks" / f"{variant}_crf{crf}.txt"
    tracks_path.parent.mkdir(parents=True, exist_ok=True)
    current_offsets = None
    current_index = -1

    def frames():
        nonlocal current_offsets, current_index
        for index, payload in enumerate(islice(sequence.iter_yuv_frames(width, height), count)):
            spatial = student.spatial_qp(yuv_to_rgb(payload, width, height), crf)
            current_offsets, prefilter = generate_ablation_offsets(
                spatial, variant, crf, width=width, height=height)
            current_index = index
            yield smooth_yuv(payload, spatial, width, height) if prefilter else payload

    def offsets(index, w, h):
        if index != current_index or (w, h) != (width, height):
            raise RuntimeError("Encoder frame/QP map synchronization failed")
        return current_offsets

    started = time.monotonic()
    source = frames()
    try:
        encoded = codec.encode_sequence(source, offsets, bitstream, width, height,
                                        fps=sequence.frame_rate)
    finally:
        source.close()
    encode_seconds = time.monotonic() - started
    if encoded["num_frames"] != count:
        raise RuntimeError("Encoded frame count does not match requested interval")
    if bitstream.stat().st_size != encoded["total_bytes"] or encoded["total_bytes"] <= 0:
        raise RuntimeError("Bitstream size does not match encoder accounting")
    tracker = SimpleByteTracker()
    decoded = codec.decode_bitstream(bitstream, width, height)
    decoded_count = track_rows = 0
    started = time.monotonic()
    try:
        with tracks_path.open("w", encoding="utf-8") as stream:
            for frame_index, rgb in enumerate(decoded, 1):
                if frame_index > count:
                    raise RuntimeError("Decoded frame count exceeds encoded sequence")
                for track in tracker.update(detector.detect(rgb)):
                    stream.write(format_mot_line(frame_index, int(track[4]), track[:4]) + "\n")
                    track_rows += 1
                decoded_count = frame_index
    finally:
        if hasattr(decoded, "close"):
            decoded.close()
    if decoded_count != count:
        raise RuntimeError(f"Decoded frame count {decoded_count} differs from encoded {count}")
    return {
        "variant": variant, "crf": crf, "num_frames": count,
        "total_bytes": encoded["total_bytes"], "bits": encoded["total_bytes"] * 8,
        "bitrate_kbps": encoded["total_bytes"] * 8 * sequence.frame_rate / count / 1000,
        "bitstream_path": bitstream.relative_to(output).as_posix(),
        "bitstream_sha256": sha256(bitstream),
        "tracks_path": tracks_path.relative_to(output).as_posix(), "track_rows": track_rows,
        "encode_seconds": encode_seconds, "decode_track_seconds": time.monotonic() - started,
        "encoder_config": asdict(codec.config),
    }


def run(args) -> Path:
    validate_options(args)
    sequence = MOTSequence(args.sequence_dir)
    qg_shape(sequence.im_width, sequence.im_height)  # Reject unsupported odd geometry early.
    count = min(sequence.seq_length, args.max_frames or sequence.seq_length)
    checkpoint = Path(args.checkpoints_dir) / "student_seed_0.pt"
    weights = Path(args.weights)
    for path in (checkpoint, weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "mot17_seq_report.json"
    with exclusive(output / ".mot17.lock"):
        gt = evaluation_gt(sequence, count, output)
        report = {
            "schema_version": 1, "status": "running", "sequence": sequence.seq_name,
            "width": sequence.im_width, "height": sequence.im_height,
            "fps": sequence.frame_rate, "sequence_frames": sequence.seq_length,
            "num_frames": count, "duration_seconds": count / sequence.frame_rate,
            "frame_interval": [1, count], "crfs": list(args.crfs),
            "variants": list(args.variants), "gop": args.gop, "bframes": args.bframes,
            "prediction_structure": "LDP" if args.bframes == 0 else "RA",
            "device": args.device, "seed": 0, "checkpoint_sha256": sha256(checkpoint),
            "weights_sha256": sha256(weights), "detector": "YOLOv11-L",
            "detector_options": {"classes": [0], "conf": 0.1, "iou": 0.7, "imgsz": 640, "max_det": 300},
            "tracker": "SimpleByteTracker", "metric_scale": "percent",
            "ground_truth_available": gt is not None,
            "ground_truth_sha256": sha256(gt) if gt else None,
            "prefilter": "positive centered asymmetric QP; strength=1, sigma=1; identical in both variants",
            "observations": [],
        }
        write_json(report_path, report)
        try:
            student = Student(checkpoint, args.device)
            detector = YOLODetector(weights, args.device)
            for variant in args.variants:
                for crf in args.crfs:
                    print(f"[{sequence.seq_name}] {variant} CRF={crf} frames={count}", flush=True)
                    codec = VideoCodecRunner(EncoderConfig(
                        rc_mode="CRF", crf=crf, gop=args.gop, bframes=args.bframes,
                        fps=sequence.frame_rate))
                    row = evaluate_point(sequence, count, student, detector, codec,
                                         variant, crf, output)
                    row["metrics"] = compute_mot_metrics(output / row["tracks_path"], gt)
                    report["observations"].append(row)
                    write_json(report_path, report)
            report["status"] = "complete"
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_json(report_path, report)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crfs", type=int, nargs="+", default=list(CRFS))
    parser.add_argument("--gop", type=int, default=32)
    parser.add_argument("--bframes", type=int, choices=(0, 3), default=0)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv=None) -> int:
    print(run(build_parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
