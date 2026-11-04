#!/usr/bin/env python3
"""Evaluate bounded residual-only preprocessing with replayed calibration evidence.

No training, downloads, QP offsets, or calibration bypass. --validate-only checks
file identities and raw evidence without loading models or native libraries.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
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

from prism.codec.x265_bridge import BuildPin, EncoderConfig, X265Library, qg_shape
from prism.evaluation.residual_evidence import load_residual_evidence
from prism.preprocessing.residual import apply_bounded_residual
from prism.video.mot_eval import MOTSequence, SimpleByteTracker, compute_mot_metrics, format_mot_line
from prism.video.video_codec import VideoCodecRunner
from tier1a.common import sha256, write_json
from tier1a.oracle_codec import yuv_to_rgb
from tier1b.run_tier1b_mot17 import Student, YOLODetector, evaluation_gt

COLOR_CONTRACT = "bt709_limited_yuv420p8_luma_clip_0_255"
VARIANTS = ("anchor", "residual")


class BlockedRun(ValueError):
    """Preflight failed before creating output or loading runtime dependencies."""


class ResidualStudent(Student):
    """Reuse checkpoint loading; consume only the pixel-residual head."""

    def predict_residual(self, rgb: np.ndarray, crf: int) -> np.ndarray:
        import torch
        from torch.nn import functional as F

        height, width = rgb.shape[:2]
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None]
            x = F.interpolate(x.to(self.device, dtype=torch.float32) / 255.0,
                              size=(256, 256), mode="bilinear", align_corners=False)
            q = torch.full((1, 1), crf / 51.0, device=self.device)
            residual = self.model(x, q, for_encoding=False)["residual"]
            resized = F.interpolate(residual, size=(height, width), mode="bilinear", align_corners=False)
            return np.ascontiguousarray(resized[0, 0].cpu().numpy(), dtype=np.float32)


def source_identity(sequence: MOTSequence) -> dict:
    """Hash full source-frame bytes and geometry, independent of detector suffix.

    Including all source frames prevents a different max-frames prefix from
    acquiring a new identity. Source grouping additionally rejects suffix aliases.
    This does not detect re-encoded or renamed copies of the same source video.
    """
    root = sequence.sequence_dir.resolve()
    match = re.fullmatch(r"(MOT17-\d{2})(?:-(?:DPM|SDP|FRCNN))?", sequence.seq_name)
    group = match.group(1) if match else sequence.seq_name
    metadata = [sequence.im_width, sequence.im_height, sequence.frame_rate, sequence.seq_length]
    digest = hashlib.sha256(json.dumps(metadata, separators=(",", ":")).encode())
    for index, path in enumerate(sequence.iter_frame_paths(), 1):
        if not path.resolve().is_relative_to(root):
            raise ValueError("Sequence frame resolves outside source directory")
        digest.update(f"\n{index}:{sha256(path)}".encode("ascii"))
    return {"source_group": group, "sha256": digest.hexdigest()}


def _read_pin(path: Path) -> BuildPin:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate build-pin key: {key}")
            value[key] = item
        return value

    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if type(data) is not dict or set(data) != {f.name for f in fields(BuildPin)}:
        raise ValueError("Build pin must contain all and only BuildPin fields")
    for key in ("library_sha256", "shim_sha256"):
        if type(data[key]) is not str or not re.fullmatch(r"[0-9a-f]{64}", data[key]):
            raise ValueError(f"Invalid build hash: {key}")
    if (type(data["version"]) is not str or not data["version"]
            or type(data["enable_assembly"]) is not bool
            or any(type(data[k]) is not int or data[k] <= 0 for k in ("abi", "picture_size"))):
        raise ValueError("Invalid build pin version/ABI/picture size/assembly types")
    return BuildPin(**data)


def _evidence_paths(entries: list[str]) -> dict[int, Path]:
    paths = {}
    for entry in entries:
        value, separator, path = entry.partition("=")
        if not separator or not value.isdecimal() or not path:
            raise ValueError("evidence must use CRF=PATH entries")
        crf = int(value)
        if not 0 <= crf <= 51 or crf in paths:
            raise ValueError("evidence CRFs must be distinct and in [0,51]")
        paths[crf] = Path(path).resolve()
    return paths


def prepare(args) -> dict:
    if (not args.crfs or any(type(c) is not int or not 0 <= c <= 51 for c in args.crfs)
            or len(set(args.crfs)) != len(args.crfs)):
        raise ValueError("CRFs must be distinct integers in [0,51]")
    if (not args.variants or any(v not in VARIANTS for v in args.variants)
            or len(set(args.variants)) != len(args.variants)):
        raise ValueError("Variants must be distinct anchor/residual labels")
    if type(args.cap) is not int or not 0 <= args.cap <= 3:
        raise ValueError("cap must be an integer in [0,3]")
    if type(args.gop) is not int or args.gop < 1:
        raise ValueError("GOP must be a positive integer")
    if args.max_frames is not None and (type(args.max_frames) is not int or args.max_frames < 1):
        raise ValueError("max-frames must be a positive integer")
    if args.device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if args.checkpoint_step is not None and (type(args.checkpoint_step) is not int or args.checkpoint_step < 0):
        raise ValueError("checkpoint-step must be a nonnegative integer")

    active = args.cap > 0 and "residual" in args.variants
    evidence_paths = _evidence_paths(args.evidence)
    if active and (args.checkpoint is None or args.checkpoint_step is None
                   or set(evidence_paths) != set(args.crfs)):
        raise ValueError("Nonzero residual requires checkpoint, checkpoint-step and evidence for every CRF")
    if not active and (args.checkpoint is not None or args.checkpoint_step is not None or evidence_paths):
        raise ValueError("Checkpoint/evidence options are only valid for nonzero residual runs")
    sequence = MOTSequence(args.sequence_dir.resolve())
    qg_shape(sequence.im_width, sequence.im_height)
    count = min(sequence.seq_length, args.max_frames or sequence.seq_length)
    output = args.output.resolve()
    input_paths = [sequence.sequence_dir, args.weights.resolve(), args.library.resolve(),
                   args.shim.resolve(), args.build_pin.resolve(), *evidence_paths.values()]
    if active:
        input_paths.append(args.checkpoint.resolve())
    if output.exists():
        raise ValueError("output must be a new run directory; refusing overwrite")
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in input_paths):
        raise ValueError("output must not contain or overlap source, weights, build, or evidence paths")
    for path in input_paths[1:]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input/evidence file: {path}")
    pin = _read_pin(args.build_pin)
    for path, expected in ((args.library, pin.library_sha256), (args.shim, pin.shim_sha256)):
        if sha256(path) != expected:
            raise ValueError(f"Build hash mismatch: {path.name}")
    source = source_identity(sequence)
    checkpoint_digest = sha256(args.checkpoint) if active else None
    configs, evidence = {}, {}
    for crf in args.crfs:
        config = EncoderConfig(rc_mode="CRF", crf=crf, gop=args.gop, bframes=0, fps=sequence.frame_rate)
        config.options(output / "validation.csv")
        configs[crf] = config
        if active:
            context = {
                "checkpoint_sha256": checkpoint_digest, "seed": 0,
                "checkpoint_step": args.checkpoint_step, "encoder_config": asdict(config),
                "build_pin": asdict(pin), "width": sequence.im_width, "height": sequence.im_height,
                "fps": sequence.frame_rate, "cap": args.cap, "color_contract": COLOR_CONTRACT,
                "rate_control_condition": {"mode": "CRF", "value": crf, "cell_qp": crf,
                    "frame_type": "unknown", "interpretation": "crf_setting_not_effective_qp"},
            }
            evidence[crf] = load_residual_evidence(evidence_paths[crf], context=context,
                                                 evaluation_source=source)
    summary = {
        "schema_version": 1, "status": "validated", "source": source,
        "sequence": sequence.seq_name, "num_frames": count, "frame_interval": [1, count],
        "source_frames": sequence.seq_length, "width": sequence.im_width, "height": sequence.im_height,
        "fps": sequence.frame_rate, "variants": list(args.variants), "cap": args.cap,
        "crfs": list(args.crfs), "encoder_configs": {str(c): asdict(v) for c, v in configs.items()},
        "build_pin": asdict(pin), "runtime_build_verified": False,
        "checkpoint_sha256": checkpoint_digest, "checkpoint_step": args.checkpoint_step,
        "seed": 0 if active else None, "weights_sha256": sha256(args.weights),
        "evidence_sha256": {str(c): ev.digest for c, ev in evidence.items()},
        "color_contract": COLOR_CONTRACT, "frame_label_policy": "unknown",
        "detector": "YOLOv11-L", "tracker": "SimpleByteTracker", "metric_scale": "percent",
        "observations": [],
    }
    return dict(sequence=sequence, count=count, output=output, pin=pin, active=active,
                configs=configs, evidence=evidence, summary=summary)


class ResidualTotals:
    """Pixel-weighted streaming summaries; retain no per-frame arrays."""

    averages = ("changed_pixel_fraction", "mean_pixel_delta", "mean_abs_pixel_delta",
                "residual_clipped_fraction", "saturation_fraction")

    def __init__(self):
        self.frames = self.pixels = self.authorized = 0
        self.sums = dict.fromkeys(self.averages, 0.0)
        self.minimum = self.maximum = None
        self.max_abs = 0
        self.bypasses = {}

    def add(self, stats):
        count = stats["num_luma_pixels"]
        self.frames += 1
        self.pixels += count
        self.authorized += int(stats["applied"])
        for key in self.averages:
            self.sums[key] += stats[key] * count
        self.minimum = stats["min_pixel_delta"] if self.minimum is None else min(self.minimum, stats["min_pixel_delta"])
        self.maximum = stats["max_pixel_delta"] if self.maximum is None else max(self.maximum, stats["max_pixel_delta"])
        self.max_abs = max(self.max_abs, stats["max_abs_pixel_delta"])
        reason = stats["bypass_reason"]
        if reason is not None:
            self.bypasses[reason] = self.bypasses.get(reason, 0) + 1

    def report(self):
        return {"frames": self.frames, "num_luma_pixels": self.pixels,
                "authorized_frames": self.authorized, "bypass_counts": dict(self.bypasses),
                "min_pixel_delta": self.minimum, "max_pixel_delta": self.maximum,
                "max_abs_pixel_delta": self.max_abs,
                **{key: value / self.pixels if self.pixels else None for key, value in self.sums.items()}}


def evaluate_point(sequence, count, student, detector, codec, variant, crf, cap, evidence, output):
    width, height = sequence.im_width, sequence.im_height
    bitstream = output / "bitstreams" / f"{variant}_crf{crf}.hevc"
    tracks = output / "tracks" / f"{variant}_crf{crf}.txt"
    tracks.parent.mkdir(parents=True, exist_ok=True)
    totals = ResidualTotals()
    student_seconds = preprocessing_seconds = 0.0

    def frames():
        nonlocal student_seconds, preprocessing_seconds
        for payload in islice(sequence.iter_yuv_frames(width, height), count):
            if variant == "anchor":
                yield payload
                continue
            if cap:
                if evidence is None or not evidence.gate.allow_delta_x(evidence.step):
                    raise RuntimeError("Residual calibration revoked or expired before frame inference")
                started = time.perf_counter()
                raw = student.predict_residual(yuv_to_rgb(payload, width, height), crf)
                student_seconds += time.perf_counter() - started
            else:
                raw = np.zeros((height, width), dtype=np.float32)
            started = time.perf_counter()
            changed, stats = apply_bounded_residual(
                payload, raw, width, height, cap=cap,
                gate=evidence.gate if evidence else None, step=evidence.step if evidence else 0,
                qp=evidence.cell_qp if evidence else None,
                frame_type=evidence.frame_type if evidence else "unknown")
            preprocessing_seconds += time.perf_counter() - started
            if cap and not stats["applied"]:
                raise RuntimeError(f"Residual calibration bypassed active frame: {stats['bypass_reason']}")
            totals.add(stats)
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
    tracker = SimpleByteTracker()
    decoded = codec.decode_bitstream(bitstream, width, height)
    decoded_count = rows = 0
    started = time.perf_counter()
    try:
        with tracks.open("x", encoding="utf-8") as stream:
            for index, rgb in enumerate(decoded, 1):
                if index > count:
                    raise RuntimeError("Decoded frame count exceeds input")
                for track in tracker.update(detector.detect(rgb)):
                    stream.write(format_mot_line(index, int(track[4]), track[:4]) + "\n")
                    rows += 1
                decoded_count = index
    finally:
        if hasattr(decoded, "close"):
            decoded.close()
    if decoded_count != count:
        raise RuntimeError("Decoded frame count differs from input")
    return {
        "variant": variant, "crf": crf, "cap": cap if variant == "residual" else 0,
        "num_frames": count, "total_bytes": encoded["total_bytes"],
        "bits": encoded["total_bytes"] * 8,
        "bitrate_kbps": encoded["total_bytes"] * 8 * sequence.frame_rate / count / 1000,
        "bitstream_path": bitstream.relative_to(output).as_posix(), "bitstream_sha256": sha256(bitstream),
        "tracks_path": tracks.relative_to(output).as_posix(), "tracks_sha256": sha256(tracks),
        "track_rows": rows, "encoder_config": asdict(codec.config), "external_qp_offsets": False,
        "evidence_sha256": evidence.digest if variant == "residual" and evidence else None,
        "residual_stats": totals.report() if variant == "residual" else None,
        "student_seconds": student_seconds, "preprocessing_seconds": preprocessing_seconds,
        "encode_pipeline_seconds": encode_pipeline_seconds,
        "decode_track_seconds": time.perf_counter() - started,
    }


def run(args):
    try:
        prepared = prepare(args)
    except (OSError, ValueError) as exc:
        raise BlockedRun(str(exc)) from exc
    report = prepared["summary"]
    if args.validate_only:
        return report
    output, sequence, count = prepared["output"], prepared["sequence"], prepared["count"]
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "residual_seq_report.json"
    report["status"] = "running"
    write_json(report_path, report)
    try:
        library = X265Library(library=args.library, shim=args.shim, pin=prepared["pin"])
        if library.pin != prepared["pin"]:
            raise ValueError("Runtime build pin differs from validated evidence")
        report["runtime_build_verified"] = True
        write_json(report_path, report)
        # Recheck mutable input identities immediately before loading model weights.
        if sha256(args.weights) != report["weights_sha256"]:
            raise ValueError("Detector weights changed after validation")
        if prepared["active"] and sha256(args.checkpoint) != report["checkpoint_sha256"]:
            raise ValueError("Student checkpoint changed after validation")
        student = ResidualStudent(args.checkpoint, args.device) if prepared["active"] else None
        detector = YOLODetector(args.weights, args.device)
        gt = evaluation_gt(sequence, count, output)
        report["ground_truth_sha256"] = sha256(gt) if gt else None
        for variant in args.variants:
            for crf in args.crfs:
                attempt = {"variant": variant, "crf": crf, "status": "failed", "started": time.time()}
                try:
                    codec = VideoCodecRunner(prepared["configs"][crf], library=library)
                    row = evaluate_point(sequence, count, student, detector, codec, variant, crf,
                                         args.cap, prepared["evidence"].get(crf), output)
                    row["metrics"] = compute_mot_metrics(output / row["tracks_path"], gt)
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
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--shim", type=Path, required=True)
    parser.add_argument("--build-pin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--evidence", action="append", default=[], metavar="CRF=PATH")
    parser.add_argument("--crfs", type=int, nargs="+", default=[22, 27, 32, 37])
    parser.add_argument("--cap", type=int, choices=range(4), default=0)
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--gop", type=int, default=32)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except BlockedRun as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result if isinstance(result, dict) else
                     {"status": "complete", "report_path": str(result)}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
