#!/usr/bin/env python3
"""Selected-frame controller diagnostics, not encoding or a full MOT benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prism.evaluation.controller_diagnostics import (
    bounded_output_summary, paired_qp_summary, summarize_numeric,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new, nonexistent directory")
    parser.add_argument("--frame-indices", type=int, nargs="+", default=[1, 16, 33, 40, 48, 64])
    parser.add_argument("--crfs", type=int, nargs="+", default=[22, 32, 37])
    parser.add_argument("--cap", type=float, default=1.)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def validate_args(args):
    if not isinstance(args.cap, (int, float)) or not np.isfinite(args.cap) or args.cap < 0:
        raise ValueError("--cap must be finite and nonnegative")
    for name, values, lower, upper in (("frame-indices", args.frame_indices, 1, float("inf")),
                                       ("crfs", args.crfs, 0, 51)):
        if (not values or any(type(v) is not int or not lower <= v <= upper for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"--{name} requires distinct integers in [{lower}, {upper}]")
    if args.device not in ("cpu", "cuda"):
        raise ValueError("--device must be cpu or cuda")
    output = args.output.resolve()
    for source in (args.sequence_dir.resolve(), args.checkpoint.resolve()):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Output must not overlap sequence or checkpoint paths")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_frames(sequence_dir, indices):
    """Use MOTSequence's unaltered YUV conversion, visiting selected paths only."""
    from prism.video.mot_eval import MOTSequence
    from tier1a.oracle_codec import yuv_to_rgb

    class SelectedSequence(MOTSequence):
        def iter_frame_paths(self):
            for index in indices:
                if not 1 <= index <= self.seq_length:
                    raise ValueError(f"Frame {index} outside metadata sequence length")
                path = self.sequence_dir / self.im_dir / f"{index:06d}{self.im_ext}"
                if not path.resolve().is_relative_to(self.sequence_dir.resolve()):
                    raise ValueError(f"Selected frame outside source directory: {path}")
                if not path.is_file():
                    raise FileNotFoundError(f"Missing selected frame {index}: {path}")
                self.current = (index, path, sha256(path))
                yield path

    sequence = SelectedSequence(sequence_dir)
    width, height = sequence.im_width, sequence.im_height
    for payload in sequence.iter_yuv_frames(width, height):
        index, path, digest = sequence.current
        if sha256(path) != digest:
            raise ValueError(f"Selected source changed during read: {path}")
        yield index, path, digest, yuv_to_rgb(payload, width, height)


def load_student(checkpoint, device):
    from tier1b.run_tier1b_mot17 import Student

    student = Student(checkpoint, device)
    if student.model.backbone_name != "mobilenetv4_conv_small":
        raise ValueError("Diagnostics require the actual mobilenetv4_conv_small backbone")
    return student


def infer_frame(student, rgb, crf, cap):
    """One eval forward, one temporary hook; return CPU scalar summaries only."""
    import torch
    from torch.nn import functional as F
    from prism.models.student import upsample_qp_map

    captured = {}

    def capture(_module, inputs, output):
        captured["conditioned"] = inputs[0].detach().float().cpu().numpy()
        captured["raw"] = output.detach().float().cpu().numpy()

    handle = student.model.residual_head.register_forward_hook(capture)
    try:
        height, width = rgb.shape[:2]
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None]
            x = F.interpolate(x.to(student.device, dtype=torch.float32) / 255.0,
                              size=(256, 256), mode="bilinear", align_corners=False)
            q = torch.full((1, 1), crf / 51.0, device=student.device)
            output = student.model(x, q, for_encoding=False)
            residual, delta = output["residual"], output["delta_qp"]
            if tuple(residual.shape) != (1, 1, 256, 256) or tuple(delta.shape) != (1, 1, 17, 30):
                raise ValueError("Unexpected controller output shapes")
            native = F.interpolate(residual, size=(height, width), mode="bilinear", align_corners=False)
            ctu = F.interpolate(delta, size=((height + 63) // 64, (width + 63) // 64),
                                mode="bilinear", align_corners=False)
            qg = upsample_qp_map(ctu, width, height)[0, 0].float().cpu().numpy()
            raw = torch.from_numpy(captured["raw"]).to(student.device)
            rebuilt = 8 * torch.tanh(F.interpolate(raw, size=(256, 256),
                                                  mode="bilinear", align_corners=False))
            error = float((rebuilt - residual).abs().max().cpu())
            delta_array = delta[0, 0].float().cpu().numpy()
            delta_stats = summarize_numeric(delta_array)
            delta_stats.update(fraction_abs_ge_5_99=float(np.mean(np.abs(delta_array) >= 5.99)),
                               fraction_le_negative_5_99=float(np.mean(delta_array <= -5.99)),
                               fraction_ge_5_99=float(np.mean(delta_array >= 5.99)))
            result = dict(input_rgb=summarize_numeric(rgb),
                          normalized_input=summarize_numeric(x.float().cpu().numpy()),
                          conditioned_features=summarize_numeric(captured["conditioned"]),
                          residual_head_pre_resize_pre_tanh=summarize_numeric(captured["raw"]),
                          residual_head_raw_shape=list(captured["raw"].shape),
                          tanh_reconstruction_max_abs_error=error,
                          residual_256=summarize_numeric(residual[0, 0].float().cpu().numpy()),
                          residual_native=bounded_output_summary(native[0, 0].float().cpu().numpy(), cap),
                          delta_qp_17x30=delta_stats, qp_controls=paired_qp_summary(qg, width, height))
            if not np.isfinite(error) or error > 1e-5:
                raise ValueError("Hook reconstruction disagrees with actual residual output")
            return result
    finally:
        handle.remove()


def write_report(path, report):
    """Atomic replacement; invalid JSON values never replace a previous report."""
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(".json.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args):
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "controller_diagnostics.json"
    report = dict(status="running", scope="selected_frames_only_not_full_benchmark",
                  parameters={k: str(v.resolve()) if isinstance(v, Path) else v
                              for k, v in vars(args).items()},
                  seed=0, model_mode="eval_inference_mode", dtype="float32",
                  color_contract="MOTSequence_BT709_limited_YUV420p8_to_original_yuv_to_rgb",
                  residual_units="continuous_float_map_not_applied_or_rounded_pixel_changes",
                  qp_controls="2x3 OFF_zero_mean_ON_adaptive_delta_rate_offset_0",
                  encoder_run=False, performance_claims=False, records=[])
    write_report(path, report)
    try:
        from prism.video.mot_eval import MOTSequence

        sequence = MOTSequence(args.sequence_dir)
        report["source"] = dict(name=sequence.seq_name, metadata_seq_length=sequence.seq_length,
                                width=sequence.im_width, height=sequence.im_height,
                                selected_frame_indices=args.frame_indices,
                                unselected_frames_validated=False)
        if any(i > sequence.seq_length for i in args.frame_indices):
            raise ValueError("Selected frame exceeds metadata sequence length")
        report["checkpoint_sha256"] = sha256(args.checkpoint)
        student = load_student(args.checkpoint, args.device)
        if sha256(args.checkpoint) != report["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed while loading")
        import torch
        import timm

        report["versions"] = dict(torch=torch.__version__, timm=timm.__version__, numpy=np.__version__)
        report["backbone"] = student.model.backbone_name
        for index, source_path, digest, rgb in selected_frames(args.sequence_dir, args.frame_indices):
            for crf in args.crfs:
                record = dict(frame_index=index, source_path=str(source_path.resolve()),
                              source_sha256=digest, crf=crf, status="running")
                report["records"].append(record)
                try:
                    record.update(infer_frame(student, rgb, crf, args.cap))
                    record["status"] = "completed"
                except Exception as exc:
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    raise
                write_report(path, report)
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    write_report(path, report)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
