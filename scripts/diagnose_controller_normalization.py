#!/usr/bin/env python3
"""Optional offline NOUPDATE normalization/conditioning probe, not a deployment recipe."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
from types import MethodType

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_controller_outputs import selected_frames, load_student, write_report, sha256
from prism.evaluation.controller_diagnostics import summarize_numeric

MODES = ("eval", "bn_batchstats", "conditionoff")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new, nonexistent directory")
    parser.add_argument("--frame-indices", type=int, nargs="+", default=[1, 16, 33, 40])
    parser.add_argument("--crf", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def validate_args(args):
    sources = [p.resolve() for p in args.sequence_dirs]
    if not 1 <= len(sources) <= 2 or len(set(sources)) != len(sources):
        raise ValueError("Require one or two distinct sequence directories")
    ids = args.frame_indices
    if (not 1 <= len(ids) <= 4 or any(type(i) is not int or i < 1 for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Require one to four distinct positive frame indices")
    if type(args.crf) is not int or not 0 <= args.crf <= 51 or args.device not in ("cpu", "cuda"):
        raise ValueError("Require CRF integer [0,51] and device cpu or cuda")
    output = args.output.resolve()
    for source in [*sources, args.checkpoint.resolve()]:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Output must not overlap source or checkpoint paths")


@contextmanager
def unchanged_state(model):
    """Detect mutations; never copy weights or running statistics back."""
    import torch
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    try:
        yield
    finally:
        after = model.state_dict()
        if before.keys() != after.keys() or any(
                v.dtype != after[k].dtype or not torch.equal(v, after[k].detach().cpu())
                for k, v in before.items()):
            raise RuntimeError("Model state mutation detected; NOUPDATE contract violated")


@contextmanager
def batch_statistics(model, enabled):
    """Keep subclass activations; only the executing BN uses buffer-free batch stats."""
    import torch
    originals, forwards = [], {}

    def forward(layer, x):
        state = (layer.track_running_stats, layer.running_mean, layer.running_var, layer.momentum)
        layer.track_running_stats = False
        layer.running_mean = layer.running_var = None
        layer.momentum = 0.
        try:
            return forwards[id(layer)](x)
        finally:
            layer.track_running_stats, layer.running_mean, layer.running_var, layer.momentum = state

    try:
        if enabled:
            for layer in model.modules():
                if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                    originals.append((layer, "forward" in layer.__dict__, layer.__dict__.get("forward")))
                    forwards[id(layer)] = layer.forward
                    layer.forward = MethodType(forward, layer)
        yield
    finally:
        for layer, owned, original in reversed(originals):
            if owned:
                layer.forward = original
            else:
                del layer.forward


def normalize_rgb(rgb, device):
    import torch
    from torch.nn import functional as F
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None]
    return F.interpolate(x.to(device, dtype=torch.float32) / 255.0,
                         size=(256, 256), mode="bilinear", align_corners=False)


def probe_batch(model, x, crf, mode):
    """One frozen forward; return per-image scalar summaries, never feature arrays."""
    import torch
    from torch.nn import functional as F
    if (mode not in MODES or any(m.training for m in model.modules())
            or x.ndim != 4 or not 1 <= x.shape[0] <= 8 or x.shape[1:] != (3, 256, 256)
            or type(crf) is not int or not 0 <= crf <= 51):
        raise ValueError("Require eval modules, supported mode, CRF [0,51], and [1..8,3,256,256]")
    captured, handles = {}, []

    def backbone_hook(_module, _inputs, output):
        captured["pre_conditioning_features"] = output[-1] if isinstance(output, (list, tuple)) else output

    def head_hook(_module, inputs, output):
        captured["conditioned_features"] = inputs[0]
        captured["residual_head_pre_resize_pre_tanh"] = output

    with unchanged_state(model), torch.inference_mode(), batch_statistics(model, mode == "bn_batchstats"):
        try:
            handles.append(model.backbone.register_forward_hook(backbone_hook))
            handles.append(model.residual_head.register_forward_hook(head_hook))
            q = torch.full((x.shape[0], 1), crf / 51., device=x.device, dtype=x.dtype)
            if mode == "conditionoff":
                features = model.backbone(x)
                features = features[-1] if isinstance(features, (list, tuple)) else features
                raw = model.residual_head(features)
                residual = 8 * torch.tanh(F.interpolate(raw, (256, 256), mode="bilinear", align_corners=False))
            else:
                residual = model(x, q, for_encoding=False)["residual"]
                raw = captured["residual_head_pre_resize_pre_tanh"]
            rebuilt = 8 * torch.tanh(F.interpolate(raw, (256, 256), mode="bilinear", align_corners=False))
            if residual.shape != (x.shape[0], 1, 256, 256):
                raise ValueError("Unexpected residual shape")
            error = float((rebuilt - residual).abs().max().cpu())
            if not np.isfinite(error) or error > 1e-5:
                raise ValueError("Actual residual disagrees with tanh reconstruction")
            rows = []
            for i in range(x.shape[0]):
                row = {k: summarize_numeric(v[i].float().cpu().numpy()) for k, v in captured.items()}
                r = residual[i].float().cpu().numpy()
                row.update(residual_head_raw_shape=list(raw[i].shape),
                           tanh_reconstruction_max_abs_error=error, residual_256=summarize_numeric(r))
                row["residual_256"].update(fraction_abs_ge_7_99=float(np.mean(np.abs(r) >= 7.99)))
                rows.append(row)
            return rows
        finally:
            for handle in handles:
                handle.remove()


def run_probe(model, x, crf):
    """Exactly six batched forwards: mixed and repeated-first for each independent mode."""
    if x.ndim != 4 or not 1 <= x.shape[0] <= 8:
        raise ValueError("Require one to eight images")
    records, n = [], x.shape[0]
    with unchanged_state(model):
        for mode in MODES:
            for kind, batch, indices in (("mixed", x, list(range(n))),
                                          ("repeated_single", x[:1].repeat(n, 1, 1, 1), [0] * n)):
                frames = probe_batch(model, batch, crf, mode)
                records.append(dict(mode=mode, batch_kind=kind, batch_size=n,
                                    source_indices=indices, frames=frames, state_unchanged=True))
    return dict(records=records, state_unchanged=True, batch_size=n, batched_forwards=6)


def run(args):
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "controller_normalization.json"
    report = dict(status="running", scope="selected_frames_only_offline_NOUPDATE",
                  training_run=False, encoder_run=False, causal_root_cause_proven=False,
                  conditionoff="intentional_ablation_not_deployment; bypass qp_condition addition only",
                  normalization="BN original forward; eval flags unchanged; buffers=None, track=False, momentum=0",
                  color_contract="MOTSequence_BT709_limited_YUV420p8_to_original_yuv_to_rgb",
                  residual_units="continuous_float_map_not_applied_or_rounded_pixel_changes",
                  parameters=dict(sequence_dirs=[str(p.resolve()) for p in args.sequence_dirs],
                                  checkpoint=str(args.checkpoint.resolve()), crf=args.crf,
                                  frame_indices=args.frame_indices, device=args.device),
                  selected_frames=[], records=[])
    write_report(path, report)
    try:
        import torch
        report["checkpoint_sha256"] = sha256(args.checkpoint)
        identities, images = [(args.checkpoint, report["checkpoint_sha256"])], []
        for source in args.sequence_dirs:
            metadata = source / "seqinfo.ini"
            identities.append((metadata, sha256(metadata)))
            for index, source_path, digest, rgb in selected_frames(source, args.frame_indices):
                identities.append((source_path, digest))
                report["selected_frames"].append(dict(sequence_dir=str(source.resolve()),
                    frame_index=index, source_path=str(source_path.resolve()), source_sha256=digest))
                images.append(normalize_rgb(rgb, args.device))
        if len(images) != len(args.sequence_dirs) * len(args.frame_indices):
            raise ValueError("Selected image count mismatch")
        student = load_student(args.checkpoint, args.device)
        if sha256(args.checkpoint) != report["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed while loading")
        report.update(backbone=student.model.backbone_name, torch_version=torch.__version__,
                      numpy_version=np.__version__, dtype="float32", model_mode="eval_inference_mode",
                      bn_layer_count=sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                                         for m in student.model.modules()))
        report.update(run_probe(student.model, torch.cat(images), args.crf))
        if any(sha256(p) != digest for p, digest in identities):
            raise ValueError("Source or checkpoint changed during probe")
        report.update(status="completed", source_files_unchanged=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    write_report(path, report)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
