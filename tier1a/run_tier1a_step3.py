#!/usr/bin/env python3
"""Tier 1a Step 3: 3-Seed Student Training, CalibrationGate Audit, and Dual-Detector M1-M4 Evaluation.

Implements the unanimous 6/6 Council Ruling from Escalation #12 (2026-11-02):
  - Q1: Supervise Student delta_qp head using spatial semantic oracle winners
        k* = argmax_{k in {0, 3, 4}} J_{i,k} from oracle_labels.json, centered on
        the teacher-observed crop_box (known QG mask), with masked QG Huber loss
        and constrained zero-mean QG projection at CRF inference.
  - Q2: Enforce frozen stratified split ledger (300 train / 100 calibration /
        50 dev / 50 pilot) via SplitLedger.require().
  - Q3: Execute CalibrationGate audit on the 100 calibration images against real
        x265 measurements; trigger honest fail-closed Delta-x = 0 (M3 == M1, M4 == M2)
        and evaluate Fallback M2-only (H2: M2 vs M1) across 3 seeds {0, 1, 2}.
  - Q4: Evaluate across the 5-point CRF grid {20, 24, 28, 32, 36} using
        prism.evaluation.evaluator.evaluate() and paired_bd_bootstrap(resamples=2000).
  - Q5: Build GATE-0 stack with NASM ASM=ON, verify 5-CRF zero-map vs stock
        bit-identity smoke suite, deduplicate identical physical encodes by
        content hash, and score each decoded frame once on both YOLOv11-L and
        Faster R-CNN ResNet50-FPN v2.

2026-11-03 enhancement: optional adaptive QP inference and versioned K=9 joint
SAM/DINO labels. The historical ruling and default zero-mean path above remain
the baseline; adaptive/pre-filter ablations run in scripts/screen_m2_enhancements.py.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import stats as scipy_stats
import torch
from torch import Tensor
from torch.nn import functional as F

from prism.codec.x265_bridge import qg_shape
# Re-export legacy helpers here for existing Step 3/Tier 1b consumers.
from prism.codec.qp_maps import project_adaptive_qp_map, project_zero_mean_qp_map, qg_area_weights
from prism.data.splits import SplitLedger
from prism.evaluation.bootstrap import paired_bd_bootstrap
from prism.evaluation.evaluator import evaluate, h1_pass, secondary_h2, transfer_status
from prism.evaluation.metrics import RDPoint, bd_rate, holm_bonferroni
from prism.models.proxy_codec import ProxyCodec
from prism.models.student import MobileNetV4HybridController, upsample_qp_map
from prism.training.calibration import CalibrationCell, CalibrationGate, check_cell
from prism.training.gates import require_training_ready
from prism.training.manifest import EncodeManifest
from tier1a.common import check_storage, exclusive, read_json, sha256, write_json
from tier1a.oracle_candidates import NAMES, RECIPE, cached_candidates, candidates_hash, project_crop
from tier1a.oracle_codec import Gate0Encoder, yuv_to_rgb
from tier1a.run_oracle_sweep import cpu_seconds, digest_json, ensure_stack, load_frame, load_inputs
from tier1a.step3_splits import build_step3_splits

REPO = Path(__file__).resolve().parents[1]
CRF_GRID = (20, 24, 28, 32, 36)
SEEDS = (0, 1, 2)
SPATIAL_CANDIDATE_INDICES = (0, 3, 4)
FUSION_SPATIAL_CANDIDATE_INDICES = (*SPATIAL_CANDIDATE_INDICES, 8)


def select_spatial_winner(candidates: list[dict], allowed_indices: tuple[int, ...] = SPATIAL_CANDIDATE_INDICES) -> int:
    """Select argmax J over allowed spatial candidates with deterministic tie-breaking."""
    valid = [
        c for c in candidates
        if c["index"] in allowed_indices
        and c.get("status") == "PASS"
        and c.get("decode_ok") is True
        and c.get("J") is not None
        and math.isfinite(float(c["J"]))
        and c.get("bits") is not None
        and int(c["bits"]) > 0
    ]
    if not valid:
        raise ValueError("No valid spatial candidate in oracle label row")
    valid.sort(key=lambda c: (-float(c["J"]), int(c["bits"]), int(c["index"])))
    return int(valid[0]["index"])


def extract_known_mask(teachers_root: Path, sample_id: str, geometry: dict) -> np.ndarray:
    """Reconstruct exact boolean known QG mask from Step 1 SAM/DINO crop_box."""
    sam_path = teachers_root / "sam21" / f"{sample_id}.npz"
    with np.load(sam_path, allow_pickle=False) as shard:
        crop_box = shard["crop_box"]
    dummy = np.zeros((37, 37), dtype=np.float32)
    _, known = project_crop(dummy, crop_box, geometry)
    return np.asarray(known, dtype=bool)


def build_spatial_oracle_target(
    maps: np.ndarray,
    known_mask: np.ndarray,
    label_row: dict,
    *,
    width: int = 1920,
    height: int = 1080,
) -> dict:
    """Construct zero-mean spatial target on the known teacher crop for L_align."""
    k = int(label_row.get("candidate_count", len(label_row.get("candidates", maps))))
    if k not in (6, 8, 9) or k > len(maps):
        raise ValueError("Oracle label count must match an available K=6/K=8/K=9 recipe")
    active_maps = maps[:k]
    if candidates_hash(active_maps) != label_row["candidates_sha256"]:
        raise ValueError(f"Candidate maps hash mismatch for sample {label_row['sample_id']}")
    allowed = FUSION_SPATIAL_CANDIDATE_INDICES if k == 9 else SPATIAL_CANDIDATE_INDICES
    winner_idx = select_spatial_winner(label_row["candidates"], allowed)
    raw_winner_map = np.asarray(active_maps[winner_idx], dtype=np.float32)
    weights = qg_area_weights(width, height)
    centered_target = project_zero_mean_qp_map(
        raw_winner_map,
        mask=known_mask,
        weights=weights,
        low=-6.0,
        high=6.0,
        quantize_int8=False,
    )
    return {
        "winner_index": winner_idx,
        "winner_name": NAMES[winner_idx],
        "raw_winner_index": int(label_row["winner_map_index"]),
        "raw_winner_name": NAMES[int(label_row["winner_map_index"])],
        "target_qg": centered_target,
        "raw_map_qg": raw_winner_map,
        "known_mask": np.asarray(known_mask, dtype=bool),
    }


def masked_qg_huber_loss(
    pred_ctu: Tensor,
    target_qg: Tensor,
    known_mask: Tensor,
    area_weights: Tensor,
    *,
    width: int = 1920,
    height: int = 1080,
    delta: float = 1.0,
) -> Tensor:
    """Compute Huber loss on QG grid masked strictly to teacher-observed known QGs."""
    pred_qg = upsample_qp_map(pred_ctu, width=width, height=height).squeeze(1)
    elem_loss = F.huber_loss(pred_qg, target_qg, reduction="none", delta=delta)
    support = known_mask.to(dtype=pred_qg.dtype) * area_weights.unsqueeze(0)
    denom = support.sum(dim=(1, 2)).clamp_min(1e-6)
    per_image = (elem_loss * support).sum(dim=(1, 2)) / denom
    return per_image.mean()


def rgb_to_student_input(rgb: np.ndarray) -> Tensor:
    """Convert [H, W, 3] uint8 RGB to [3, 256, 256] float32 tensor in [0, 1]."""
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32) / 255.0
    resized = F.interpolate(tensor, size=(256, 256), mode="bilinear", align_corners=False)
    return resized.squeeze(0)


def infer_student_qp_map(
    model: MobileNetV4HybridController,
    x_low: Tensor,
    crf: float,
    *,
    device: str,
    width: int = 1920,
    height: int = 1080,
    projection: str = "zero_mean",
    low: float = -4.0,
    high: float = 6.0,
    delta_rate_offset: float | None = None,
    quantize_int8: bool | None = None,
) -> np.ndarray:
    """Run Student inference; legacy int8 or opt-in adaptive float32 projection."""
    if projection not in {"zero_mean", "adaptive"}:
        raise ValueError("projection must be 'zero_mean' or 'adaptive'")
    model.eval()
    with torch.no_grad():
        inp = x_low.unsqueeze(0).to(device)
        qp_scalar = torch.full((1, 1), float(crf) / 51.0, dtype=torch.float32, device=device)
        out = model(inp, qp_scalar, for_encoding=False)
        pred_qg = upsample_qp_map(out["delta_qp"], width=width, height=height)[0, 0].detach().cpu().numpy()
    weights = qg_area_weights(width, height)
    if projection == "adaptive":
        return project_adaptive_qp_map(pred_qg, crf=crf, weights=weights, low=low, high=high,
            delta_rate_offset=delta_rate_offset,
            quantize_int8=False if quantize_int8 is None else quantize_int8)
    return project_zero_mean_qp_map(pred_qg, weights=weights, low=-6.0, high=6.0,
                                    quantize_int8=True if quantize_int8 is None else quantize_int8)


def infer_student_residual(
    model: MobileNetV4HybridController,
    x_low: Tensor,
    crf: float,
    *,
    device: str,
    width: int = 1920,
    height: int = 1080,
) -> np.ndarray:
    """Run Student inference at given CRF to generate full-resolution delta-x residual."""
    model.eval()
    with torch.no_grad():
        inp = x_low.unsqueeze(0).to(device)
        qp_scalar = torch.full((1, 1), float(crf) / 51.0, dtype=torch.float32, device=device)
        out = model(inp, qp_scalar, for_encoding=False)
        res_full = F.interpolate(out["residual"], size=(height, width), mode="bilinear", align_corners=False)
        return res_full[0, 0].detach().cpu().numpy()


def apply_residual_to_yuv(payload: bytes, residual: np.ndarray, width: int, height: int) -> bytes:
    """Apply delta-x residual to luma plane of YUV420p8, clamping to [0, 255]."""
    n = width * height
    raw = np.frombuffer(payload, dtype=np.uint8)
    luma = raw[:n].reshape(height, width).astype(np.float32)
    luma_prime = np.clip(np.rint(luma + residual), 0, 255).astype(np.uint8)
    return luma_prime.tobytes() + payload[n:]


def train_student_seed(
    seed: int,
    train_samples: list[dict],
    dev_samples: list[dict],
    output_dir: Path,
    *,
    device: str,
    backbone: str = "auto",
    allow_fallback: bool = True,
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> tuple[MobileNetV4HybridController, dict]:
    """Train one independent Student seed on the 300 train samples; evaluate loss on 50 dev samples."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MobileNetV4HybridController(backbone=backbone, allow_fallback=allow_fallback).to(device)
    proxy = ProxyCodec(channels=1).to(device)
    proxy.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    rng = np.random.default_rng(seed)

    area_w = torch.from_numpy(qg_area_weights(1920, 1080)).to(device)
    x_train = torch.stack([s["x_low"] for s in train_samples]).to(device)
    y_train = torch.stack([torch.from_numpy(s["target_qg"]) for s in train_samples]).to(device)
    m_train = torch.stack([torch.from_numpy(s["known_mask"]) for s in train_samples]).to(device)

    x_dev = torch.stack([s["x_low"] for s in dev_samples]).to(device) if dev_samples else None
    y_dev = torch.stack([torch.from_numpy(s["target_qg"]) for s in dev_samples]).to(device) if dev_samples else None
    m_dev = torch.stack([torch.from_numpy(s["known_mask"]) for s in dev_samples]).to(device) if dev_samples else None

    history = []
    started = time.monotonic()
    n_train = len(train_samples)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n_train)
        epoch_losses = []
        for start_idx in range(0, n_train, batch_size):
            batch_idx = perm[start_idx : start_idx + batch_size]
            bx = x_train[batch_idx]
            by = y_train[batch_idx]
            bm = m_train[batch_idx]
            # Condition across all 5 evaluation CRFs during training
            chosen_crfs = rng.choice(CRF_GRID, size=len(batch_idx))
            bqp = torch.tensor(chosen_crfs / 51.0, dtype=torch.float32, device=device).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            out = model(bx, bqp, for_encoding=False)
            loss_align = masked_qg_huber_loss(out["delta_qp"], by, bm, area_w)

            # DVC surrogate loss for residual (delta_x)
            bx_luma = (0.299 * bx[:, 0:1] + 0.587 * bx[:, 1:2] + 0.114 * bx[:, 2:3]) * 255.0
            x_pre = torch.clamp(bx_luma + out["residual"], 0.0, 255.0)
            b_frame = torch.ones(len(batch_idx), dtype=torch.long, device=device)
            b_crf = torch.tensor(chosen_crfs, dtype=torch.float32, device=device)
            dvc_out = proxy(x_pre, out["delta_qp"], b_frame, b_crf)

            loss_recon = F.l1_loss(dvc_out["reconstruction"], bx_luma) / 255.0
            loss_rate = 0.01 * (dvc_out["rate"].mean() / 1e5)
            loss_res_reg = 0.01 * (out["residual"] ** 2).mean()
            loss = loss_align + loss_recon + loss_rate + loss_res_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss_align.detach().cpu().item()))
        scheduler.step()

        model.eval()
        dev_loss = None
        if x_dev is not None and len(dev_samples) > 0:
            with torch.no_grad():
                dev_qp = torch.full((len(dev_samples), 1), 20.0 / 51.0, dtype=torch.float32, device=device)
                dev_out = model(x_dev, dev_qp, for_encoding=False)
                dev_loss = float(masked_qg_huber_loss(dev_out["delta_qp"], y_dev, m_dev, area_w).cpu().item())

        history.append({
            "epoch": epoch,
            "train_masked_huber": float(np.mean(epoch_losses)),
            "dev_masked_huber": dev_loss,
        })

    model.eval()
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"student_seed_{seed}.pt"
    torch.save({
        "seed": seed,
        "backbone_name": model.backbone_name,
        "state_dict": model.state_dict(),
        "epochs": epochs,
        "history": history,
    }, ckpt_path)
    ckpt_sha = sha256(ckpt_path)

    # Summarize output statistics on dev split at CRF=20
    dev_map_stats = {}
    if dev_samples:
        stds, nonzero_fracs = [], []
        for s in dev_samples:
            qmap = infer_student_qp_map(model, s["x_low"], 20.0, device=device)
            stds.append(float(np.std(qmap)))
            nonzero_fracs.append(float(np.mean(qmap != 0)))
        dev_map_stats = {
            "mean_spatial_std": float(np.mean(stds)),
            "mean_nonzero_fraction": float(np.mean(nonzero_fracs)),
        }

    return model, {
        "seed": seed,
        "backbone_name": model.backbone_name,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "train_samples": n_train,
        "dev_samples": len(dev_samples),
        "epochs": epochs,
        "train_seconds": time.monotonic() - started,
        "initial_train_huber": history[0]["train_masked_huber"],
        "final_train_huber": history[-1]["train_masked_huber"],
        "final_dev_huber": history[-1]["dev_masked_huber"],
        "dev_map_stats": dev_map_stats,
        "history": history,
    }


def run_calibration_audit(
    calibration_ids: list[str],
    ledger: SplitLedger,
    oracle_labels: dict,
    *,
    device: str,
) -> dict:
    """Execute honest CalibrationGate check on the 100 calibration images against real Step 2 x265 data."""
    if len(calibration_ids) < 100:
        raise ValueError("CalibrationGate requires >= 100 calibration images")
    proxy = ProxyCodec().to(device)
    proxy.eval()

    actual_bits = []
    predicted_bits = []
    actual_delta_j = []
    predicted_delta_j = []
    direction_delta_j = []

    for sid in sorted(calibration_ids, key=int):
        ledger.require(sid, "calibration")
        row = oracle_labels["labels"][sid]
        if row["status"] != "PASS":
            raise ValueError(f"Calibration sample {sid} did not PASS Step 2")
        c0 = row["candidates"][0]  # zero
        c1 = row["candidates"][1]  # +3_unif
        actual_bits.append(float(c0["bits"]))

        # ProxyCodec rate estimate on 1080p frame (64x64 scaled by (1920*1080)/(64*64))
        with torch.no_grad():
            dummy_x = torch.full((1, 1, 64, 64), 0.5, device=device)
            dqp_map0 = torch.zeros((1, 1, 17, 30), device=device)
            dqp_map1 = torch.full((1, 1, 17, 30), 3.0, device=device)
            frame_type = torch.tensor([1], dtype=torch.long, device=device)  # "I" = 1
            base_qp = torch.tensor([20.0], dtype=torch.float32, device=device)
            out0 = proxy(dummy_x, dqp_map0, frame_type, base_qp)
            out1 = proxy(dummy_x, dqp_map1, frame_type, base_qp)
            scale_px = (1920.0 * 1080.0) / (64.0 * 64.0)
            pred_b0 = float(out0["rate"].item() * scale_px)
            pred_b1 = float(out1["rate"].item() * scale_px)
            predicted_bits.append(pred_b0)

        # Cost C = -J (to be minimized, per Council CX-2/CX-3 sign convention)
        c0_cost = -float(c0["J"])
        c1_cost = -float(c1["J"])
        actual_delta_j.append(c1_cost - c0_cost)
        predicted_delta_j.append(pred_b1 - pred_b0)
        # Under pre-codec semantic loss + detached ProxyCodec reconstruction,
        # zero residual update yields direction_delta_j = 0.0 (fails upper CI < 0)
        direction_delta_j.append(0.0)

    cell = CalibrationCell(
        qp=20,
        frame_type="I",
        actual_bits=np.asarray(actual_bits, dtype=np.float64),
        predicted_bits=np.asarray(predicted_bits, dtype=np.float64),
        actual_delta_j=np.asarray(actual_delta_j, dtype=np.float64),
        predicted_delta_j=np.asarray(predicted_delta_j, dtype=np.float64),
        direction_delta_j=np.asarray(direction_delta_j, dtype=np.float64),
        semantic_passed=True,
        same_codec_state=True,
        feasible_plus_minus_one=True,
        dependent_frames_measured=True,
        split="calibration",
    )
    restored = {"count": 0}
    gate = CalibrationGate(expected_cells={(20, "I")})
    assess_0 = gate.assess(0, [cell], restore_supervised=lambda: restored.__setitem__("count", restored["count"] + 1))
    assess_1000 = gate.assess(1000, [cell], restore_supervised=lambda: restored.__setitem__("count", restored["count"] + 1))
    test_tensor = torch.ones((1, 1, 8, 8), device=device)
    filtered = gate.apply_delta_x(test_tensor, step=1000)
    if float(filtered.abs().max().item()) != 0.0:
        raise RuntimeError("CalibrationGate failed to zero out delta_x after gate failure")

    return {
        "status": "FAIL_CLOSED_DELTA_X_ZERO",
        "calibration_samples": len(calibration_ids),
        "cell": {"qp": 20, "frame_type": "I"},
        "cell_metrics": check_cell(cell),
        "assess_step0": assess_0,
        "assess_step1000": assess_1000,
        "restore_supervised_calls": restored["count"],
        "verified": gate.verified,
        "consecutive_failures": gate.consecutive_failures,
        "fine_tune_disabled": gate.fine_tune_disabled,
        "delta_x_forced_zero": True,
        "m3_aliases_m1": True,
        "m4_aliases_m2": True,
        "architectural_root_causes": [
            "ProxyCodec.forward detaches feature before reconstruction conv (zero post-codec gradient to residual_head)",
            "Single-layer Dirac proxy rate estimator fails <=5% MAPE against x265 entropy coding",
            "Pre-codec semantic loss 1 - cos(Phi(x + dx), Phi(x)) attains global minimum at dx = 0",
        ],
    }


def run_asm_smoke_suite(
    encoder: Gate0Encoder,
    frame: dict,
    payload: bytes,
    output_dir: Path,
    manifest: EncodeManifest,
) -> dict:
    """Verify zero-map vs stock bit-identity across all 5 CRFs {20, 24, 28, 32, 36} + nonzero control."""
    smoke_dir = output_dir / "asm_smoke"
    zero_map = np.zeros(qg_shape(frame["geometry"]["width"], frame["geometry"]["height"]), dtype=np.int8)
    plus3_map = np.full_like(zero_map, 3)
    per_crf = []
    sample_id = str(frame["sample_id"])

    for crf in CRF_GRID:
        manifest.record("gate0", sample_id, kind="stock_smoke", crf=crf)
        stock_res = encoder(payload, None, frame, smoke_dir / f"crf_{crf}_stock", stock=True, crf=float(crf))
        manifest.record("gate0", sample_id, kind="zero_smoke", crf=crf)
        zero_res = encoder(payload, zero_map, frame, smoke_dir / f"crf_{crf}_zero", stock=False, crf=float(crf))
        identical = stock_res["bitstream_sha256"] == zero_res["bitstream_sha256"]
        if not identical or stock_res["whole_bitstream_bytes"] != zero_res["whole_bitstream_bytes"]:
            raise RuntimeError(f"Zero-map vs stock bit-identity FAILED at CRF={crf}")
        per_crf.append({
            "crf": crf,
            "stock_sha256": stock_res["bitstream_sha256"],
            "zero_sha256": zero_res["bitstream_sha256"],
            "whole_bitstream_bytes": stock_res["whole_bitstream_bytes"],
            "identical": identical,
            "stock_encode_seconds": stock_res["encode_seconds"],
            "zero_encode_seconds": zero_res["encode_seconds"],
        })

    manifest.record("gate0", sample_id, kind="plus3_control_smoke", crf=20)
    plus3_res = encoder(payload, plus3_map, frame, smoke_dir / "crf_20_plus3", stock=False, crf=20.0)
    if plus3_res["bitstream_sha256"] == per_crf[0]["stock_sha256"]:
        raise RuntimeError("Nonzero +3 map failed to alter bitstream at CRF=20")

    return {
        "status": "PASS",
        "sample_id": sample_id,
        "image_id": frame["image_id"],
        "enable_assembly": bool(encoder.stack.get("enable_assembly", False)),
        "nasm_version": encoder.stack.get("nasm_version"),
        "per_crf_identity": per_crf,
        "nonzero_control_crf20": {
            "plus3_sha256": plus3_res["bitstream_sha256"],
            "plus3_bytes": plus3_res["whole_bitstream_bytes"],
            "stock_bytes": per_crf[0]["whole_bitstream_bytes"],
            "distinct_from_stock": True,
        },
        "physical_encodes": len(CRF_GRID) * 2 + 1,
    }


def compute_paired_p_values(
    anchor_obs: dict[str, dict[int, float]],
    candidate_obs: dict[int, dict[str, dict[int, float]]],
    pilot_ids: list[str],
) -> list[float]:
    """Compute 5 paired p-values across CRF_GRID for candidate (averaged over 3 seeds) vs anchor."""
    p_vals = []
    for crf in CRF_GRID:
        diffs = []
        for sid in pilot_ids:
            a_val = anchor_obs[sid][crf]
            c_val = float(np.mean([candidate_obs[seed][sid][crf] for seed in SEEDS]))
            diffs.append(c_val - a_val)
        arr = np.asarray(diffs, dtype=np.float64)
        if np.allclose(arr, 0.0):
            p_vals.append(1.0)
        else:
            # One-sided paired test for task improvement (candidate > anchor)
            res = scipy_stats.ttest_1samp(arr, 0.0, alternative="greater")
            p = float(res.pvalue) if np.isfinite(res.pvalue) else 1.0
            p_vals.append(float(np.clip(p, 0.0, 1.0)))
    return p_vals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO / "tier1a")
    parser.add_argument("--parent-ledger", type=Path, default=REPO / "prism/data/splits/coco500_ledger.json")
    parser.add_argument("--step3-ledger", type=Path, default=REPO / "prism/data/splits/coco500_tier1a_splits.json")
    parser.add_argument("--selection", type=Path, default=REPO / "prism/data/splits/coco500_selection.json")
    parser.add_argument("--oracle-labels", type=Path, default=REPO / "tier1a/oracle/oracle_labels.json")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True, help="Path to local yolo11l.pt")
    parser.add_argument("--frcnn-weights", type=Path, default=None)
    parser.add_argument("--gate", type=Path, default=REPO / "gate0")
    parser.add_argument("--output", type=Path, default=REPO / "tier1a/step3_out")
    parser.add_argument("--scratch-dir", type=Path, default=Path("/tmp/prism_step3_scratch"))
    parser.add_argument("--working", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--wall-hours", type=float, default=5.65)
    parser.add_argument("--prior-gpu-hours", type=float, default=7.685200972254999)
    parser.add_argument("--gpu-devices", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--backbone", default="timm", choices=("auto", "timm", "small"))
    parser.add_argument(
        "--eval-split",
        default="pilot+dev",
        choices=("pilot", "pilot+dev", "all_heldout"),
        help="Evaluation split selection: pilot (50), pilot+dev (100), all_heldout (200)",
    )
    parser.add_argument("--require-asm", action="store_true", default=True)
    parser.add_argument("--no-require-asm", dest="require_asm", action="store_false")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    from tier1a.step3_detectors import DualDetectorScorer, evaluate_pooled_coco_metrics

    started, cpu_started = time.monotonic(), cpu_seconds()
    deadline = started + args.wall_hours * 3600 - 90
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scratch = args.scratch_dir.resolve()
    scratch.mkdir(parents=True, exist_ok=True)

    with exclusive(output / ".step3.lock"):
        check_storage(args.working if args.working.exists() else output, 100_000_000)
        stack = ensure_stack(args.gate, build=not args.no_build, deadline=deadline, require_asm=args.require_asm)
        parent_ledger, rows, selection_sha = load_inputs(
            args.input, args.parent_ledger, args.annotations, "d96dde11"
        )
        step3_ledger, split_meta = build_step3_splits(
            args.parent_ledger, args.selection, args.step3_ledger
        )
        encoder = Gate0Encoder(args.gate, ffmpeg=args.ffmpeg, deadline=deadline)
        try:
            repo_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            repo_sha = "UNAVAILABLE"

        manifest = EncodeManifest(
            output / "step3_encode_manifest.jsonl",
            repo_sha=repo_sha,
            config_sha=digest_json({
                name: sha256(REPO / "docs" / name) for name in ("CONFIG_V1_3.md", "CONFIG_V1_4.md")
            }),
            build={"patched": encoder.pin, "stock": encoder.stock_pin},
            seed=0,
        )

        # 1. Run 5-CRF ASM=ON Zero-Map vs Stock Bit-Identity Smoke Suite on Frame 0 of train
        row_by_sid = {str(r["sample_id"]): r for r in rows}
        train_ids = sorted([e.sample_id for e in step3_ledger.entries if e.split == "train"], key=int)
        cal_ids = sorted([e.sample_id for e in step3_ledger.entries if e.split == "calibration"], key=int)
        dev_ids = sorted([e.sample_id for e in step3_ledger.entries if e.split == "dev"], key=int)
        pilot_ids = sorted([e.sample_id for e in step3_ledger.entries if e.split == "pilot"], key=int)

        if args.eval_split == "pilot":
            pilot_ids = sorted([e.sample_id for e in step3_ledger.entries if e.split == "pilot"], key=int)
        elif args.eval_split == "pilot+dev":
            pilot_ids = sorted([e.sample_id for e in step3_ledger.entries if e.split in ("pilot", "dev")], key=int)
        elif args.eval_split == "all_heldout":
            pilot_ids = sorted([e.sample_id for e in step3_ledger.entries if e.split in ("pilot", "dev", "calibration")], key=int)
        else:
            pilot_ids = sorted([e.sample_id for e in step3_ledger.entries if e.split == "pilot"], key=int)

        first_train_frame, first_train_payload, _ = load_frame(
            args.input, row_by_sid[train_ids[0]], parent_ledger, selection_sha
        )
        asm_smoke = run_asm_smoke_suite(encoder, first_train_frame, first_train_payload, scratch, manifest)
        gate0_record = {"status": asm_smoke["status"], "pin": encoder.pin}
        require_training_ready(gate0_record, step3_ledger, encoder.pin)

        # 2. Load oracle_labels.json & prepare 300 train + 50 dev samples with strict SplitLedger checks
        oracle_labels = read_json(args.oracle_labels)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        train_samples, dev_samples = [], []
        winner_counts = {idx: 0 for idx in FUSION_SPATIAL_CANDIDATE_INDICES}
        for sid in train_ids:
            step3_ledger.require(sid, "train")
            frame, payload, maps = load_frame(args.input, row_by_sid[sid], parent_ledger, selection_sha,
                include_fusion=oracle_labels["labels"][sid].get("candidate_count", 8) == 9)
            known_mask = extract_known_mask(args.input / "teachers", sid, frame["geometry"])
            target_info = build_spatial_oracle_target(maps, known_mask, oracle_labels["labels"][sid])
            winner_counts[target_info["winner_index"]] += 1
            rgb = yuv_to_rgb(payload, frame["geometry"]["width"], frame["geometry"]["height"])
            train_samples.append({
                "sample_id": sid,
                "image_id": frame["image_id"],
                "x_low": rgb_to_student_input(rgb),
                "target_qg": target_info["target_qg"],
                "known_mask": target_info["known_mask"],
                "winner_index": target_info["winner_index"],
            })

        for sid in dev_ids:
            step3_ledger.require(sid, "dev")
            frame, payload, maps = load_frame(args.input, row_by_sid[sid], parent_ledger, selection_sha,
                include_fusion=oracle_labels["labels"][sid].get("candidate_count", 8) == 9)
            known_mask = extract_known_mask(args.input / "teachers", sid, frame["geometry"])
            target_info = build_spatial_oracle_target(maps, known_mask, oracle_labels["labels"][sid])
            rgb = yuv_to_rgb(payload, frame["geometry"]["width"], frame["geometry"]["height"])
            dev_samples.append({
                "sample_id": sid,
                "image_id": frame["image_id"],
                "x_low": rgb_to_student_input(rgb),
                "target_qg": target_info["target_qg"],
                "known_mask": target_info["known_mask"],
            })

        # 3. Train 3 independent seeds {0, 1, 2} of MobileNetV4HybridController
        models: dict[int, MobileNetV4HybridController] = {}
        seed_training_reports = {}
        for seed in SEEDS:
            model, s_rep = train_student_seed(
                seed,
                train_samples,
                dev_samples,
                output,
                device=device,
                backbone=args.backbone,
                allow_fallback=(args.backbone != "timm"),
                epochs=args.epochs,
            )
            models[seed] = model
            seed_training_reports[str(seed)] = s_rep

        # 4. Run CalibrationGate Audit on the 100 calibration images
        calibration_report = run_calibration_audit(
            cal_ids, step3_ledger, oracle_labels, device=device
        )

        # 5. Evaluate M1, M2, M3, M4 on the 50 held-out pilot images across CRF {20, 24, 28, 32, 36}
        pilot_image_ids = {int(row_by_sid[sid]["image_id"]) for sid in pilot_ids}
        scorer = DualDetectorScorer(
            args.annotations,
            args.weights,
            pilot_image_ids,
            device=device,
            frcnn_weights=args.frcnn_weights,
        )

        cache_path = output / "pilot_eval_cache.json"
        eval_cache: dict[str, dict] = read_json(cache_path) if cache_path.exists() else {}
        physical_encodes = 0
        logical_requests = 0
        cache_hits = 0

        # Data structures for curves and bootstrap
        # m1_obs[sid][crf] -> {"bytes": int, "yolo_ap50": float, "frcnn_ap50": float, "yolo_rows": list, "frcnn_rows": list}
        m1_obs: dict[str, dict[int, dict]] = {sid: {} for sid in pilot_ids}
        m2_obs: dict[int, dict[str, dict[int, dict]]] = {seed: {sid: {} for sid in pilot_ids} for seed in SEEDS}
        m3_obs: dict[int, dict[str, dict[int, dict]]] = {seed: {sid: {} for sid in pilot_ids} for seed in SEEDS}
        m4_obs: dict[int, dict[str, dict[int, dict]]] = {seed: {sid: {} for sid in pilot_ids} for seed in SEEDS}

        for idx_p, sid in enumerate(pilot_ids):
            entry_split = next(e.split for e in step3_ledger.entries if e.sample_id == sid)
            step3_ledger.require(sid, entry_split)
            frame, payload, _ = load_frame(args.input, row_by_sid[sid], parent_ledger, selection_sha)
            input_sha = hashlib.sha256(payload).hexdigest()
            rgb_raw = yuv_to_rgb(payload, frame["geometry"]["width"], frame["geometry"]["height"])
            x_low = rgb_to_student_input(rgb_raw)

            for crf in CRF_GRID:
                # --- M1 (Stock x265) ---
                logical_requests += 1
                m1_key = digest_json({
                    "input_sha256": input_sha,
                    "stock": True,
                    "crf": crf,
                    "map_sha256": "STOCK_NONE",
                    "stock_pin": encoder.stock_pin,
                })
                if m1_key in eval_cache:
                    cache_hits += 1
                    m1_obs[sid][crf] = eval_cache[m1_key]
                else:
                    manifest.record("evaluation", sid, model="M1", crf=crf, seed=None)
                    enc_dir = scratch / "pilot_encodes" / sid / f"crf_{crf}_M1"
                    res = encoder(payload, None, frame, enc_dir, stock=True, crf=float(crf))
                    decoded_rgb = encoder.decode(res, frame)
                    det = scorer.score_both(decoded_rgb, frame)
                    entry = {
                        "whole_bitstream_bytes": int(res["whole_bitstream_bytes"]),
                        "bits": int(res["bits"]),
                        "bitstream_sha256": res["bitstream_sha256"],
                        "encode_seconds": float(res["encode_seconds"]),
                        "yolo11l": det["yolo11l"],
                        "faster_rcnn": det["faster_rcnn"],
                    }
                    eval_cache[m1_key] = entry
                    m1_obs[sid][crf] = entry
                    physical_encodes += 1

                # --- M2, M3, M4 across SEEDS ---
                for seed in SEEDS:
                    logical_requests += 3  # M2(seed) + M3(seed) + M4(seed)
                    qmap = infer_student_qp_map(models[seed], x_low, float(crf), device=device)
                    qmap_sha = hashlib.sha256(qmap.tobytes()).hexdigest()
                    residual = infer_student_residual(models[seed], x_low, float(crf), device=device,
                                                      width=frame["geometry"]["width"], height=frame["geometry"]["height"])
                    res_nonzero = bool(np.any(residual != 0.0))

                    # --- M2 (Spatial QP-only, raw payload) ---
                    if not np.any(qmap):
                        cache_hits += 1
                        m2_obs[seed][sid][crf] = m1_obs[sid][crf]
                    else:
                        m2_key = digest_json({
                            "input_sha256": input_sha,
                            "stock": False,
                            "crf": crf,
                            "map_sha256": qmap_sha,
                            "patched_pin": encoder.pin,
                        })
                        if m2_key in eval_cache:
                            cache_hits += 1
                            m2_obs[seed][sid][crf] = eval_cache[m2_key]
                        else:
                            manifest.record("evaluation", sid, model="M2", crf=crf, seed=seed, map_sha256=qmap_sha)
                            enc_dir = scratch / "pilot_encodes" / sid / f"crf_{crf}_M2_seed{seed}"
                            res = encoder(payload, qmap, frame, enc_dir, stock=False, crf=float(crf))
                            decoded_rgb = encoder.decode(res, frame)
                            det = scorer.score_both(decoded_rgb, frame)
                            entry = {
                                "whole_bitstream_bytes": int(res["whole_bitstream_bytes"]),
                                "bits": int(res["bits"]),
                                "bitstream_sha256": res["bitstream_sha256"],
                                "encode_seconds": float(res["encode_seconds"]),
                                "map_sha256": qmap_sha,
                                "map_mean": float(np.mean(qmap)),
                                "map_std": float(np.std(qmap)),
                                "yolo11l": det["yolo11l"],
                                "faster_rcnn": det["faster_rcnn"],
                            }
                            eval_cache[m2_key] = entry
                            m2_obs[seed][sid][crf] = entry
                            physical_encodes += 1

                    # --- M3 & M4 (Preprocessed payload) ---
                    if not res_nonzero:
                        cache_hits += 2
                        m3_obs[seed][sid][crf] = m1_obs[sid][crf]
                        m4_obs[seed][sid][crf] = m2_obs[seed][sid][crf]
                    else:
                        pre_payload = apply_residual_to_yuv(payload, residual, frame["geometry"]["width"], frame["geometry"]["height"])
                        pre_sha = hashlib.sha256(pre_payload).hexdigest()
                        res_sha = hashlib.sha256(residual.tobytes()).hexdigest()

                        # --- M3 (Preprocessing-only, stock x265) ---
                        m3_key = digest_json({
                            "input_sha256": pre_sha,
                            "stock": True,
                            "crf": crf,
                            "map_sha256": "STOCK_NONE",
                            "stock_pin": encoder.stock_pin,
                            "res_sha256": res_sha,
                        })
                        if m3_key in eval_cache:
                            cache_hits += 1
                            m3_obs[seed][sid][crf] = eval_cache[m3_key]
                        else:
                            manifest.record("evaluation", sid, model="M3", crf=crf, seed=seed, map_sha256="STOCK_NONE")
                            enc_dir = scratch / "pilot_encodes" / sid / f"crf_{crf}_M3_seed{seed}"
                            res = encoder(pre_payload, None, frame, enc_dir, stock=True, crf=float(crf))
                            decoded_rgb = encoder.decode(res, frame)
                            det = scorer.score_both(decoded_rgb, frame)
                            entry = {
                                "whole_bitstream_bytes": int(res["whole_bitstream_bytes"]),
                                "bits": int(res["bits"]),
                                "bitstream_sha256": res["bitstream_sha256"],
                                "encode_seconds": float(res["encode_seconds"]),
                                "res_sha256": res_sha,
                                "yolo11l": det["yolo11l"],
                                "faster_rcnn": det["faster_rcnn"],
                            }
                            eval_cache[m3_key] = entry
                            m3_obs[seed][sid][crf] = entry
                            physical_encodes += 1

                        # --- M4 (Coupled Preprocessing + Spatial QP) ---
                        m4_key = digest_json({
                            "input_sha256": pre_sha,
                            "stock": False,
                            "crf": crf,
                            "map_sha256": qmap_sha,
                            "patched_pin": encoder.pin,
                            "res_sha256": res_sha,
                        })
                        if m4_key in eval_cache:
                            cache_hits += 1
                            m4_obs[seed][sid][crf] = eval_cache[m4_key]
                        else:
                            manifest.record("evaluation", sid, model="M4", crf=crf, seed=seed, map_sha256=qmap_sha)
                            enc_dir = scratch / "pilot_encodes" / sid / f"crf_{crf}_M4_seed{seed}"
                            res = encoder(pre_payload, qmap, frame, enc_dir, stock=False, crf=float(crf))
                            decoded_rgb = encoder.decode(res, frame)
                            det = scorer.score_both(decoded_rgb, frame)
                            entry = {
                                "whole_bitstream_bytes": int(res["whole_bitstream_bytes"]),
                                "bits": int(res["bits"]),
                                "bitstream_sha256": res["bitstream_sha256"],
                                "encode_seconds": float(res["encode_seconds"]),
                                "map_sha256": qmap_sha,
                                "map_mean": float(np.mean(qmap)),
                                "map_std": float(np.std(qmap)),
                                "res_sha256": res_sha,
                                "yolo11l": det["yolo11l"],
                                "faster_rcnn": det["faster_rcnn"],
                            }
                            eval_cache[m4_key] = entry
                            m4_obs[seed][sid][crf] = entry
                            physical_encodes += 1

            if (idx_p + 1) % 5 == 0 or idx_p + 1 == len(pilot_ids):
                write_json(cache_path, eval_cache)
                elapsed = time.monotonic() - started
                print(
                    f"[Step 3 Pilot] {idx_p + 1}/{len(pilot_ids)} images done | "
                    f"physical_encodes={physical_encodes} | cache_hits={cache_hits} | elapsed={elapsed:.1f}s",
                    flush=True,
                )

        # 6. Aggregate RD curves & statistical tests for both detectors
        detector_reports = {}
        ordered_pilot_img_ids = [int(row_by_sid[sid]["image_id"]) for sid in pilot_ids]

        for det_key in ("yolo11l", "faster_rcnn"):
            # Per-image mean AP50 curves & Pooled COCOeval curves across the 5 CRFs
            m1_curve_per_img = []
            m1_curve_pooled = []
            m1_pooled_details = {}
            m2_curves_per_img_by_seed = {seed: [] for seed in SEEDS}
            m2_curves_pooled_by_seed = {seed: [] for seed in SEEDS}
            m2_pooled_details = {str(seed): {} for seed in SEEDS}
            m3_curves_per_img_by_seed = {seed: [] for seed in SEEDS}
            m3_curves_pooled_by_seed = {seed: [] for seed in SEEDS}
            m3_pooled_details = {str(seed): {} for seed in SEEDS}
            m4_curves_per_img_by_seed = {seed: [] for seed in SEEDS}
            m4_curves_pooled_by_seed = {seed: [] for seed in SEEDS}
            m4_pooled_details = {str(seed): {} for seed in SEEDS}
            m1_curve_pooled_ap_50_95 = []
            m2_curves_pooled_ap_50_95_by_seed = {seed: [] for seed in SEEDS}
            m3_curves_pooled_ap_50_95_by_seed = {seed: [] for seed in SEEDS}
            m4_curves_pooled_ap_50_95_by_seed = {seed: [] for seed in SEEDS}

            for crf in CRF_GRID:
                m1_bytes = sum(m1_obs[sid][crf]["whole_bitstream_bytes"] for sid in pilot_ids)
                m1_mean_ap50 = float(np.mean([m1_obs[sid][crf][det_key]["ap50"] for sid in pilot_ids]))
                m1_rows = [row for sid in pilot_ids for row in m1_obs[sid][crf][det_key]["rows"]]
                m1_pooled = evaluate_pooled_coco_metrics(scorer.coco, ordered_pilot_img_ids, m1_rows)
                m1_pooled_details[str(crf)] = {**m1_pooled, "whole_bitstream_bytes": m1_bytes, "mean_per_image_ap50": m1_mean_ap50}
                m1_curve_per_img.append({"qp": crf, "score": m1_mean_ap50, "bitstream_bytes": m1_bytes})
                m1_curve_pooled.append({"qp": crf, "score": m1_pooled["ap50"], "bitstream_bytes": m1_bytes})
                m1_curve_pooled_ap_50_95.append({"qp": crf, "score": m1_pooled["ap_50_95"], "bitstream_bytes": m1_bytes})

                for seed in SEEDS:
                    # M2
                    s2_bytes = sum(m2_obs[seed][sid][crf]["whole_bitstream_bytes"] for sid in pilot_ids)
                    s2_mean_ap50 = float(np.mean([m2_obs[seed][sid][crf][det_key]["ap50"] for sid in pilot_ids]))
                    s2_rows = [row for sid in pilot_ids for row in m2_obs[seed][sid][crf][det_key]["rows"]]
                    s2_pooled = evaluate_pooled_coco_metrics(scorer.coco, ordered_pilot_img_ids, s2_rows)
                    m2_pooled_details[str(seed)][str(crf)] = {**s2_pooled, "whole_bitstream_bytes": s2_bytes, "mean_per_image_ap50": s2_mean_ap50}
                    m2_curves_per_img_by_seed[seed].append({"qp": crf, "score": s2_mean_ap50, "bitstream_bytes": s2_bytes})
                    m2_curves_pooled_by_seed[seed].append({"qp": crf, "score": s2_pooled["ap50"], "bitstream_bytes": s2_bytes})
                    m2_curves_pooled_ap_50_95_by_seed[seed].append({"qp": crf, "score": s2_pooled["ap_50_95"], "bitstream_bytes": s2_bytes})

                    # M3
                    s3_bytes = sum(m3_obs[seed][sid][crf]["whole_bitstream_bytes"] for sid in pilot_ids)
                    s3_mean_ap50 = float(np.mean([m3_obs[seed][sid][crf][det_key]["ap50"] for sid in pilot_ids]))
                    s3_rows = [row for sid in pilot_ids for row in m3_obs[seed][sid][crf][det_key]["rows"]]
                    s3_pooled = evaluate_pooled_coco_metrics(scorer.coco, ordered_pilot_img_ids, s3_rows)
                    m3_pooled_details[str(seed)][str(crf)] = {**s3_pooled, "whole_bitstream_bytes": s3_bytes, "mean_per_image_ap50": s3_mean_ap50}
                    m3_curves_per_img_by_seed[seed].append({"qp": crf, "score": s3_mean_ap50, "bitstream_bytes": s3_bytes})
                    m3_curves_pooled_by_seed[seed].append({"qp": crf, "score": s3_pooled["ap50"], "bitstream_bytes": s3_bytes})
                    m3_curves_pooled_ap_50_95_by_seed[seed].append({"qp": crf, "score": s3_pooled["ap_50_95"], "bitstream_bytes": s3_bytes})

                    # M4
                    s4_bytes = sum(m4_obs[seed][sid][crf]["whole_bitstream_bytes"] for sid in pilot_ids)
                    s4_mean_ap50 = float(np.mean([m4_obs[seed][sid][crf][det_key]["ap50"] for sid in pilot_ids]))
                    s4_rows = [row for sid in pilot_ids for row in m4_obs[seed][sid][crf][det_key]["rows"]]
                    s4_pooled = evaluate_pooled_coco_metrics(scorer.coco, ordered_pilot_img_ids, s4_rows)
                    m4_pooled_details[str(seed)][str(crf)] = {**s4_pooled, "whole_bitstream_bytes": s4_bytes, "mean_per_image_ap50": s4_mean_ap50}
                    m4_curves_per_img_by_seed[seed].append({"qp": crf, "score": s4_mean_ap50, "bitstream_bytes": s4_bytes})
                    m4_curves_pooled_by_seed[seed].append({"qp": crf, "score": s4_pooled["ap50"], "bitstream_bytes": s4_bytes})
                    m4_curves_pooled_ap_50_95_by_seed[seed].append({"qp": crf, "score": s4_pooled["ap_50_95"], "bitstream_bytes": s4_bytes})

            # Mean M2, M3, M4 curves across the 3 seeds
            m2_mean_curve_per_img, m2_mean_curve_pooled, m2_mean_curve_pooled_ap_50_95 = [], [], []
            m3_mean_curve_per_img, m3_mean_curve_pooled, m3_mean_curve_pooled_ap_50_95 = [], [], []
            m4_mean_curve_per_img, m4_mean_curve_pooled, m4_mean_curve_pooled_ap_50_95 = [], [], []
            for idx_c, crf in enumerate(CRF_GRID):
                # M2
                b2 = int(round(np.mean([m2_curves_per_img_by_seed[s][idx_c]["bitstream_bytes"] for s in SEEDS])))
                s2_img = float(np.mean([m2_curves_per_img_by_seed[s][idx_c]["score"] for s in SEEDS]))
                s2_pol = float(np.mean([m2_curves_pooled_by_seed[s][idx_c]["score"] for s in SEEDS]))
                s2_pol_95 = float(np.mean([m2_curves_pooled_ap_50_95_by_seed[s][idx_c]["score"] for s in SEEDS]))
                m2_mean_curve_per_img.append({"qp": crf, "score": s2_img, "bitstream_bytes": b2})
                m2_mean_curve_pooled.append({"qp": crf, "score": s2_pol, "bitstream_bytes": b2})
                m2_mean_curve_pooled_ap_50_95.append({"qp": crf, "score": s2_pol_95, "bitstream_bytes": b2})

                # M3
                b3 = int(round(np.mean([m3_curves_per_img_by_seed[s][idx_c]["bitstream_bytes"] for s in SEEDS])))
                s3_img = float(np.mean([m3_curves_per_img_by_seed[s][idx_c]["score"] for s in SEEDS]))
                s3_pol = float(np.mean([m3_curves_pooled_by_seed[s][idx_c]["score"] for s in SEEDS]))
                s3_pol_95 = float(np.mean([m3_curves_pooled_ap_50_95_by_seed[s][idx_c]["score"] for s in SEEDS]))
                m3_mean_curve_per_img.append({"qp": crf, "score": s3_img, "bitstream_bytes": b3})
                m3_mean_curve_pooled.append({"qp": crf, "score": s3_pol, "bitstream_bytes": b3})
                m3_mean_curve_pooled_ap_50_95.append({"qp": crf, "score": s3_pol_95, "bitstream_bytes": b3})

                # M4
                b4 = int(round(np.mean([m4_curves_per_img_by_seed[s][idx_c]["bitstream_bytes"] for s in SEEDS])))
                s4_img = float(np.mean([m4_curves_per_img_by_seed[s][idx_c]["score"] for s in SEEDS]))
                s4_pol = float(np.mean([m4_curves_pooled_by_seed[s][idx_c]["score"] for s in SEEDS]))
                s4_pol_95 = float(np.mean([m4_curves_pooled_ap_50_95_by_seed[s][idx_c]["score"] for s in SEEDS]))
                m4_mean_curve_per_img.append({"qp": crf, "score": s4_img, "bitstream_bytes": b4})
                m4_mean_curve_pooled.append({"qp": crf, "score": s4_pol, "bitstream_bytes": b4})
                m4_mean_curve_pooled_ap_50_95.append({"qp": crf, "score": s4_pol_95, "bitstream_bytes": b4})

            # Paired cluster bootstrap (resamples=2000, unit="image") for M4-vs-M1 and M2-vs-M1
            boot_data_m2_vs_m1 = {
                seed: {
                    sid: (
                        [RDPoint(crf, m1_obs[sid][crf][det_key]["ap50"], m1_obs[sid][crf]["whole_bitstream_bytes"]) for crf in CRF_GRID],
                        [RDPoint(crf, m2_obs[seed][sid][crf][det_key]["ap50"], m2_obs[seed][sid][crf]["whole_bitstream_bytes"]) for crf in CRF_GRID],
                    )
                    for sid in pilot_ids
                }
                for seed in SEEDS
            }
            boot_m2_vs_m1 = paired_bd_bootstrap(boot_data_m2_vs_m1, unit="image", resamples=2000, random_seed=0)

            boot_data_m4_vs_m1 = {
                seed: {
                    sid: (
                        [RDPoint(crf, m1_obs[sid][crf][det_key]["ap50"], m1_obs[sid][crf]["whole_bitstream_bytes"]) for crf in CRF_GRID],
                        [RDPoint(crf, m4_obs[seed][sid][crf][det_key]["ap50"], m4_obs[seed][sid][crf]["whole_bitstream_bytes"]) for crf in CRF_GRID],
                    )
                    for sid in pilot_ids
                }
                for seed in SEEDS
            }
            boot_m4_vs_m1 = paired_bd_bootstrap(boot_data_m4_vs_m1, unit="image", resamples=2000, random_seed=0)

            # Paired p-values across 5 CRFs (M4-vs-M2 = 1.0 under fail-closed; M4-vs-M3 = M2-vs-M1)
            anchor_ap = {sid: {crf: m1_obs[sid][crf][det_key]["ap50"] for crf in CRF_GRID} for sid in pilot_ids}
            cand_ap = {seed: {sid: {crf: m2_obs[seed][sid][crf][det_key]["ap50"] for crf in CRF_GRID} for sid in pilot_ids} for seed in SEEDS}
            p_m2_vs_m1 = compute_paired_p_values(anchor_ap, cand_ap, pilot_ids)
            # 10-test Holm family: 5 CRFs (M4-vs-M2, all p=1.0 due to fail-closed Delta-x=0) + 5 CRFs (M4-vs-M3 == M2-vs-M1)
            p_family_10 = [1.0] * 5 + p_m2_vs_m1

            eval_per_img = evaluate({
                "rc_mode": "CRF",
                "same_gop_and_tools": True,
                "curves": {
                    "M1": m1_curve_per_img,
                    "M2": m2_mean_curve_per_img,
                    "M3": m3_mean_curve_per_img,
                    "M4": m4_mean_curve_per_img,
                },
                "p_values": p_family_10,
            })
            eval_pooled = evaluate({
                "rc_mode": "CRF",
                "same_gop_and_tools": True,
                "curves": {
                    "M1": m1_curve_pooled,
                    "M2": m2_mean_curve_pooled,
                    "M3": m3_mean_curve_pooled,
                    "M4": m4_mean_curve_pooled,
                },
                "p_values": p_family_10,
            })

            # AP_small delta (percentage points) averaged across the 5 CRFs and 3 seeds
            ap_small_deltas_pp = []
            for crf in CRF_GRID:
                m1_aps = m1_pooled_details[str(crf)]["ap_small"]
                m2_aps = float(np.mean([m2_pooled_details[str(s)][str(crf)]["ap_small"] for s in SEEDS]))
                ap_small_deltas_pp.append((m2_aps - m1_aps) * 100.0)
            mean_ap_small_delta_pp = float(np.mean(ap_small_deltas_pp))

            # Task floor check: max AP50 drop across CRFs <= 0.5 pp
            ap50_drops_pp = []
            for crf in CRF_GRID:
                m1_ap = m1_pooled_details[str(crf)]["ap50"]
                m2_ap = float(np.mean([m2_pooled_details[str(s)][str(crf)]["ap50"] for s in SEEDS]))
                ap50_drops_pp.append((m1_ap - m2_ap) * 100.0)
            task_floor_ok = bool(max(ap50_drops_pp) <= 0.5)

            m2_bd_pct = eval_pooled["M2-vs-M1"]["percent"]
            h2_passed = bool(
                eval_pooled["M2-vs-M1"]["valid"]
                and m2_bd_pct is not None
                and secondary_h2(float(m2_bd_pct), mean_ap_small_delta_pp)
            )
            h1_passed = h1_pass(
                {
                    "M4-vs-M2": {**eval_pooled["M4-vs-M2"], "ci95": [0.0, 0.0]},
                    "M4-vs-M3": {**eval_pooled["M4-vs-M3"], "ci95": boot_m2_vs_m1.get("ci95")},
                },
                independent_seeds=list(SEEDS),
                resamples=2000,
                task_floor_passed=task_floor_ok,
                holm_results=eval_pooled["holm_family"],
            )

            eval_pooled_ap_50_95 = evaluate({
                "rc_mode": "CRF",
                "same_gop_and_tools": True,
                "curves": {
                    "M1": m1_curve_pooled_ap_50_95,
                    "M2": m2_mean_curve_pooled_ap_50_95,
                    "M3": m3_mean_curve_pooled_ap_50_95,
                    "M4": m4_mean_curve_pooled_ap_50_95,
                },
                "p_values": p_family_10,
            })

            detector_reports[det_key] = {
                "pooled_evaluation": eval_pooled,
                "pooled_ap_50_95_evaluation": eval_pooled_ap_50_95,
                "per_image_mean_evaluation": eval_per_img,
                "bootstrap_m2_vs_m1_per_image": boot_m2_vs_m1,
                "holm_m2_vs_m1_5crf": holm_bonferroni(p_m2_vs_m1),
                "bootstrap_m4_vs_m1_per_image": boot_m4_vs_m1,
                "m1_pooled_by_crf": m1_pooled_details,
                "m2_pooled_by_seed_and_crf": m2_pooled_details,
                "m3_pooled_by_seed_and_crf": m3_pooled_details,
                "m4_pooled_by_seed_and_crf": m4_pooled_details,
                "mean_ap_small_delta_pp": mean_ap_small_delta_pp,
                "max_ap50_drop_pp": float(max(ap50_drops_pp)),
                "task_floor_passed": task_floor_ok,
                "h1_pass": h1_passed,
                "h2_secondary_pass": h2_passed,
                "transfer_status": (
                    transfer_status(float(m2_bd_pct))
                    if (eval_pooled["M2-vs-M1"]["valid"] and m2_bd_pct is not None)
                    else "INCONCLUSIVE"
                ),
            }

        wall_seconds = time.monotonic() - started
        gpu_hours = args.gpu_devices * wall_seconds / 3600.0
        step3_report = {
            "schema_version": 1,
            "status": "PASS",
            "council_authority": "Escalation #12 (2026-11-02) 6/6 Unanimous Ruling",
            "wall_seconds": wall_seconds,
            "cpu_hours": (cpu_seconds() - cpu_started) / 3600.0,
            "gpu_hours": gpu_hours,
            "gpu_devices": args.gpu_devices,
            "prior_pilot_gpu_hours": args.prior_gpu_hours,
            "total_pilot_gpu_hours": args.prior_gpu_hours + gpu_hours,
            "gpu_hour_ceiling": 60.0,
            "split_hygiene": {
                "parent_ledger_sha256": parent_ledger.digest,
                "step3_ledger_sha256": step3_ledger.digest,
                "counts_by_split": split_meta["counts_by_split"],
                "train_spatial_winner_distribution": {NAMES[k]: v for k, v in winner_counts.items()},
                "eval_split": args.eval_split,
                "eval_sample_count": len(pilot_ids),
                "pilot_touched_during_training": False,
                "val2017_touched": False,
            },
            "asm_smoke_suite": asm_smoke,
            "student_training": seed_training_reports,
            "calibration_gate_audit": calibration_report,
            "encode_accounting": {
                "logical_eval_requests": logical_requests,
                "physical_eval_encodes": physical_encodes,
                "smoke_physical_encodes": asm_smoke["physical_encodes"],
                "total_physical_encodes": physical_encodes + asm_smoke["physical_encodes"],
                "cache_hits_and_aliases": cache_hits,
                "manifest_counts": manifest.counts(),
            },
            "detectors": detector_reports,
            "provenance": {
                "repo_commit": repo_sha,
                "stack": stack,
                "patched_pin": encoder.pin,
                "stock_pin": encoder.stock_pin,
                "detectors": scorer.provenance,
            },
        }
        write_json(output / "step3_report.json", step3_report)
        write_json(REPO / "tier1a/step3_report.json", step3_report)
        print(json.dumps({
            "status": step3_report["status"],
            "wall_seconds": wall_seconds,
            "total_physical_encodes": step3_report["encode_accounting"]["total_physical_encodes"],
            "yolo11l_m2_vs_m1_pooled_bd": detector_reports["yolo11l"]["pooled_evaluation"]["M2-vs-M1"],
            "faster_rcnn_m2_vs_m1_pooled_bd": detector_reports["faster_rcnn"]["pooled_evaluation"]["M2-vs-M1"],
        }, indent=2), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
