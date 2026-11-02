#!/usr/bin/env python3
"""Evaluate a deterministic COCO val2017 shard with three independent students.

M1 is shared across seeds; M2/M3/M4 retain separate observations per seed.
Completed images are checkpointed atomically. Only one image's decoded frames
and predictions are held in memory, and temporary codec files are discarded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tier1a.common import exclusive, read_json, sha256, write_json

CRFS = (20, 24, 28, 32, 36)
SEEDS = (0, 1, 2)
MODELS = ("M1", "M2", "M3", "M4")
DETECTORS = ("yolo11l", "faster_rcnn")
SCHEMA_VERSION = 1


def shard_images(all_images: list[dict], shard_idx: int, num_shards: int = 19) -> list[dict]:
    """Sort annotation image entries by ID, then take a strided partition."""
    if num_shards <= 0 or not 0 <= shard_idx < num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_idx < num_shards")
    ordered = sorted(all_images, key=lambda row: int(row["id"]))
    if len({int(row["id"]) for row in ordered}) != len(ordered):
        raise ValueError("Duplicate image IDs in annotations")
    return ordered[shard_idx::num_shards]


def validate_crfs(crfs: list[int] | tuple[int, ...]) -> list[int]:
    if not crfs or len(set(crfs)) != len(crfs) or any(not 0 <= c <= 51 for c in crfs):
        raise ValueError("CRFs must be distinct integers in [0, 51]")
    return sorted(crfs)


def observation_keys(image_ids, crfs):
    return {(int(i), model, seed, int(crf)) for i in image_ids for crf in crfs
            for model in MODELS for seed in ([None] if model == "M1" else SEEDS)}


def validate_observations(observations: list[dict], image_ids, crfs, category_ids) -> None:
    """Reject missing/duplicate cells and malformed detections before pooling."""
    expected = observation_keys(image_ids, crfs)
    seen = set()
    for row in observations:
        key = (row["image_id"], row["model"], row["seed"], row["crf"])
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate observation: {key}")
        seen.add(key)
        size = row["whole_bitstream_bytes"]
        if not isinstance(size, int) or size <= 0 or row["bits"] != size * 8:
            raise ValueError(f"Invalid whole-bitstream size: {key}")
        for detector in DETECTORS:
            detection = row[detector]
            ap50 = detection["ap50"]
            if ap50 is not None and (not math.isfinite(ap50) or not 0 <= ap50 <= 1):
                raise ValueError(f"Invalid per-image AP50: {key}")
            if not isinstance(detection["rows"], list):
                raise ValueError(f"Missing detection rows: {key}")
            for prediction in detection["rows"]:
                bbox = prediction["bbox"]
                score = prediction["score"]
                if (prediction["image_id"] != row["image_id"]
                        or prediction["category_id"] not in category_ids
                        or len(bbox) != 4 or not all(math.isfinite(v) for v in bbox)
                        or min(bbox) < 0 or min(bbox[2:]) <= 0
                        or not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError(f"Invalid detection row: {key}")
    if seen != expected:
        raise ValueError(f"Missing {len(expected - seen)} observations")


def load_students(checkpoints_dir: Path, device: str):
    import torch
    from prism.models.student import MobileNetV4HybridController

    models, provenance = {}, {}
    for seed in SEEDS:
        path = checkpoints_dir / f"student_seed_{seed}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("seed") != seed:
            raise ValueError(f"Checkpoint seed mismatch: {path}")
        backbone = {"mobilenetv4_conv_small": "timm", "small_conv_smoke_only": "small"}.get(
            checkpoint.get("backbone_name"))
        if backbone is None:
            raise ValueError(f"Unknown checkpoint backbone: {path}")
        model = MobileNetV4HybridController(backbone=backbone, allow_fallback=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        models[seed] = model.to(device).eval()
        provenance[str(seed)] = {"sha256": sha256(path), "backbone": model.backbone_name}
    return models, provenance


def infer_controls(model, x_low, crf: int, device: str):
    """Produce both controls in one forward pass, matching Step 3 inference."""
    import torch
    from torch.nn import functional as F
    from prism.models.student import upsample_qp_map
    from tier1a.run_tier1a_step3 import project_zero_mean_qp_map, qg_area_weights

    with torch.inference_mode():
        out = model(x_low.unsqueeze(0).to(device),
                    torch.full((1, 1), crf / 51.0, device=device), for_encoding=False)
        qg = upsample_qp_map(out["delta_qp"])[0, 0].cpu().numpy()
        residual = F.interpolate(out["residual"], size=(1080, 1920), mode="bilinear",
                                 align_corners=False)[0, 0].cpu().numpy()
    return project_zero_mean_qp_map(qg, weights=qg_area_weights()), residual


def evaluate_image(frame, payload, models, encoder, scorer, crfs, directory, device):
    from tier1a.oracle_codec import yuv_to_rgb
    from tier1a.run_tier1a_step3 import apply_residual_to_yuv, rgb_to_student_input

    x_low = rgb_to_student_input(yuv_to_rgb(payload, 1920, 1080))
    observations, encoded_cache, scored_cache = [], {}, {}

    def observe(model, seed, crf, source, offsets, stock):
        input_sha = hashlib.sha256(source).hexdigest()
        map_sha = None if offsets is None else hashlib.sha256(offsets.tobytes()).hexdigest()
        key = (input_sha, map_sha, stock, crf)
        if key not in encoded_cache:
            with tempfile.TemporaryDirectory(dir=directory, prefix="encode_") as scratch:
                result = encoder(source, offsets, frame, Path(scratch), stock=stock, crf=crf)
                digest = result["bitstream_sha256"]
                if digest not in scored_cache:
                    scored_cache[digest] = scorer.score_both(
                        encoder.decode(result, frame), frame, allow_undefined_ap50=True)
                encoded_cache[key] = {
                    name: result[name] for name in ("whole_bitstream_bytes", "bits",
                                                   "bitstream_sha256", "encode_seconds")}
                encoded_cache[key].update(scored_cache[digest])
        observations.append({"image_id": frame["image_id"], "model": model, "seed": seed,
                             "crf": crf, "input_sha256": input_sha, "map_sha256": map_sha,
                             "stock": stock, **encoded_cache[key]})

    for crf in crfs:
        observe("M1", None, crf, payload, None, True)
        for seed in SEEDS:
            qmap, residual = infer_controls(models[seed], x_low, crf, device)
            preprocessed = apply_residual_to_yuv(payload, residual, 1920, 1080)
            observe("M2", seed, crf, payload, qmap, False)
            observe("M3", seed, crf, preprocessed, None, True)
            observe("M4", seed, crf, preprocessed, qmap, False)
    return {"image_id": frame["image_id"], "geometry": frame["geometry"],
            "observations": observations, "physical_encodes": len(encoded_cache),
            "scored_frames": len(scored_cache)}


def write_shard_report(path: Path, metadata: dict, image_paths: list[Path]) -> None:
    """Stream the final report from atomic per-image checkpoints."""
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, allow_nan=False)[:-1] + ', "observations": [')
        first = True
        for image_path in image_paths:
            for row in read_json(image_path)["observations"]:
                if not first:
                    stream.write(",\n")
                json.dump(row, stream, allow_nan=False, separators=(",", ":"))
                first = False
        stream.write("]}\n")
    temporary.replace(path)


def run(args) -> Path:
    import torch
    import numpy as np
    from PIL import Image
    from tier1a.oracle_codec import Gate0Encoder
    from tier1a.run_oracle_sweep import ensure_stack
    from tier1a.step3_detectors import DualDetectorScorer
    from tier1a.yuv import letterbox_yuv

    annotations = read_json(args.annotations)
    all_images = annotations["images"]
    if not getattr(args, "allow_partial", False) and len(all_images) != 5000:
        raise ValueError(f"Expected full COCO val2017 (5000 images), got {len(all_images)}")
    selected = shard_images(all_images, args.shard_idx, args.num_shards)
    crfs = validate_crfs(args.crfs)
    categories = {row["id"] for row in annotations["categories"]}
    for row in selected:
        path = args.images_dir / row["file_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in [args.weights, args.frcnn_weights]:
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    models, checkpoints = load_students(args.checkpoints_dir, device)
    ensure_stack(args.gate, build=False, deadline=float("inf"), require_asm=True)
    encoder = Gate0Encoder(args.gate, deadline=float("inf"))
    scorer = DualDetectorScorer(args.annotations, args.weights, {row["id"] for row in selected},
                                device=device, frcnn_weights=args.frcnn_weights)
    revision = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    # Exclude host paths and binary hashes (independent builds), retain source/config identity.
    codec = {key: encoder.stack[key] for key in
             ("x265_commit", "stock_commit", "patch_sha256", "telemetry_patch_sha256", "enable_assembly")}
    codec["parameters"] = encoder.config.options(Path("frame.csv"))
    detectors = json.loads(json.dumps(scorer.provenance))
    for detector in detectors.values():
        detector.pop("device", None)
    metadata = {"schema_version": SCHEMA_VERSION, "shard_idx": args.shard_idx,
                "num_shards": args.num_shards, "dataset_image_count": len(all_images),
                "image_ids": [row["id"] for row in selected], "crfs": crfs, "seeds": list(SEEDS),
                "annotations_sha256": sha256(args.annotations), "repo_commit": revision,
                "checkpoints": checkpoints, "codec": codec, "detectors": detectors,
                "preprocessing": "tier1a.yuv.letterbox_yuv:1920x1080:yuv420p8:bt709:limited",
                "undefined_per_image_ap50": "null; detections retained in pooled COCOeval"}
    progress = args.output / f"shard_{args.shard_idx}_progress"
    report_path = args.output / f"shard_{args.shard_idx}_report.json"
    with exclusive(args.output / f"shard_{args.shard_idx}.lock"):
        progress.mkdir(parents=True, exist_ok=True)
        receipt = progress / "settings.json"
        if receipt.exists() and read_json(receipt) != metadata:
            raise ValueError("Resume settings changed; use a new output directory")
        write_json(receipt, metadata)
        image_paths, image_info = [], []
        started = time.monotonic()
        for index, row in enumerate(selected, 1):
            image_path = progress / f"image_{row['id']}.json"
            if image_path.exists():
                result = read_json(image_path)
            else:
                with Image.open(args.images_dir / row["file_name"]) as image:
                    if image.size != (row["width"], row["height"]):
                        raise ValueError(f"Source dimensions disagree with COCO: {row['id']}")
                    payload, geometry = letterbox_yuv(np.asarray(image.convert("RGB")))
                frame = {"image_id": row["id"], "sample_id": str(row["id"]), "geometry": geometry}
                with tempfile.TemporaryDirectory(prefix="prism_tier1b_") as scratch:
                    result = evaluate_image(frame, payload, models, encoder, scorer, crfs,
                                            Path(scratch), device)
                write_json(image_path, result)
            validate_observations(result["observations"], [row["id"]], crfs, categories)
            image_paths.append(image_path)
            image_info.append({key: value for key, value in result.items() if key != "observations"})
            print(f"[shard {args.shard_idx}] {index}/{len(selected)} images; "
                  f"elapsed={time.monotonic() - started:.1f}s", flush=True)
        write_shard_report(report_path, {**metadata, "status": "complete", "images": image_info,
                           "observation_count": len(selected) * len(crfs) * 10}, image_paths)
    return report_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images-dir", "annotations", "checkpoints-dir", "output", "gate", "weights"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--frcnn-weights", type=Path)
    parser.add_argument("--shard-idx", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=19)
    parser.add_argument("--crfs", nargs="+", type=int, default=list(CRFS))
    parser.add_argument("--allow-partial", action="store_true", help="Allow running on partial dataset (<5000 images) for testing")
    args = parser.parse_args(argv)
    try:
        shard_images([], args.shard_idx, args.num_shards)
        validate_crfs(args.crfs)
    except ValueError as exc:
        parser.error(str(exc))
    return args


if __name__ == "__main__":
    print(run(parse_args()))
