#!/usr/bin/env python3
"""Cache DINOv2-Base and SAM 2.1-Base only, against a frozen COCO ledger."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism.data.splits import SplitLedger
from tier1a.common import StepFailure, check_storage, exclusive, read_json, sha256, write_json
from tier1a.resources import ResourceBudget

REPO = Path(__file__).resolve().parents[1]


def cache_teachers(output: Path, weights: Path, ledger_path: Path, *, working: Path,
                   budget: ResourceBudget, offline: bool = False) -> dict:
    # Freeze and source contract are checked before importing model libraries.
    ledger = SplitLedger.read(ledger_path)
    selection_path = output / "coco500_selection.json"
    selection = read_json(selection_path)
    rows = selection["images"]
    if (not ledger.frozen or len(rows) != 500 or len(ledger.entries) != 500
            or selection["source"] != "train2017"
            or {r["sample_id"] for r in rows} != set(ledger.by_id)):
        raise StepFailure("teacher_ledger", "Require frozen 500-image train2017 selection and matching ledger")
    for row in rows:
        entry = ledger.require(row["sample_id"], "train")
        if entry.source != "train2017":
            raise StepFailure("teacher_ledger", "Only train2017 is permitted")
    import torch
    from prism.data.cached_dataset import CachedDataset
    from scripts.cache_teacher_targets import extract
    from tier1a.teachers import RECIPE, TeacherAdapter

    torch.manual_seed(0)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not budget.devices:
            raise StepFailure("budget", "CUDA available but no allocated GPU was accounted")
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    teachers = output / "teachers"
    adapter = TeacherAdapter(teachers, weights, rows, ledger.digest, sha256(selection_path),
                             device=device, offline=offline)
    index_path = teachers / "cache/index.json"
    completed = {r["sample_id"]: r for r in read_json(index_path)["samples"]} if index_path.exists() else {}
    pending = sum(any(not (teachers / name / f"{r['image_id']}.{suffix}").exists()
                      for name in ("dinov2", "sam21") for suffix in ("json", "npz"))
                  or r["sample_id"] not in completed
                  or not (teachers / "cache" / completed[r["sample_id"]]["shard"]).exists() for r in rows)
    # Planning assumption, not measured T4 throughput. Updated after each completed image.
    initial_seconds = 15.0
    budget.check(pending, initial_seconds, setup_seconds=300 if pending else 0)
    check_storage(working, 2_500_000_000 if pending else 10_000_000)
    seen, measured_seconds, measured_images = 0, 0.0, 0
    steady_seconds, steady_images = 0.0, 0
    sample_started = time.monotonic()

    def before_sample(sample_id: str) -> None:
        nonlocal sample_started
        sample_started = time.monotonic()
        check_storage(working, 12_000_000)
        rate = max(initial_seconds, steady_seconds / steady_images) if steady_images else initial_seconds
        budget.check(pending, rate)

    def after_sample(sample_id: str, created: bool) -> None:
        nonlocal seen, measured_seconds, measured_images, pending, steady_seconds, steady_images
        if device == "cuda":
            torch.cuda.synchronize()
        seen += 1
        if created:
            duration = time.monotonic() - sample_started
            measured_seconds += duration
            # First image includes weight downloads/model setup. Charge wall time,
            # but do not multiply that one-time cost across all remaining images.
            if measured_images:
                steady_seconds += duration
                steady_images += 1
            measured_images += 1
            pending = max(0, pending - 1)
        if seen % 10 == 0 or seen == 500:
            print(json.dumps({"step": "teachers", "completed": seen, "total": 500,
                              "device": device, **adapter.stats}), flush=True)
        rate = max(initial_seconds, steady_seconds / steady_images) if steady_images else initial_seconds
        budget.check(pending, rate)

    sources = {row["sample_id"]: str(output / "coco500/images" / row["file_name"]) for row in rows}
    stats = extract(ledger, sources, teachers / "cache", adapter, split="train", teacher_size=518,
                    center_crop=True, fp16=True, resume=True,
                    before_sample=before_sample, after_sample=after_sample)
    dataset = CachedDataset(teachers / "cache/index.json", ledger, split="train")
    mask_count = empty_mask_images = 0
    try:
        if len(dataset) != 500:
            raise StepFailure("teacher_cache", "CachedDataset must expose exactly 500 training samples")
        for index in range(len(dataset)):
            sample = dataset[index]
            if sample["x_low"].dtype.name != "float16" or sample["dino_features"].dtype.name != "float16":
                raise StepFailure("teacher_cache", "Expected fp16 floating cache tensors")
            count = sample["sam_masks"].shape[0]
            if count > 32 or sample["sam_masks"].dtype.name != "bool":
                raise StepFailure("teacher_cache", "Require <=32 bool SAM masks per image")
            mask_count += count
            empty_mask_images += int(count == 0)
    finally:
        dataset.close()
    report = {"status": "PASS", "device": device, "ledger_sha256": ledger.digest,
              "recipe": RECIPE, "samples": stats, "per_teacher": adapter.stats,
              "measured_images": measured_images, "measured_extraction_seconds": measured_seconds,
              "steady_state_images": steady_images, "steady_state_seconds": steady_seconds,
              "mask_count": mask_count, "empty_mask_images": empty_mask_images,
              "projection": budget.data.get("projection"),
              "teacher_cache_is_not_reconstruction_evaluation": True}
    write_json(teachers / "cache_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/tier1a"))
    parser.add_argument("--weights", type=Path, default=Path("/kaggle/working/cache/weights"))
    parser.add_argument("--ledger", type=Path, default=REPO / "prism/data/splits/coco500_ledger.json")
    parser.add_argument("--working", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--prior-gpu-hours", type=float, default=0)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    budget = None
    locks = ExitStack()
    status = "FAIL"
    try:
        import torch
        locks.enter_context(exclusive(args.output / "teachers/.writer.lock"))
        budget = ResourceBudget(args.output / "teachers/gpu_hours.json",
                                devices=torch.cuda.device_count(), prior_gpu_hours=args.prior_gpu_hours)
        budget.arm(args.output / "teachers/cache_report.json")
        cache_teachers(args.output, args.weights, args.ledger, working=args.working,
                       budget=budget, offline=args.offline)
        status = "PASS"
        return 0
    except Exception as exc:
        report = {"status": "FAIL", "step": getattr(exc, "step", "teachers"), "reason": str(exc)}
        write_json(args.output / "teachers/cache_report.json", report)
        print(json.dumps(report), file=sys.stderr)
        return 1
    finally:
        if budget:
            budget.finish(status)
        locks.close()


if __name__ == "__main__":
    raise SystemExit(main())
