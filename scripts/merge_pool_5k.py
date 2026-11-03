#!/usr/bin/env python3
"""Merge and evaluate Tier 1b sharded evaluation results across the 19 Kaggle accounts.

Aggregates all 19 shard reports (5,000 images total), verifies complete coverage
without omissions or duplicates, executes standard COCOeval mAP / AP50 evaluation,
and calculates MPEG-VCM Bjontegaard and PAVA-PCHIP BD-rates.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
import gc
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
import warnings

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from prism.evaluation.metrics import RDPoint, bjontegaard_bd_rate, pava_pchip_bd_rate
from tier1a.common import read_json, sha256, write_json
from tier1a.step3_detectors import evaluate_pooled_coco_metrics

DEFAULT_RESULTS_DIR = Path("C:/Users/Wagur1/Desktop/HOI_DONG_9/results_tier1b")
CRF_GRID = (20, 24, 28, 32, 36)
SEEDS = (0, 1, 2)
MODELS = ("M1", "M2", "M3", "M4")
DETECTORS = ("yolo11l", "faster_rcnn")


def find_shard_reports(results_dir: Path) -> dict[int, Path]:
    """Locate reports, rejecting conflicting copies of the same shard."""
    found = {}
    for p in results_dir.glob("**/shard_*_report.json"):
        stem = p.name.replace(".json", "")
        parts = stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            shard_idx = int(parts[1])
            if shard_idx in found:
                if sha256(found[shard_idx]) != sha256(p):
                    raise ValueError(f"Conflicting reports for shard {shard_idx}")
                canonical = results_dir / f"shard_{shard_idx}" / p.name
                if p == canonical:
                    found[shard_idx] = p
            else:
                found[shard_idx] = p
    return dict(sorted(found.items()))


def download_shard_outputs(pool_dir: Path, results_dir: Path, num_shards: int = 19) -> dict[int, Path]:
    """Download outputs from each Kaggle account in the pool."""
    from scripts.dispatch_pool_5k import discover_pool_accounts, get_account_for_shard
    accounts = discover_pool_accounts(pool_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}

    for s_idx in range(num_shards):
        acc = get_account_for_shard(s_idx, accounts)
        kernel_id = f"{acc['username']}/prism-tier1b-shard{s_idx}"
        target_dir = results_dir / f"shard_{s_idx}"
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{acc['username']}] Downloading output for {kernel_id} to {target_dir}...", flush=True)

        env = os.environ.copy()
        env["KAGGLE_API_TOKEN"] = str(acc["key"])
        cmd = [sys.executable, "-m", "kaggle", "kernels", "output", kernel_id, "-p", str(target_dir)]
        try:
            subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
            report_file = target_dir / f"shard_{s_idx}_report.json"
            if report_file.exists():
                print(f"  [OK] Downloaded {report_file.name} ({report_file.stat().st_size} bytes)")
                downloaded[s_idx] = report_file
            else:
                print(f"  [Warning] Output downloaded but {report_file.name} not found.")
        except subprocess.CalledProcessError as exc:
            print(f"  [Error] Failed to download {kernel_id}: {exc.stderr.strip()}", file=sys.stderr)
    return downloaded


def compute_bd_rates(curves: dict, available_seeds: list[int]) -> dict:
    """Calculate BD-Rates (MPEG Bjontegaard and PAVA-PCHIP) against M1 baseline."""
    results_bd = {}
    for det in DETECTORS:
        results_bd[det] = {}
        m1_pts = curves.get((det, "M1", None), [])
        if len(m1_pts) < 4:
            continue
        for cand_model in ("M2", "M3", "M4"):
            results_bd[det][cand_model] = {}
            seed_bd_map_bjontegaard = []
            seed_bd_map_pava = []
            seed_bd_ap50_bjontegaard = []
            seed_bd_ap50_pava = []

            for seed in available_seeds:
                cand_pts = curves.get((det, cand_model, seed), [])
                if len(cand_pts) < 4:
                    continue
                def evaluate_bd(function, metric):
                    anchor = [RDPoint(p["crf"], p[metric], p["bytes"]) for p in m1_pts]
                    candidate = [RDPoint(p["crf"], p[metric], p["bytes"]) for p in cand_pts]
                    if any(not 0 <= p.score <= 1 for p in anchor + candidate):
                        return None
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error")
                            result = function(anchor, candidate)
                        value = result.get("bd_rate")
                        return float(value) if result.get("valid") and value is not None and np.isfinite(value) else None
                    except (ValueError, ArithmeticError, Warning, np.linalg.LinAlgError):
                        return None

                bd_m_bj = evaluate_bd(bjontegaard_bd_rate, "ap_50_95")
                bd_50_bj = evaluate_bd(bjontegaard_bd_rate, "ap50")
                bd_m_pava = evaluate_bd(pava_pchip_bd_rate, "ap_50_95")
                bd_50_pava = evaluate_bd(pava_pchip_bd_rate, "ap50")

                seed_entry = {
                    "seed": seed,
                    "bjontegaard_bd_rate_map": bd_m_bj,
                    "bjontegaard_bd_rate_ap50": bd_50_bj,
                    "pava_pchip_bd_rate_map": bd_m_pava,
                    "pava_pchip_bd_rate_ap50": bd_50_pava,
                }
                results_bd[det][cand_model][f"seed_{seed}"] = seed_entry
                if bd_m_bj is not None: seed_bd_map_bjontegaard.append(bd_m_bj)
                if bd_m_pava is not None: seed_bd_map_pava.append(bd_m_pava)
                if bd_50_bj is not None: seed_bd_ap50_bjontegaard.append(bd_50_bj)
                if bd_50_pava is not None: seed_bd_ap50_pava.append(bd_50_pava)

            def seed_mean(values):
                return float(np.mean(values)) if len(values) == len(available_seeds) else None

            results_bd[det][cand_model]["mean"] = {
                "bjontegaard_bd_rate_map": seed_mean(seed_bd_map_bjontegaard),
                "bjontegaard_bd_rate_ap50": seed_mean(seed_bd_ap50_bjontegaard),
                "pava_pchip_bd_rate_map": seed_mean(seed_bd_map_pava),
                "pava_pchip_bd_rate_ap50": seed_mean(seed_bd_ap50_pava),
            }
    return results_bd


def _evaluate_config_worker(task):
    det, model, seed, crf, b, chunk_path_str, annotations_str, all_image_ids = task
    import pickle
    import time
    from pathlib import Path
    from pycocotools.coco import COCO
    from tier1a.step3_detectors import evaluate_pooled_coco_metrics

    chunk_path = Path(chunk_path_str)
    rows = []
    if chunk_path.is_file():
        with open(chunk_path, "rb") as f:
            while True:
                try:
                    rows.extend(pickle.load(f))
                except EOFError:
                    break
        chunk_path.unlink(missing_ok=True)
    t0 = time.monotonic()
    coco = COCO(annotations_str)
    m = evaluate_pooled_coco_metrics(coco, all_image_ids, rows)
    dt = time.monotonic() - t0
    return (det, model, seed, crf, b, m, dt)


def stream_shards_and_evaluate(
    shard_reports: dict[int, Path],
    expected_num_shards: int,
    coco,
    results_dir: Path,
    annotations_path: Path | None = None,
    validate_shards: bool = False,
    max_workers: int = 8,
) -> tuple[dict, dict]:
    """Stream shards sequentially, storing detection chunks on disk, then evaluate in parallel."""
    from tier1b.run_tier1b_val5k import validate_observations

    if set(shard_reports) != set(range(expected_num_shards)):
        raise ValueError(f"Expected exactly shards 0..{expected_num_shards - 1}")

    seen_image_ids = set()
    metadata_list = []
    total_observations = 0
    total_bytes = defaultdict(int)
    available_seeds = set()
    category_ids = set(coco.getCatIds()) if coco else None

    scratch_dir = tempfile.TemporaryDirectory(dir=results_dir, prefix="pool_eval_scratch_")
    scratch_path = Path(scratch_dir.name)

    try:
        print(f"Streaming and aggregating {len(shard_reports)} shard reports sequentially (disk scratch: {scratch_path})...", flush=True)
        t0 = time.monotonic()
        for s_idx in sorted(shard_reports.keys()):
            path = shard_reports[s_idx]
            data = read_json(path)
            if data.get("shard_idx", s_idx) != s_idx:
                raise ValueError(f"Shard index mismatch in {path}")
            shard_img_ids = data.get("image_ids", [])
            shard_obs = data.get("observations", [])
            shard_ids_set = set(shard_img_ids)
            if not shard_ids_set or len(shard_ids_set) != len(shard_img_ids):
                raise ValueError(f"Shard {s_idx} has missing or duplicate image IDs")
            overlap = seen_image_ids.intersection(shard_ids_set)
            if overlap:
                raise ValueError(f"Shard {s_idx} contains overlapping images already seen: {overlap}")
            seen_image_ids.update(shard_ids_set)
            total_observations += len(shard_obs)

            if validate_shards and category_ids is not None:
                validate_observations(shard_obs, shard_ids_set, CRF_GRID, category_ids)

            shard_chunks = defaultdict(list)
            for obs in shard_obs:
                model = obs["model"]
                seed = obs.get("seed")
                crf = int(obs["crf"])
                if seed is not None:
                    available_seeds.add(seed)
                model_key = (model, seed, crf)
                total_bytes[model_key] += obs["whole_bitstream_bytes"]
                for det in DETECTORS:
                    shard_chunks[(det, model, seed, crf)].extend(obs[det]["rows"])

            for key, rows in shard_chunks.items():
                det, model, seed, crf = key
                chunk_file = scratch_path / f"{det}_{model}_{seed}_{crf}.pkl"
                with open(chunk_file, "ab") as f:
                    pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)

            metadata_list.append({
                "shard_idx": s_idx,
                "image_count": len(shard_img_ids),
                "observation_count": len(shard_obs),
                "path": str(path),
            })
            del data, shard_obs, shard_img_ids, shard_ids_set, shard_chunks
            gc.collect()
            print(f"  [Shard {s_idx:02d}/18] aggregated ({len(seen_image_ids)} images total, elapsed {time.monotonic() - t0:.1f}s)", flush=True)

        summary = {
            "shards_found": len(shard_reports),
            "expected_shards": expected_num_shards,
            "total_unique_images": len(seen_image_ids),
            "total_observations": total_observations,
            "shards": metadata_list,
        }

        if len(seen_image_ids) != 5000:
            raise ValueError(f"Expected 5,000 images, found {len(seen_image_ids)}")
        if coco and set(seen_image_ids) != set(coco.getImgIds()):
            raise ValueError("Shard coverage differs from annotation image IDs")

        if not coco or not annotations_path:
            print("Note: Annotations not provided; skipping COCO evaluation.")
            return summary, {}

        seeds_list = sorted(available_seeds) if available_seeds else [0]
        all_image_ids = sorted(seen_image_ids)
        curves = defaultdict(list)
        detailed_metrics = defaultdict(dict)

        tasks = []
        for det in DETECTORS:
            for crf in CRF_GRID:
                key = ("M1", None, crf)
                b = total_bytes[key]
                chunk_file = scratch_path / f"{det}_M1_None_{crf}.pkl"
                tasks.append((det, "M1", None, crf, b, str(chunk_file), str(annotations_path), all_image_ids))

            for model in ("M2", "M3", "M4"):
                for seed in seeds_list:
                    for crf in CRF_GRID:
                        key = (model, seed, crf)
                        b = total_bytes[key]
                        chunk_file = scratch_path / f"{det}_{model}_{seed}_{crf}.pkl"
                        tasks.append((det, model, seed, crf, b, str(chunk_file), str(annotations_path), all_image_ids))

        total_evals = len(tasks)
        num_workers = min(max_workers, os.cpu_count() or 4)
        print(f"\nExecuting {total_evals} pooled COCOeval evaluations across 5,000 images using {num_workers} parallel workers...", flush=True)
        t_eval_start = time.monotonic()
        completed = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {executor.submit(_evaluate_config_worker, t): t for t in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                completed += 1
                det, model, seed, crf, b, m, dt = future.result()
                curves[(det, model, seed)].append({
                    "crf": crf,
                    "bytes": b,
                    "ap_50_95": m["ap_50_95"],
                    "ap50": m["ap50"],
                })
                metric_key = f"M1_crf{crf}" if model == "M1" else f"{model}_seed{seed}_crf{crf}"
                detailed_metrics[det][metric_key] = {**m, "bytes": b}
                seed_str = "" if seed is None else f"seed={seed} "
                print(f"  [{completed:03d}/{total_evals}] {det:11s} {model:2s} {seed_str:7s}CRF={crf} -> mAP={m['ap_50_95']:.4f}, AP50={m['ap50']:.4f} ({dt:.1f}s)", flush=True)

        for k in curves:
            curves[k].sort(key=lambda p: p["crf"])

        print(f"COCO evaluations complete in {time.monotonic() - t_eval_start:.1f}s.", flush=True)
        bd_rates = compute_bd_rates(curves, seeds_list)
        results = {"metrics": detailed_metrics, "bd_rates": bd_rates}
        return summary, results
    finally:
        scratch_dir.cleanup()


def print_summary_table(results: dict, summary: dict):
    """Print clean terminal summary tables for paper presentation."""
    print("\n" + "=" * 80)
    print(f" PRISM-VCM TIER 1b (5,000 IMAGES) EMPIRICAL VERIFICATION REPORT")
    print("=" * 80)
    print(f"Shards Aggregated: {summary['shards_found']}/{summary['expected_shards']}")
    print(f"Total Unique Images Evaluated: {summary['total_unique_images']:,} images")
    print(f"Total Observations: {summary['total_observations']:,}")
    print("-" * 80)

    for det in DETECTORS:
        det_name = "YOLOv11-L (Target Detector)" if det == "yolo11l" else "Faster R-CNN ResNet50-FPN v2 (Cross-Detector Transfer)"
        print(f"\n[Detector: {det_name}]")
        print(f"{'Candidate vs M1 (Stock)':<25} | {'MPEG Bjontegaard BD-Rate':<25} | {'PAVA-PCHIP BD-Rate':<25}")
        print(f"{'':<25} | {'mAP (50-95)':<12} {'AP50':<11} | {'mAP (50-95)':<12} {'AP50':<11}")
        print("-" * 80)
        bd = results.get("bd_rates", {}).get(det, {})
        for cand in ("M2", "M3", "M4"):
            m = bd.get(cand, {}).get("mean", {})
            bj_map = f"{m.get('bjontegaard_bd_rate_map'):.2f}%" if m.get('bjontegaard_bd_rate_map') is not None else "N/A"
            bj_50 = f"{m.get('bjontegaard_bd_rate_ap50'):.2f}%" if m.get('bjontegaard_bd_rate_ap50') is not None else "N/A"
            pava_map = f"{m.get('pava_pchip_bd_rate_map'):.2f}%" if m.get('pava_pchip_bd_rate_map') is not None else "N/A"
            pava_50 = f"{m.get('pava_pchip_bd_rate_ap50'):.2f}%" if m.get('pava_pchip_bd_rate_ap50') is not None else "N/A"
            label = {"M2": "M2 (Spatial QP only)", "M3": "M3 (Delta_x only)", "M4": "M4 (PRISM-VCM Coupled)"}.get(cand, cand)
            print(f"{label:<25} | {bj_map:<12} {bj_50:<11} | {pava_map:<12} {pava_50:<11}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory containing shard report JSONs")
    parser.add_argument("--pool-dir", type=Path, default=Path.home() / ".kaggle_pool", help="Kaggle pool directory")
    parser.add_argument("--download", action="store_true", help="Download reports from Kaggle pool before merging")
    parser.add_argument("--annotations", type=Path, help="Path to instances_val2017.json for COCOeval")
    parser.add_argument("--output", type=Path, help="Path to save merged report JSON")
    parser.add_argument("--num-shards", type=int, default=19, help="Expected number of shards (default: 19)")
    parser.add_argument("--validate-shards", action="store_true", help="Re-validate per-shard observations during aggregation")
    parser.add_argument("--max-workers", type=int, default=8, help="Number of parallel workers for COCOeval (default: 8)")
    args = parser.parse_args()

    if args.download:
        print(f"Downloading shard outputs to {args.results_dir}...")
        download_shard_outputs(args.pool_dir, args.results_dir, args.num_shards)

    shard_reports = find_shard_reports(args.results_dir)
    if not shard_reports:
        print(f"No shard reports found in {args.results_dir}!", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(shard_reports)} shard reports in {args.results_dir}")

    coco = None
    if args.annotations and args.annotations.is_file():
        from pycocotools.coco import COCO
        print(f"Loading COCO annotations from {args.annotations}...")
        coco = COCO(str(args.annotations))

    summary, results = stream_shards_and_evaluate(
        shard_reports, args.num_shards, coco, args.results_dir,
        annotations_path=args.annotations, validate_shards=args.validate_shards,
        max_workers=args.max_workers
    )
    if results:
        print_summary_table(results, summary)

    output_path = args.output if args.output else (args.results_dir / "tier1b_val5k_final_report.json")
    final_data = {
        "summary": summary,
        "results": results,
    }
    write_json(output_path, final_data)
    print(f"Final merged report written to {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
