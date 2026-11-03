#!/usr/bin/env python3
"""Poll the 19 Tier 1b Kaggle shards, validate outputs, then merge and plot.

Example:
    python scripts/auto_harvester.py --annotations C:/data/instances_val2017.json

Use --once for one polling pass (including finalization if all shards are ready).
Kaggle runs in credential-isolated subprocesses; keys never enter receipts/logs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.dispatch_pool_5k import discover_pool_accounts, get_account_for_shard, NUM_SHARDS
from tier1a.common import exclusive, read_json, sha256, write_json
from tier1b.run_tier1b_val5k import CRFS, SEEDS, validate_observations

DEFAULT_RESULTS_DIR = Path("C:/Users/Wagur1/Desktop/HOI_DONG_9/results_tier1b")
FINAL_REPORT = "tier1b_val5k_final_report.json"
ARTIFACT_NAMES = (
    "rd_curves.pdf", "rd_curves.png", "bd_rate_comparison.pdf",
    "bd_rate_comparison.png", "table_m1_m4_coco.tex", "table_cross_detector.tex",
)


def parse_kernel_status(value) -> str:
    """Accept Kaggle enums, response objects/dicts, JSON, and CLI status lines.

    Only an explicit status token is accepted: an error message mentioning a
    previous COMPLETE run must never make the current shard eligible to merge.
    """
    if isinstance(value, dict):
        return parse_kernel_status(value.get("status"))
    if not isinstance(value, str):
        if hasattr(value, "status"):
            return parse_kernel_status(value.status)
        return parse_kernel_status(value.name) if hasattr(value, "name") else "UNKNOWN"
    value = value.strip()
    if value.startswith("{"):
        try:
            return parse_kernel_status(json.loads(value))
        except ValueError:
            return "UNKNOWN"
    match = re.search(r'^.*?\bhas status\s+[\"\']?((?:KernelWorkerStatus\.)?[A-Za-z_]+)[\"\']?\s*$',
                      value, flags=re.MULTILINE | re.IGNORECASE)
    token = match.group(1) if match else value.strip('"\'')
    token = re.sub(r"^KernelWorkerStatus\.", "", token, flags=re.IGNORECASE).upper()
    if token in {"ERROR", "FAILED", "FAILURE", "CANCELED", "CANCELLED"}:
        return "ERROR"
    return token if token in {"COMPLETE", "RUNNING", "QUEUED", "PENDING", "INITIALIZING"} else "UNKNOWN"


def account_environment(account: dict) -> dict:
    env = os.environ.copy()
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        env.pop(key, None)
    env["KAGGLE_API_TOKEN"] = str(account["key"])
    env["KAGGLE_CONFIG_DIR"] = str(account["dir"])
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run_kaggle(account: dict, arguments: list[str], *, timeout: int) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", *arguments],
        env=account_environment(account), capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=True, timeout=timeout,
    )
    return result.stdout


def check_shard_status(shard_idx: int, account: dict) -> str:
    kernel = f"{account['username']}/prism-tier1b-shard{shard_idx}"
    return parse_kernel_status(run_kaggle(account, ["status", kernel], timeout=60))


def download_shard_outputs(shard_idx: int, account: dict, directory: Path) -> None:
    """Download only shard_{shard_idx}_report.json; retain partial output for retry after failure."""
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / ".download_in_progress"
    pending.touch()
    kernel = f"{account['username']}/prism-tier1b-shard{shard_idx}"
    file_pattern = f"^shard_{shard_idx}_report\\.json$"
    arguments = [
        "output", kernel,
        "-p", str(directory),
        "--file-pattern", file_pattern,
        "--page-size", "200",
        "--force",
    ]
    seen_tokens = set()
    while True:
        output = run_kaggle(account, arguments, timeout=3600)
        match = re.search(r"^Next page token:\s*(\S+)\s*$", output, re.MULTILINE)
        if not match:
            break
        token = match.group(1)
        if token in seen_tokens:
            raise ValueError("Kaggle repeated an output page token")
        seen_tokens.add(token)
        arguments = [
            "output", kernel,
            "-p", str(directory),
            "--file-pattern", file_pattern,
            "--page-size", "200",
            "--force",
            "--page-token", token,
        ]
    if pending.exists():
        pending.unlink()


def validate_shard_report(path: Path, shard_idx: int, *, expected_ids=None,
                          category_ids=None, annotations_sha256=None) -> dict:
    """Reject truncated reports, wrong partitions, and incomplete model/seed/CRF grids."""
    try:
        report = read_json(path)
        if (report["shard_idx"] != shard_idx or report["num_shards"] != NUM_SHARDS
                or report["dataset_image_count"] != 5000 or report["schema_version"] != 1
                or parse_kernel_status(report["status"]) != "COMPLETE"):
            raise ValueError("Wrong shard metadata or incomplete status")
        if report["crfs"] != list(CRFS) or report["seeds"] != list(SEEDS):
            raise ValueError("Wrong CRF grid or seeds")
        ids = report["image_ids"]
        expected_count = len(range(shard_idx, 5000, NUM_SHARDS))
        if (not isinstance(ids, list) or len(ids) != expected_count
                or any(type(i) is not int for i in ids) or ids != sorted(set(ids))):
            raise ValueError("Wrong image count or duplicate/unsorted image IDs")
        if expected_ids is not None and ids != list(expected_ids):
            raise ValueError("Image IDs do not match the annotation shard partition")
        if annotations_sha256 is not None and report["annotations_sha256"] != annotations_sha256:
            raise ValueError("Annotations SHA-256 mismatch")
        observations = report["observations"]
        if not isinstance(observations, list) or report["observation_count"] != len(observations):
            raise ValueError("Observation count mismatch")
        if category_ids is None:
            from tier1a.oracle_detector import COCO_IDS
            category_ids = set(COCO_IDS)
        validate_observations(observations, ids, CRFS, category_ids)
        return {
            "shard_idx": shard_idx, "path": str(path.resolve()), "sha256": sha256(path),
            "image_ids": ids, "observation_count": len(observations),
            "provenance": {key: report.get(key) for key in
                           ("annotations_sha256", "repo_commit", "checkpoints", "codec", "detectors", "preprocessing")},
        }
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError(f"Malformed shard {shard_idx} report: {type(exc).__name__}") from exc


def find_report(directory: Path, shard_idx: int) -> Path:
    name = f"shard_{shard_idx}_report.json"
    root = directory / name
    if root.is_file():
        return root
    matches = sorted(directory.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected one {name}, found {len(matches)}")
    return matches[0]


def poll_shards(accounts: list[dict], results_dir: Path, *, annotation_ids=None,
                category_ids=None, annotations_sha256=None) -> list[dict]:
    if not accounts:
        raise ValueError("No Kaggle pool accounts found")
    states = []
    for index in range(NUM_SHARDS):
        account = get_account_for_shard(index, accounts)
        directory = results_dir / f"shard_{index}"
        state = {"shard_idx": index, "account": account["username"],
                 "kernel_id": f"{account['username']}/prism-tier1b-shard{index}",
                 "status": "UNKNOWN", "downloaded": False}
        options = {"expected_ids": annotation_ids[index::NUM_SHARDS] if annotation_ids is not None else None,
                   "category_ids": category_ids, "annotations_sha256": annotations_sha256}
        try:
            state["status"] = check_shard_status(index, account)
            # Even a valid cached report cannot turn a RUNNING/ERROR kernel into COMPLETE.
            if state["status"] == "COMPLETE":
                try:
                    if (directory / ".download_in_progress").exists():
                        raise ValueError("Download was interrupted")
                    validated = validate_shard_report(find_report(directory, index), index, **options)
                except (OSError, ValueError):
                    print(f"[Harvester] Downloading shard {index} ({account['username']})", flush=True)
                    download_shard_outputs(index, account, directory)
                    validated = validate_shard_report(find_report(directory, index), index, **options)
                state.update(downloaded=True, report=validated)
        except subprocess.TimeoutExpired:
            state["error"] = "Kaggle request timed out; will retry"
        except subprocess.CalledProcessError as exc:
            # Never log remote stderr: some clients include authentication headers.
            state["error"] = f"Kaggle command failed (exit {exc.returncode}); will retry"
        except (OSError, ValueError) as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
        states.append(state)
        detail = "validated" if state["downloaded"] else state.get("error", "waiting")
        print(f"[Harvester] Shard {index:02d} | {state['status']:<12} | {detail}", flush=True)
    count = sum(s["downloaded"] for s in states)
    print(f"[Harvester] {count}/{NUM_SHARDS} shards complete ({100 * count / NUM_SHARDS:.1f}%)", flush=True)
    return states


def all_shards_complete(states: list[dict]) -> bool:
    return (len(states) == NUM_SHARDS and {s["shard_idx"] for s in states} == set(range(NUM_SHARDS))
            and all(s["status"] == "COMPLETE" and s["downloaded"] for s in states))


def finalize_harvest(states: list[dict], results_dir: Path, annotations: Path) -> dict:
    """Write a receipt only after both subprocesses and all output checks succeed."""
    if not all_shards_complete(states):
        raise ValueError("All 19 shards must be COMPLETE and validated before merging")
    reports = [s["report"] for s in sorted(states, key=lambda s: s["shard_idx"])]
    ids = [i for report in reports for i in report["image_ids"]]
    if len(ids) != 5000 or len(set(ids)) != 5000:
        raise ValueError("Shards must cover exactly 5,000 distinct images")
    if any(r["provenance"] != reports[0]["provenance"] for r in reports[1:]):
        raise ValueError("Shard provenance differs; refusing to pool incompatible runs")
    report_path = results_dir / FINAL_REPORT
    artifact_dir = results_dir / "paper_artifacts"
    receipt_path = results_dir / "harvest_receipt.json"
    signature = {"annotations_sha256": sha256(annotations),
                 "shards": [{"shard_idx": r["shard_idx"], "sha256": r["sha256"]} for r in reports]}
    # A restart with identical inputs and intact outputs is a no-op.
    if receipt_path.is_file():
        try:
            previous = read_json(receipt_path)
            outputs = [report_path, *(artifact_dir / n for n in ARTIFACT_NAMES)]
            if (previous.get("status") == "complete" and previous.get("inputs") == signature
                    and all(p.is_file() and previous.get("outputs", {}).get(str(p.resolve())) == sha256(p)
                            for p in outputs)):
                print(f"[Harvester] Existing completion receipt verified: {receipt_path}", flush=True)
                return previous
        except (OSError, ValueError, AttributeError):
            pass
    print("[Harvester] All shards validated; computing pooled COCO metrics and BD-rates", flush=True)
    subprocess.run([sys.executable, str(REPO / "scripts/merge_pool_5k.py"),
                    "--results-dir", str(results_dir), "--annotations", str(annotations),
                    "--num-shards", str(NUM_SHARDS), "--output", str(report_path)],
                   cwd=REPO, check=True)
    merged = read_json(report_path)
    if (merged["summary"]["total_unique_images"] != 5000
            or merged["summary"]["shards_found"] != NUM_SHARDS
            or not merged["results"].get("metrics") or not merged["results"].get("bd_rates")):
        raise ValueError("Final report is incomplete")
    subprocess.run([sys.executable, str(REPO / "scripts/generate_paper_artifacts.py"),
                    "--report", str(report_path), "--output-dir", str(artifact_dir)], cwd=REPO, check=True)
    outputs = [report_path, *(artifact_dir / n for n in ARTIFACT_NAMES)]
    if any(not p.is_file() or p.stat().st_size == 0 for p in outputs):
        raise ValueError("Final report or paper artifacts missing/empty")
    receipt = {"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
               "total_images": 5000, "shards_complete": NUM_SHARDS, "inputs": signature,
               "outputs": {str(p.resolve()): sha256(p) for p in outputs},
               "report": str(report_path.resolve()), "artifacts_dir": str(artifact_dir.resolve())}
    write_json(receipt_path, receipt)
    print(f"[Harvester] Complete: {receipt_path}", flush=True)
    return receipt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=Path.home() / ".kaggle_pool")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--poll-interval", type=float, default=180)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="instances_val2017.json, required to verify coverage and compute final COCO AP")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_interval) or args.poll_interval <= 0:
        parser.error("--poll-interval must be a positive finite number")
    for key in ("pool_dir", "results_dir", "annotations"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        annotations = read_json(args.annotations)
        ids = sorted(row["id"] for row in annotations["images"])
        if len(ids) != 5000 or len(set(ids)) != 5000:
            raise ValueError("--annotations must contain exactly 5,000 distinct COCO val2017 images")
        categories = {row["id"] for row in annotations["categories"]}
        annotation_hash = sha256(args.annotations)
        accounts = discover_pool_accounts(args.pool_dir)
        if not accounts:
            raise ValueError("No valid accounts found in --pool-dir")
        with exclusive(args.results_dir / ".harvester.lock"):
            while True:
                states = poll_shards(accounts, args.results_dir, annotation_ids=ids,
                                     category_ids=categories, annotations_sha256=annotation_hash)
                try:
                    if all_shards_complete(states):
                        finalize_harvest(states, args.results_dir, args.annotations)
                        return 0
                except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                    print(f"[Harvester] Finalization failed: {exc}", file=sys.stderr, flush=True)
                    if args.once:
                        return 1
                if args.once:
                    return int(any(s.get("error") or s["status"] in {"ERROR", "UNKNOWN"} for s in states))
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("[Harvester] Stopped; validated downloads are retained.", flush=True)
        return 130
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"[Harvester] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
