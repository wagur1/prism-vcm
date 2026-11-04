#!/usr/bin/env python3
"""Poll MOT17 evaluation kernels across Kaggle accounts, download reports, and merge.

Example:
    python scripts/auto_harvester_mot17.py --poll-interval 120
    python scripts/auto_harvester_mot17.py --once
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.merge_mot17_results import SEQUENCES, VARIANTS, CRFS, merge_and_report, validate_report
from tier1a.common import read_json, sha256

DEFAULT_MANIFEST = REPO / "outputs/mot17_kernels/dispatch_manifest.json"
DEFAULT_REPORTS_DIR = REPO / "docs/reports/mot17"
DEFAULT_POOL_DIR = Path("C:/Users/Wagur1/.kaggle_pool")


def parse_kernel_status(value: str) -> str:
    """Parse status token from kaggle CLI output."""
    if not isinstance(value, str):
        return "UNKNOWN"
    match = re.search(r'\bhas status\s+[\"\']?((?:KernelWorkerStatus\.)?[A-Za-z_]+)[\"\']?\s*$',
                      value, flags=re.MULTILINE | re.IGNORECASE)
    token = match.group(1) if match else value.strip('\"\'')
    token = re.sub(r"^KernelWorkerStatus\.", "", token, flags=re.IGNORECASE).upper()
    if token in {"ERROR", "FAILED", "FAILURE", "CANCELED", "CANCELLED"}:
        return "ERROR"
    return token if token in {"COMPLETE", "RUNNING", "QUEUED", "PENDING", "INITIALIZING"} else "UNKNOWN"


def load_assignments(manifest_path: Path, pool_dir: Path) -> list[dict]:
    """Load sequence kernel assignments from manifest or build fallback."""
    if manifest_path.is_file():
        data = read_json(manifest_path)
        assignments = data.get("assignments", [])
    else:
        # Fallback mapping
        default_mapping = {
            "MOT17-02-FRCNN": "thuha1205",
            "MOT17-04-FRCNN": "dieulinhh",
            "MOT17-05-FRCNN": "hoangminhhuy123",
            "MOT17-09-FRCNN": "huolgggnuyen",
            "MOT17-10-FRCNN": "linhowi05",
            "MOT17-11-FRCNN": "nguyenhoanglan1232",
            "MOT17-13-FRCNN": "shungg05",
        }
        assignments = []
        for seq, user in default_mapping.items():
            slug = f"prism-tier1b-{seq.lower().replace('_', '-')}"
            assignments.append({
                "sequence": seq,
                "username": user,
                "kernel_id": f"{user}/{slug}",
            })

    # Attach credentials info
    for a in assignments:
        user = a["username"]
        cred_path = pool_dir / user / "kaggle.json"
        if not cred_path.is_file():
            cred_path = Path.home() / f".kaggle/kaggle_{user}.json"
        if not cred_path.is_file():
            raise FileNotFoundError(f"Missing Kaggle credentials for account {user} at {cred_path}")
        cred = read_json(cred_path)
        a["key"] = cred["key"]
        a["cred_dir"] = cred_path.parent
    return assignments


def run_kaggle(account: dict, args: list[str], timeout: int = 60) -> str:
    env = os.environ.copy()
    for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        env.pop(k, None)
    env["KAGGLE_API_TOKEN"] = str(account["key"])
    env["KAGGLE_CONFIG_DIR"] = str(account["cred_dir"])
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    res = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", *args],
        capture_output=True, text=True, env=env, timeout=timeout,
        encoding="utf-8", errors="replace"
    )
    return res.stdout


def check_status(account: dict, kernel_id: str) -> str:
    out = run_kaggle(account, ["status", kernel_id], timeout=45)
    return parse_kernel_status(out)


def download_output(account: dict, kernel_id: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_pattern = r"^mot17_seq_report\.json$"
    run_kaggle(account, [
        "output", kernel_id,
        "-p", str(dest_dir),
        "--file-pattern", file_pattern,
        "--page-size", "200",
        "--force",
    ], timeout=180)
    canonical = dest_dir / "mot17_seq_report.json"
    if not canonical.is_file():
        candidates = list(dest_dir.glob("**/mot17_seq_report.json"))
        if candidates:
            import shutil
            shutil.copy2(candidates[0], canonical)


def poll_cycle(assignments: list[dict], reports_dir: Path) -> tuple[int, int, list[dict]]:
    """Poll each assignment, download if complete, and return progress summary."""
    status_list = []
    num_complete = 0

    for a in assignments:
        seq = a["sequence"]
        kernel_id = a["kernel_id"]
        seq_dir = reports_dir / seq
        report_file = seq_dir / "mot17_seq_report.json"

        # Check if already downloaded and validated
        is_ready = False
        if report_file.is_file():
            try:
                data = read_json(report_file)
                validate_report(data, seq)
                is_ready = True
            except Exception:
                pass

        if is_ready:
            st = "COMPLETE"
            num_complete += 1
            status_list.append({"seq": seq, "account": a["username"], "status": st, "state": "validated"})
            continue

        # Query Kaggle
        try:
            st = check_status(a, kernel_id)
        except Exception as exc:
            st = "QUERY_ERR"

        if st == "COMPLETE":
            print(f"[Harvester-MOT17] Downloading completed sequence {seq} from {kernel_id}...", flush=True)
            try:
                download_output(a, kernel_id, seq_dir)
                data = read_json(report_file)
                validate_report(data, seq)
                is_ready = True
                num_complete += 1
                status_list.append({"seq": seq, "account": a["username"], "status": st, "state": "downloaded"})
            except Exception as exc:
                status_list.append({"seq": seq, "account": a["username"], "status": st, "state": f"dl_err: {exc}"})
        else:
            status_list.append({"seq": seq, "account": a["username"], "status": st, "state": "running/pending"})

    return num_complete, len(assignments), status_list


def print_status_table(status_list: list[dict], num_complete: int, total: int) -> None:
    pct = (num_complete / total) * 100 if total > 0 else 0
    print(f"\n[Harvester-MOT17] {num_complete}/{total} sequences ready ({pct:.1f}%)")
    print("-" * 75)
    for row in status_list:
        print(f"  {row['seq']:16} | {row['account']:18} | {row['status']:10} | {row['state']}")
    print("-" * 75, flush=True)


def harvest(manifest_path: Path, reports_dir: Path, pool_dir: Path,
            poll_interval: int = 120, once: bool = False) -> None:
    assignments = load_assignments(manifest_path, pool_dir)
    print(f"[Harvester-MOT17] Tracking {len(assignments)} MOT17 sequences across pool accounts.", flush=True)

    while True:
        num_complete, total, status_list = poll_cycle(assignments, reports_dir)
        print_status_table(status_list, num_complete, total)

        if num_complete == total:
            print("[Harvester-MOT17] ALL 7 SEQUENCES COMPLETED! Merging results...", flush=True)
            merged_dir = reports_dir / "merged"
            final_report = merge_and_report(reports_dir, merged_dir)
            print(f"[Harvester-MOT17] Done! Final report: {final_report}", flush=True)
            break

        if once:
            break

        time.sleep(poll_interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--poll-interval", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    harvest(args.manifest, args.reports_dir, args.pool_dir, args.poll_interval, args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
