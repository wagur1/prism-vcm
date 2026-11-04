#!/usr/bin/env python3
"""Check GPU/TPU quota across all accounts in the Kaggle pool."""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import re
import subprocess
import sys

POOL_DIR = Path(r"C:\Users\Wagur1\.kaggle_pool")


def parse_quota_output(output: str) -> dict:
    gpu_match = re.search(r'GPU\s+([\d\.]+h)\s+([\d\.]+h)\s+([\d\.]+h)\s+(\S+)', output)
    tpu_match = re.search(r'TPU\s+([\d\.]+h)\s+([\d\.]+h)\s+([\d\.]+h)\s+(\S+)', output)

    return {
        "gpu_used": gpu_match.group(1) if gpu_match else "N/A",
        "gpu_remaining": gpu_match.group(2) if gpu_match else "N/A",
        "gpu_total": gpu_match.group(3) if gpu_match else "N/A",
        "refresh_at": gpu_match.group(4) if gpu_match else "N/A",
        "tpu_remaining": tpu_match.group(2) if tpu_match else "N/A",
    }


def check_account(user_dir: Path) -> dict:
    cred_file = user_dir / "kaggle.json"
    if not cred_file.is_file():
        return {"username": user_dir.name, "status": "NO_CRED"}

    try:
        with open(cred_file, encoding="utf-8") as f:
            cred = json.load(f)
        env = os.environ.copy()
        for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
            env.pop(k, None)
        env["KAGGLE_API_TOKEN"] = str(cred["key"])
        env["KAGGLE_USERNAME"] = cred["username"]
        env["KAGGLE_KEY"] = cred["key"]
        env["KAGGLE_CONFIG_DIR"] = str(user_dir)
        env["PYTHONIOENCODING"] = "utf-8"

        res = subprocess.run(
            [sys.executable, "-m", "kaggle", "quota"],
            capture_output=True, text=True, env=env, timeout=30,
            encoding="utf-8", errors="replace"
        )
        data = parse_quota_output(res.stdout)
        data["username"] = cred["username"]
        data["status"] = "OK" if data["gpu_remaining"] != "N/A" else "PARSE_ERR"
        # Parse float remaining for sorting
        rem_str = data["gpu_remaining"].replace("h", "")
        data["rem_float"] = float(rem_str) if rem_str != "N/A" else -1.0
        return data
    except Exception as exc:
        return {"username": user_dir.name, "status": f"ERR: {exc}", "rem_float": -1.0}


def main():
    account_dirs = [d for d in POOL_DIR.iterdir() if d.is_dir() and (d / "kaggle.json").is_file()]
    print(f"Checking quotas for {len(account_dirs)} accounts in {POOL_DIR}...\n", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(check_account, account_dirs))

    # Sort by GPU remaining descending
    results.sort(key=lambda x: x.get("rem_float", 0.0), reverse=True)

    header = f"{'Account':<20} | {'GPU Used':<10} | {'GPU Remaining':<14} | {'Total':<8} | {'Refresh At':<12} | Status"
    print(header)
    print("-" * len(header))

    total_remaining = 0.0
    for r in results:
        u = r["username"]
        used = r.get("gpu_used", "N/A")
        rem = r.get("gpu_remaining", "N/A")
        tot = r.get("gpu_total", "N/A")
        ref = r.get("refresh_at", "N/A")
        st = r["status"]
        if r.get("rem_float", 0) > 0:
            total_remaining += r["rem_float"]

        warning = " [OUT OF QUOTA]" if r.get("rem_float") == 0.0 else ""
        print(f"{u:<20} | {used:<10} | {rem:<14} | {tot:<8} | {ref:<12} | {st}{warning}")

    print("-" * len(header))
    print(f"Total available GPU hours in pool: {total_remaining:.2f} hours across {len(results)} accounts.")


if __name__ == "__main__":
    main()
