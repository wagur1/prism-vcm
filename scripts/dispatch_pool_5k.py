#!/usr/bin/env python3
"""Dispatch PRISM-VCM Tier 1b full 5,000-image evaluation across the 19-account Kaggle pool.

Generates kernel workspaces and runners for each shard (0..18), pushes kernels
using individual account credentials from ~/.kaggle_pool, and monitors status.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
DEFAULT_POOL_DIR = Path.home() / ".kaggle_pool"
DEFAULT_KERNELS_DIR = Path("C:/Users/Wagur1/Desktop/HOI_DONG_9/kernels/tier1b_shards")
NUM_SHARDS = 19


def discover_pool_accounts(pool_dir: Path) -> list[dict]:
    """Find all accounts with valid kaggle.json in pool_dir, sorted deterministically."""
    if not pool_dir.is_dir():
        raise FileNotFoundError(f"Pool directory not found: {pool_dir}")
    accounts = []
    for entry in sorted(pool_dir.iterdir()):
        if entry.is_dir():
            kjson = entry / "kaggle.json"
            if kjson.is_file():
                try:
                    with open(kjson, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    username = data.get("username")
                    key = data.get("key")
                    if username and key:
                        accounts.append({
                            "username": username,
                            "key": key,
                            "dir": entry,
                            "config_path": kjson,
                        })
                except Exception as exc:
                    print(f"[Warning] Failed to read {kjson}: {exc}", file=sys.stderr)
    return sorted(accounts, key=lambda a: a["username"])


OVERRIDE_SHARD_ACCOUNTS = {
    6: "hoangminhhuy123",      # Tested & Verified GPU quota available (replaces htran123456)
    9: "baooo25r",             # Re-dispatch failed shard 9 to baooo25r (~25h GPU remaining)
    11: "huolgggnuyen",        # 29.99h GPU quota remaining
    15: "nguyenhoanglan1232",  # 27.81h GPU quota remaining
}


def get_account_for_shard(shard_idx: int, accounts: list[dict]) -> dict:
    """Resolve account for shard, applying quota overrides when needed."""
    if shard_idx in OVERRIDE_SHARD_ACCOUNTS:
        target_user = OVERRIDE_SHARD_ACCOUNTS[shard_idx]
        for acc in accounts:
            if acc["username"] == target_user:
                return acc
    return accounts[shard_idx % len(accounts)]


def make_shard_runner_script(shard_idx: int, num_shards: int, repo_url: str = "https://github.com/wagur1/prism-vcm.git", branch: str = "main") -> str:
    """Generate the self-contained tier1b_shard.py script for Kaggle execution."""
    return f'''#!/usr/bin/env python3
"""PRISM-VCM Tier 1b Shard {shard_idx} Runner on Kaggle."""
import os
import subprocess
import sys

cmd = r\'\'\'set -e
LOG_FILE="/kaggle/working/shard_{shard_idx}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== [PRISM Tier 1b] Shard {shard_idx}/{num_shards} START $(date -u +%FT%TZ) ==="

# 1. Inspect Kaggle environment and locate inputs with mount retry loop
echo "=== Scanning /kaggle/input ==="
for i in $(seq 1 15); do
    ANN_PATH=$(find /kaggle/input -name 'instances_val2017.json' 2>/dev/null | head -n 1)
    VAL2017_DIR=$(find /kaggle/input -type d -name 'val2017' 2>/dev/null | head -n 1)
    CKPT_PATH=$(find /kaggle/input -name 'student_seed_0.pt' 2>/dev/null | head -n 1)
    if [ -n "$ANN_PATH" ] && [ -n "$VAL2017_DIR" ] && [ -n "$CKPT_PATH" ]; then
        break
    fi
    echo "[$i/15] Waiting for /kaggle/input datasets to mount (ann: $ANN_PATH, val: $VAL2017_DIR, ckpt: $CKPT_PATH)..."
    sleep 5
done

if [ -z "$ANN_PATH" ]; then
    echo "ERROR: Could not locate instances_val2017.json"
    ls -la /kaggle/input
    exit 1
fi
echo "Located annotations: $ANN_PATH"

if [ -z "$VAL2017_DIR" ]; then
    echo "ERROR: Could not locate val2017 directory"
    exit 1
fi
echo "Located images dir: $VAL2017_DIR"

if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: Could not locate student_seed_0.pt"
    exit 1
fi
CKPT_DIR=$(dirname "$CKPT_PATH")
echo "Located checkpoints dir: $CKPT_DIR"

# 2. Clone latest PRISM repo
rm -rf /kaggle/working/prism-vcm
echo "Cloning {repo_url} (branch: {branch})..."
git clone --depth 1 -b {branch} {repo_url} /kaggle/working/prism-vcm
cd /kaggle/working/prism-vcm

# CRLF line ending fix
sed -i 's/\\r$//' gate0/build_gate0_stack.sh

# 3. Install dependencies
echo "Installing dependencies..."
pip install --quiet ultralytics timm pycocotools scipy

# 4. Provision YOLOv11-L weights
mkdir -p /kaggle/working/cache/weights
WEIGHTS_PATH="/kaggle/working/cache/weights/yolo11l.pt"
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "Downloading YOLO11-L weights..."
    curl -sSL -f -o "$WEIGHTS_PATH" https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt || (python3 -c "from ultralytics import YOLO; YOLO('yolo11l.pt')" && mv -f yolo11l.pt "$WEIGHTS_PATH")
fi
echo "YOLO11-L weights ready at: $WEIGHTS_PATH"

# 5. Build GATE-0 stack (ASM=ON)
echo "Building GATE-0 x265 stack with NASM ASM=ON..."
export GATE0_JOBS=4
export GATE0_PYTHON=python3
export GATE0_ASSEMBLY=ON
bash gate0/build_gate0_stack.sh
echo "GATE-0 stack built successfully."

# 6. Execute run_tier1b_val5k.py
OUTPUT_DIR="/kaggle/working/tier1b_output"
mkdir -p "$OUTPUT_DIR"

echo "Executing run_tier1b_val5k.py for shard {shard_idx}..."
python3 -u tier1b/run_tier1b_val5k.py \\
    --images-dir "$VAL2017_DIR" \\
    --annotations "$ANN_PATH" \\
    --checkpoints-dir "$CKPT_DIR" \\
    --shard-idx {shard_idx} \\
    --num-shards {num_shards} \\
    --output "$OUTPUT_DIR" \\
    --gate gate0 \\
    --weights "$WEIGHTS_PATH" \\
    --crfs 20 24 28 32 36

# Copy final report to /kaggle/working
if [ -f "$OUTPUT_DIR/shard_{shard_idx}_report.json" ]; then
    cp -f "$OUTPUT_DIR/shard_{shard_idx}_report.json" /kaggle/working/shard_{shard_idx}_report.json
    echo "=== SUCCESS: shard_{shard_idx}_report.json created ($(stat -c%s /kaggle/working/shard_{shard_idx}_report.json) bytes) ==="
else
    echo "ERROR: Report file was not created!"
    exit 1
fi
\'\'\'

res = subprocess.run(["bash", "-c", cmd])
sys.exit(res.returncode)
'''


def generate_shard_kernel(shard_idx: int, num_shards: int, username: str, target_dir: Path) -> Path:
    """Generate kernel-metadata.json and tier1b_shard.py in target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    slug = f"prism-tier1b-shard{shard_idx}"
    metadata = {
        "id": f"{username}/{slug}",
        "title": f"prism-tier1b-shard{shard_idx}",
        "code_file": "tier1b_shard.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [
            "awsaf49/coco-2017-dataset"
        ],
        "competition_sources": [],
        "kernel_sources": [
            "htran123456/prism-tier1a-step3"
        ],
        "model_sources": []
    }
    with open(target_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    script_content = make_shard_runner_script(shard_idx, num_shards)
    with open(target_dir / "tier1b_shard.py", "w", encoding="utf-8") as f:
        f.write(script_content)

    return target_dir


def push_shard_kernel(kernel_dir: Path, account: dict, dry_run: bool = False) -> bool:
    """Push kernel to Kaggle using account-specific credentials."""
    username = account["username"]
    slug = kernel_dir.name
    print(f"[{account['username']}] Pushing {slug} from {kernel_dir}...", flush=True)
    if dry_run:
        print(f"  [DRY-RUN] Skipped push for {username}/{slug}")
        return True

    env = os.environ.copy()
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        env.pop(key, None)
    env["KAGGLE_API_TOKEN"] = str(account["key"])
    env["KAGGLE_CONFIG_DIR"] = str(account["dir"])
    cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kernel_dir)]
    try:
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        print(f"  Output: {res.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  ERROR: {exc.stderr.strip() or exc.stdout.strip()}", file=sys.stderr)
        return False


def check_shard_status(shard_idx: int, account: dict) -> dict:
    """Query Kaggle kernel status for shard_idx."""
    username = account["username"]
    kernel_id = f"{username}/prism-tier1b-shard{shard_idx}"
    env = os.environ.copy()
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        env.pop(key, None)
    env["KAGGLE_API_TOKEN"] = str(account["key"])
    env["KAGGLE_CONFIG_DIR"] = str(account["dir"])
    cmd = [sys.executable, "-m", "kaggle", "kernels", "status", kernel_id]
    try:
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        output = res.stdout.strip()
        status = "UNKNOWN"
        for part in output.split("\n"):
            if "status" in part:
                status = part.strip()
        return {"shard_idx": shard_idx, "account": username, "kernel_id": kernel_id, "status": status, "raw": output}
    except subprocess.CalledProcessError as exc:
        return {"shard_idx": shard_idx, "account": username, "kernel_id": kernel_id, "status": "ERROR", "error": exc.stderr.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR, help="Path to kaggle pool directory")
    parser.add_argument("--kernels-dir", type=Path, default=DEFAULT_KERNELS_DIR, help="Directory to save generated shard kernels")
    parser.add_argument("--num-shards", type=int, default=NUM_SHARDS, help="Number of shards (default: 19)")
    parser.add_argument("--shard-idx", nargs="+", type=int, help="Specific shard indices to process (default: all)")
    parser.add_argument("--prepare", action="store_true", help="Generate kernel workspaces and scripts")
    parser.add_argument("--push", action="store_true", help="Push kernels to Kaggle accounts")
    parser.add_argument("--status", action="store_true", help="Check status of all shard kernels")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without modifying Kaggle state")
    args = parser.parse_args()

    accounts = discover_pool_accounts(args.pool_dir)
    print(f"Discovered {len(accounts)} accounts in pool {args.pool_dir}")
    if len(accounts) < args.num_shards:
        print(f"[Warning] Found {len(accounts)} accounts, but requested {args.num_shards} shards.")

    target_shards = args.shard_idx if args.shard_idx is not None else list(range(args.num_shards))

    if args.status:
        print("\n=== Checking Shard Status Across Pool ===")
        print(f"{'Shard':<8} | {'Account':<20} | {'Status'}")
        print("-" * 65)
        for s_idx in target_shards:
            acc = get_account_for_shard(s_idx, accounts)
            st = check_shard_status(s_idx, acc)
            print(f"Shard {s_idx:<2} | {acc['username']:<20} | {st['status']}")
        return

    # Default action: prepare and optionally push
    do_prepare = args.prepare or not args.push
    if do_prepare:
        print(f"\n=== Preparing {len(target_shards)} Shard Kernels in {args.kernels_dir} ===")
        for s_idx in target_shards:
            acc = get_account_for_shard(s_idx, accounts)
            shard_dir = args.kernels_dir / f"shard_{s_idx}"
            generate_shard_kernel(s_idx, args.num_shards, acc["username"], shard_dir)
            print(f"  [OK] Shard {s_idx:02d} -> {shard_dir} ({acc['username']})")

    if args.push:
        print(f"\n=== Pushing {len(target_shards)} Shards to Kaggle ===")
        success_count = 0
        for s_idx in target_shards:
            acc = get_account_for_shard(s_idx, accounts)
            shard_dir = args.kernels_dir / f"shard_{s_idx}"
            if push_shard_kernel(shard_dir, acc, dry_run=args.dry_run):
                success_count += 1
            time.sleep(1)
        print(f"\nPushed {success_count}/{len(target_shards)} shard kernels successfully.")


if __name__ == "__main__":
    main()
