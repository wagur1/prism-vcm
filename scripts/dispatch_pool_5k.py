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


def make_shard_runner_script(shard_idx: int, num_shards: int, repo_url: str = "https://github.com/wagur1/prism-vcm.git", branch: str = "main") -> str:
    """Generate the self-contained tier1b_shard.py script for Kaggle execution."""
    return f'''#!/usr/bin/env python3
"""PRISM-VCM Tier 1b Shard {shard_idx} Runner on Kaggle.
Evaluates shard {shard_idx}/{num_shards} of COCO val2017 (5,000 images).
"""
import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SHARD_IDX = {shard_idx}
NUM_SHARDS = {num_shards}
REPO_URL = "{repo_url}"
BRANCH = "{branch}"

print(f"=== [PRISM Tier 1b] Starting Shard {{SHARD_IDX}}/{{NUM_SHARDS}} ===", flush=True)

# 1. Inspect Kaggle environment and locate inputs
input_dir = Path("/kaggle/input")
working_dir = Path("/kaggle/working")
repo_dir = working_dir / "prism-vcm"

# Locate annotations and images from awsaf49/coco-2017-dataset
ann_matches = list(input_dir.glob("**/instances_val2017.json"))
if not ann_matches:
    raise FileNotFoundError("Could not locate instances_val2017.json in /kaggle/input")
annotations_path = ann_matches[0]
print(f"Located annotations: {{annotations_path}}", flush=True)

val2017_matches = [p for p in input_dir.glob("**/val2017") if p.is_dir() and not p.name.endswith(".json")]
if not val2017_matches:
    raise FileNotFoundError("Could not locate val2017/ directory in /kaggle/input")
images_dir = val2017_matches[0]
print(f"Located images: {{images_dir}}", flush=True)

# Locate student checkpoints from htran123456/prism-tier1a-step3
ckpt_matches = list(input_dir.glob("**/student_seed_0.pt"))
if not ckpt_matches:
    raise FileNotFoundError("Could not locate student checkpoints in /kaggle/input")
checkpoints_dir = ckpt_matches[0].parent
print(f"Located checkpoints: {{checkpoints_dir}}", flush=True)

# 2. Clone latest PRISM repo
if repo_dir.exists():
    shutil.rmtree(repo_dir)
print(f"Cloning {{REPO_URL}} (branch: {{BRANCH}})...", flush=True)
subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(repo_dir)], check=True)

# 3. Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "ultralytics", "timm", "pycocotools", "scipy"], check=True)

# 4. Download / verify YOLOv11-L weights
weights_path = working_dir / "yolo11l.pt"
if not weights_path.exists():
    print("Provisioning YOLO11-L weights...", flush=True)
    yolo_matches = list(input_dir.glob("**/yolo11l.pt"))
    if yolo_matches:
        shutil.copy(yolo_matches[0], weights_path)
    else:
        from ultralytics import YOLO
        model = YOLO("yolo11l.pt")
        shutil.copy(Path(model.ckpt_path), weights_path)
print(f"YOLO11-L weights ready: {{weights_path}}", flush=True)

# 5. Build GATE-0 stack (ASM=ON)
gate_dir = repo_dir / "gate0"
print("Building GATE-0 x265 stack with NASM ASM=ON...", flush=True)
build_env = os.environ.copy()
build_env["GATE0_JOBS"] = "4"
build_env["GATE0_PYTHON"] = sys.executable
build_env["GATE0_ASSEMBLY"] = "ON"
subprocess.run(["bash", str(gate_dir / "build_gate0_stack.sh")], cwd=str(repo_dir), env=build_env, check=True)

with open(gate_dir / "stack.json") as f:
    stack = json.load(f)
print(f"GATE-0 stack verified: enable_assembly={{stack.get('enable_assembly')}}", flush=True)

# 6. Execute run_tier1b_val5k.py for this shard
output_dir = working_dir / "tier1b_output"
output_dir.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, str(repo_dir / "tier1b" / "run_tier1b_val5k.py"),
    "--images-dir", str(images_dir),
    "--annotations", str(annotations_path),
    "--checkpoints-dir", str(checkpoints_dir),
    "--shard-idx", str(SHARD_IDX),
    "--num-shards", str(NUM_SHARDS),
    "--output", str(output_dir),
    "--gate", str(gate_dir),
    "--weights", str(weights_path),
    "--crfs", "20", "24", "28", "32", "36",
]
print(f"Executing: {{' '.join(cmd)}}", flush=True)
subprocess.run(cmd, cwd=str(repo_dir), check=True)

report_src = output_dir / f"shard_{{SHARD_IDX}}_report.json"
report_dst = working_dir / f"shard_{{SHARD_IDX}}_report.json"
if report_src.exists():
    shutil.copy(report_src, report_dst)
    print(f"SUCCESS! Shard {{SHARD_IDX}} report saved to {{report_dst}} ({{report_dst.stat().st_size}} bytes)", flush=True)
else:
    raise FileNotFoundError(f"Expected {{report_src}} was not created!")
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
    env["KAGGLE_API_TOKEN"] = str(account["key"])
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
    env["KAGGLE_API_TOKEN"] = str(account["key"])
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
            acc = accounts[s_idx % len(accounts)]
            st = check_shard_status(s_idx, acc)
            print(f"Shard {s_idx:<2} | {acc['username']:<20} | {st['status']}")
        return

    # Default action: prepare and optionally push
    do_prepare = args.prepare or not args.push
    if do_prepare:
        print(f"\n=== Preparing {len(target_shards)} Shard Kernels in {args.kernels_dir} ===")
        for s_idx in target_shards:
            acc = accounts[s_idx % len(accounts)]
            shard_dir = args.kernels_dir / f"shard_{s_idx}"
            generate_shard_kernel(s_idx, args.num_shards, acc["username"], shard_dir)
            print(f"  [OK] Shard {s_idx:02d} -> {shard_dir} ({acc['username']})")

    if args.push:
        print(f"\n=== Pushing {len(target_shards)} Shards to Kaggle ===")
        success_count = 0
        for s_idx in target_shards:
            acc = accounts[s_idx % len(accounts)]
            shard_dir = args.kernels_dir / f"shard_{s_idx}"
            if push_shard_kernel(shard_dir, acc, dry_run=args.dry_run):
                success_count += 1
            time.sleep(1)
        print(f"\nPushed {success_count}/{len(target_shards)} shard kernels successfully.")


if __name__ == "__main__":
    main()
