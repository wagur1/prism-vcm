#!/usr/bin/env python3
"""Prepare/dispatch seven MOT17 sequence kernels to distinct idle Kaggle accounts.

Examples (install the kaggle extra/package in the calling Python environment):
  uv run --no-sync --with kaggle scripts/dispatch_pool_mot17.py --dry-run
  uv run --no-sync --with kaggle scripts/dispatch_pool_mot17.py --push
  uv run --no-sync --with kaggle scripts/dispatch_pool_mot17.py --status
  uv run --no-sync --with kaggle scripts/dispatch_pool_mot17.py --pool-status

Dry runs query activity and write local workspaces but never push. Idle means
all listed owned kernels have terminal status; it does not guarantee GPU quota.
Assignments persist in dispatch_manifest.json so status never remaps sequences.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.dispatch_pool_5k import discover_pool_accounts
from tier1a.common import write_json
from tier1b.run_tier1b_mot17 import CRFS, VARIANTS, validate_options
from prism.video.mot_eval import MOT17_VALIDATION_SEQUENCES

SEQUENCES = tuple(f"{name}-FRCNN" for name in MOT17_VALIDATION_SEQUENCES)
DEFAULT_POOL_DIR = Path.home() / ".kaggle_pool"
DEFAULT_KERNELS_DIR = REPO / "outputs" / "mot17_kernels"
TERMINAL = {"complete", "error", "cancelled", "canceled"}


def kaggle(account: dict, arguments: list[str]) -> str:
    env = os.environ.copy()
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        env.pop(key, None)
    env["KAGGLE_CONFIG_DIR"] = str(Path(account["dir"]).resolve())
    if str(account["key"]).startswith("KGAT_"):
        env["KAGGLE_API_TOKEN"] = account["key"]
    else:
        env.update(KAGGLE_USERNAME=account["username"], KAGGLE_KEY=account["key"])
    result = subprocess.run([sys.executable, "-m", "kaggle", *arguments], env=env,
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=120)
    if result.returncode:
        # Do not echo subprocess diagnostics that might contain credentials.
        raise RuntimeError(f"Kaggle {arguments[:2]} failed for {account['username']} (exit {result.returncode})")
    return result.stdout


def check_kernel_status(account: dict, kernel_id: str) -> str:
    output = kaggle(account, ["kernels", "status", kernel_id])
    match = re.search(r"has status\s+[\"']?(?:KernelWorkerStatus\.)?(\w+)", output, re.I)
    return match.group(1).lower() if match else "unknown"


def account_activity(account: dict) -> dict:
    """Inspect every owned kernel; never infer idle from an API/auth failure."""
    if account["username"] == "baooo25r":
        return {"username": account["username"], "idle": False, "reason": "reserved for shard 09"}
    try:
        refs = set()
        page = 1
        while True:
            output = kaggle(account, ["kernels", "list", "--mine", "--csv",
                                      "--page-size", "100", "--page", str(page)])
            if output.strip().lower() in {"no kernels found", "not found"}:
                break
            # Skip warning lines and pagination tokens before CSV header
            clean_lines = [
                line for line in output.splitlines(keepends=True)
                if not line.startswith("Warning:") and not line.startswith("Next Page Token = ")
            ]
            reader = csv.DictReader(io.StringIO("".join(clean_lines)))
            if not reader.fieldnames or "ref" not in reader.fieldnames:
                break
            rows = list(reader)
            # Only monitor PRISM-related kernels to avoid dozens of redundant API calls
            current = {row["ref"] for row in rows if row.get("ref") and ("prism-" in row["ref"])}
            refs.update(current)
            if len(rows) < 100 or page >= 2:
                break
            page += 1
        statuses = {ref: check_kernel_status(account, ref) for ref in sorted(refs)}
        return {"username": account["username"],
                "idle": all(status in TERMINAL for status in statuses.values()),
                "kernels": statuses}
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        return {"username": account["username"], "idle": False, "error": str(exc)}


def select_idle_accounts(accounts: list[dict]) -> list[dict]:
    selected = []
    seen = set()
    for account in accounts:
        if account["username"] in seen:
            continue
        seen.add(account["username"])
        activity = account_activity(account)
        print(f"{account['username']}: {'idle' if activity['idle'] else 'busy/unavailable'}", flush=True)
        if activity["idle"]:
            selected.append(account)
        if len(selected) == len(SEQUENCES):
            return selected
    raise RuntimeError(f"Need 7 distinct idle accounts; found {len(selected)}")


def source_bundle() -> tuple[str, str, str]:
    """Embed current runtime sources so uncommitted runner changes reach Kaggle."""
    revision = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    paths = list((REPO / "prism").rglob("*.py"))
    paths += [REPO / name for name in (
        "tier1a/__init__.py", "tier1a/common.py", "tier1a/oracle_codec.py",
        "tier1b/__init__.py", "tier1b/run_tier1b_mot17.py",
        "prism/codec/native/CMakeLists.txt", "prism/codec/native/x265_shim.c",
        "gate0/build_gate0_stack.sh", "gate0/x265_residual_patch.patch",
        "gate0/x265_residual_telemetry.patch",
    )]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            # Normalize Windows line endings for Bash and patch consumers.
            archive.writestr(path.relative_to(REPO).as_posix(),
                             path.read_bytes().replace(b"\r\n", b"\n"))
    payload = buffer.getvalue()
    return base64.b64encode(payload).decode("ascii"), hashlib.sha256(payload).hexdigest(), revision


def make_sequence_runner_script(sequence: str, bundle: tuple[str, str, str], options: dict) -> str:
    payload, digest, revision = bundle
    return f'''#!/usr/bin/env python3
"""PRISM Plan B: {sequence}. Current source bundle SHA256: {digest}."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

SEQUENCE = {sequence!r}
OPTIONS = {options!r}
SOURCE_SHA256 = {digest!r}
work = Path('/kaggle/working')
repo = work / 'prism-vcm'
inputs = Path('/kaggle/input')

def call(args, **kwargs):
    subprocess.run(args, check=True, **kwargs)

def locate_inputs():
    sequences = sorted(p.parent for p in inputs.rglob('seqinfo.ini')
                       if p.parent.name == SEQUENCE and (p.parent / 'gt/gt.txt').is_file())
    checkpoints = sorted(inputs.rglob('student_seed_0.pt'))
    return sequences, checkpoints

sequences, checkpoints = [], []
for attempt in range(15):
    sequences, checkpoints = locate_inputs()
    if checkpoints and (sequences or any(inputs.rglob('*.zip'))):
        break
    time.sleep(5)
if not sequences:
    # Some versions of the MOT17 dataset mount the original zip archive.
    for archive_path in sorted(inputs.rglob('*.zip')):
        with zipfile.ZipFile(archive_path) as archive:
            roots = [name[:-len('seqinfo.ini')] for name in archive.namelist()
                     if name.endswith('/' + SEQUENCE + '/seqinfo.ini')
                     and name[:-len('seqinfo.ini')] + 'gt/gt.txt' in archive.namelist()]
            for prefix in roots:
                target = work / 'mot17' / SEQUENCE
                target.mkdir(parents=True, exist_ok=True)
                for name in archive.namelist():
                    if not name.startswith(prefix) or name.endswith('/'):
                        continue
                    relative = Path(name[len(prefix):])
                    if relative.is_absolute() or '..' in relative.parts:
                        raise RuntimeError('Unsafe dataset archive member')
                    dest = target / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(name) as src, dest.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                sequences.append(target)
if len(sequences) != 1 or len(checkpoints) != 1:
    raise RuntimeError(f'Expected one labeled {{SEQUENCE}} and seed-0 checkpoint; '
                       f'found {{len(sequences)}} and {{len(checkpoints)}}')

call(['git', 'clone', 'https://github.com/wagur1/prism-vcm.git', str(repo)])
call(['git', '-C', str(repo), 'checkout', {revision!r}])
payload = base64.b64decode({payload!r})
assert hashlib.sha256(payload).hexdigest() == SOURCE_SHA256
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(repo)
call([sys.executable, '-m', 'pip', 'install', '--quiet', 'ultralytics', 'timm', 'Pillow', 'scipy', 'PyYAML'])
env = os.environ.copy()
env.update(GATE0_JOBS='4', GATE0_PYTHON=sys.executable, GATE0_ASSEMBLY='ON')
call(['bash', 'gate0/build_gate0_stack.sh'], cwd=repo, env=env)
stack = json.loads((repo / 'gate0/stack.json').read_text())
env.update(PRISM_X265_LIBRARY=stack['library'], PRISM_X265_SHIM=stack['shim'])
weights = work / 'yolo11l.pt'
urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt', weights)
output = work / SEQUENCE
command = [sys.executable, '-u', str(repo / 'tier1b/run_tier1b_mot17.py'),
           '--sequence-dir', str(sequences[0]), '--checkpoints-dir', str(checkpoints[0].parent),
           '--weights', str(weights), '--output', str(output), '--device', 'cuda',
           '--gop', str(OPTIONS['gop']), '--bframes', str(OPTIONS['bframes']),
           '--crfs', *map(str, OPTIONS['crfs']), '--variants', *OPTIONS['variants']]
if OPTIONS['max_frames'] is not None:
    command += ['--max-frames', str(OPTIONS['max_frames'])]
call(command, cwd=repo, env=env)
report_path = output / 'mot17_seq_report.json'
report = json.loads(report_path.read_text())
report['source_bundle_sha256'] = SOURCE_SHA256
report['base_repo_revision'] = {revision!r}
report_path.write_text(json.dumps(report, indent=2) + '\\n')
shutil.copy2(report_path, work / 'mot17_seq_report.json')
'''


def generate_sequence_kernel(sequence, username, target_dir, bundle, options) -> dict:
    slug = f"prism-tier1b-{sequence.lower()}"
    metadata = {
        "id": f"{username}/{slug}", "title": slug, "code_file": "tier1b_mot17.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": ["ahmedsamir1598/mot17challenge"],
        "kernel_sources": ["htran123456/prism-tier1a-step3"],
        "competition_sources": [], "model_sources": [],
    }
    target_dir.mkdir(parents=True, exist_ok=True)
    write_json(target_dir / "kernel-metadata.json", metadata)
    (target_dir / "tier1b_mot17.py").write_text(
        make_sequence_runner_script(sequence, bundle, options), encoding="utf-8")
    return {"sequence": sequence, "username": username, "kernel_id": metadata["id"],
            "kernel_dir": str(target_dir.resolve()), "push_status": "prepared"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--kernels-dir", type=Path, default=DEFAULT_KERNELS_DIR)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--push", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--pool-status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--crfs", nargs="+", type=int, default=list(CRFS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--gop", type=int, default=32)
    parser.add_argument("--bframes", type=int, choices=(0, 3), default=0)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    args.device = "cuda"
    validate_options(args)
    accounts = discover_pool_accounts(args.pool_dir)
    by_user = {a["username"]: a for a in accounts}
    manifest_path = args.kernels_dir / "dispatch_manifest.json"
    if args.pool_status:
        for account in accounts:
            print(json.dumps(account_activity(account)))
        return 0
    if args.status:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        failed = False
        for row in manifest["assignments"]:
            try:
                status = check_kernel_status(by_user[row["username"]], row["kernel_id"])
            except (KeyError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                status, failed = str(exc), True
            print(f"{row['sequence']} | {row['kernel_id']} | {status}", flush=True)
        return int(failed)
    options = {key: getattr(args, key) for key in ("crfs", "variants", "gop", "bframes", "max_frames")}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["options"] != options:
            raise ValueError("Options differ from saved dispatch; use a new --kernels-dir")
    else:
        selected = select_idle_accounts(accounts)
        bundle = source_bundle()
        assignments = [generate_sequence_kernel(seq, account["username"], args.kernels_dir / seq,
                                                 bundle, options)
                       for seq, account in zip(SEQUENCES, selected)]
        manifest = {"options": options, "source_bundle_sha256": bundle[1], "assignments": assignments}
        write_json(manifest_path, manifest)
    for row in manifest["assignments"]:
        print(f"{row['sequence']} -> {row['kernel_id']} ({row['push_status']})", flush=True)
    failures = 0
    if args.push and not args.dry_run:
        for row in manifest["assignments"]:
            if row["push_status"] == "pushed":
                continue
            account = by_user[row["username"]]
            if not account_activity(account)["idle"]:
                print(f"Skipping busy/unavailable account: {row['username']}", flush=True)
                failures += 1
                continue
            try:
                kaggle(account, ["kernels", "push", "-p", row["kernel_dir"]])
                row["push_status"] = "pushed"
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                row["push_status"] = "failed"
                failures += 1
                print(str(exc), file=sys.stderr)
            write_json(manifest_path, manifest)
    print(f"Manifest: {manifest_path}")
    print("Check sequence status with --status; check account activity with --pool-status.")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
