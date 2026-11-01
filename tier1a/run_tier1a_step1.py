#!/usr/bin/env python3
"""Kaggle Step 1 entrypoint. Also usable as a standalone repo bootstrap file."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO_URL = "https://github.com/wagur1/prism-vcm.git"
SAM_INSTALL = "git+https://github.com/facebookresearch/sam2.git"


def bootstrap(repo: Path) -> None:
    if not (repo / ".git").exists():
        subprocess.run(["git", "clone", REPO_URL, str(repo)], check=True, timeout=300)
    if not (repo / "tier1a/cache_teachers.py").is_file():
        raise RuntimeError("Checkout lacks Step 1; chair must publish this commit. Existing checkouts are not pulled/reset.")
    sys.path.insert(0, str(repo))


def device_count() -> int:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True, timeout=15)
        return len(result.stdout.strip().splitlines())
    except (OSError, subprocess.SubprocessError):
        try:
            import torch
            return torch.cuda.device_count()
        except ImportError:
            return 0


def dependency_versions() -> dict:
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "Pillow", "SAM-2")}
    distribution = importlib.metadata.distribution("SAM-2")
    versions["sam_origin"] = json.loads(distribution.read_text("direct_url.json") or "{}")
    return versions


def install_dependencies(repo: Path, cache: Path, output: Path, *, deadline: float) -> str:
    from tier1a.common import read_json, write_json

    receipt = output / "dependencies.json"
    if receipt.exists():
        try:
            current = dependency_versions()
        except importlib.metadata.PackageNotFoundError:
            current = None
        if current is not None:
            if current != read_json(receipt):
                raise RuntimeError("Dependency versions changed since the frozen extraction environment")
            return "SKIPPED"
    env = dict(os.environ, PIP_CACHE_DIR=str(cache / "pip"), TORCH_HOME=str(cache / "weights/torch_hub"),
               SAM2_BUILD_CUDA="0", TMPDIR=str(cache / "tmp"))
    (cache / "tmp").mkdir(parents=True, exist_ok=True)
    commands = [[sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                 "--no-cache-dir", "numpy>=1.24", "Pillow>=10", "torch>=2.5.1", "torchvision>=0.20.1"],
                [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                 "--no-cache-dir", "--no-build-isolation", SAM_INSTALL]]
    # sam2 is Meta's SAM-2 distribution; use its official source, not a namesake.
    with (output / "install.log").open("a", encoding="utf-8") as log:
        for command in commands:
            log.write("ARGV " + json.dumps(command) + "\n")
            log.flush()
            timeout = min(1800, deadline - time.monotonic() - 120)
            if timeout <= 0:
                raise RuntimeError("Insufficient job allowance to provision dependencies")
            subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)
    versions = dependency_versions()
    if receipt.exists() and versions != read_json(receipt):
        raise RuntimeError("Reinstalled dependencies differ from the frozen extraction environment")
    write_json(receipt, versions)
    return "PASS"


def artifact_manifest(output: Path, ledger: Path) -> list[dict]:
    paths = [p for p in output.rglob("*") if p.is_file() and
             p.name not in {"tier1a_step1_report.json", ".writer.lock", ".prepare.lock"}]
    if ledger.exists():
        paths.append(ledger)
    artifacts = []
    for path in sorted(set(paths)):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        artifacts.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    local = Path(__file__).resolve().parents[1]
    parser.add_argument("--repo", type=Path, default=local if (local / ".git").exists() else Path("/kaggle/working/prism-vcm"))
    parser.add_argument("--working", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--prior-gpu-hours", type=float, default=0,
                        help="All pilot GPU usage outside this persistent Step 1 account; never decreases")
    parser.add_argument("--offline", action="store_true", help="No install or network; require verified caches")
    parser.add_argument("--skip-install", action="store_true", help="Use pre-provisioned torch/torchvision/SAM-2")
    args = parser.parse_args()
    args.repo, args.working = args.repo.resolve(), args.working.resolve()
    output, cache = args.working / "tier1a", args.working / "cache"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "tier1a_step1_report.json"
    ledger = args.repo / "prism/data/splits/coco500_ledger.json"
    report = {"status": "RUNNING", "step1_complete": False,
              "started_utc": datetime.now(timezone.utc).isoformat(), "steps": {}, "seed": 0,
              "argv": list(sys.argv),
              "oracle_encodes": 0, "student_executed": False, "val2017_accessed": False}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    budget = None
    locks = ExitStack()
    phase = "bootstrap"
    started = time.monotonic()
    cpu_started = time.process_time()
    try:
        if args.offline and not (args.repo / ".git").exists():
            raise RuntimeError(f"Offline repo cache miss; URL={REPO_URL}")
        bootstrap(args.repo)
        from tier1a.common import check_storage, exclusive, sha256, write_json
        from tier1a.resources import ResourceBudget
        # Hold the lock through final accounting and manifest publication.
        locks.enter_context(exclusive(output / "teachers/.writer.lock"))
        budget = ResourceBudget(output / "teachers/gpu_hours.json", devices=device_count(),
                                prior_gpu_hours=args.prior_gpu_hours)
        budget.arm(report_path)
        check_storage(args.working, 4_000_000_000)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=args.repo, text=True).strip()
        report.update(repo_sha=head, repo_dirty=bool(dirty),
            config_sha256={name: sha256(args.repo / "docs" / name)
                           for name in ("CONFIG_V1_4.md", "CONFIG_V1_3.md")},
            python=sys.version, allocated_gpus=budget.devices)
        code_paths = list((args.repo / "tier1a").glob("*.py")) + [
            args.repo / "prism/data/splits.py", args.repo / "prism/data/transforms.py",
            args.repo / "prism/data/cached_dataset.py", args.repo / "scripts/cache_teacher_targets.py"]
        report["code_sha256"] = {p.relative_to(args.repo).as_posix(): sha256(p) for p in code_paths}
        phase = "dependencies"
        # Resume forecasts are refined from verified artifacts in cache_teachers.
        budget.check(0, setup_seconds=300)
        dependency_status = "SKIPPED"
        if not args.offline and not args.skip_install:
            dependency_status = install_dependencies(args.repo, cache, output,
                deadline=budget.started + budget.seconds)
        report["steps"][phase] = {"status": dependency_status}
        write_json(report_path, report)
        from tier1a.prepare_coco500 import prepare
        phase = "prepare_coco500"
        with exclusive(output / ".prepare.lock"):
            report["steps"][phase] = prepare(output, cache, ledger, working=args.working, offline=args.offline)
        report["ledger_sha256"] = report["steps"][phase]["ledger_sha256"]
        write_json(report_path, report)
        # No model import before the source ledger has been frozen.
        from tier1a.cache_teachers import cache_teachers
        phase = "cache_teachers"
        report["steps"][phase] = cache_teachers(output, cache / "weights", ledger,
            working=args.working, budget=budget, offline=args.offline)
        report["teacher_cache_stats"] = report["steps"][phase]["per_teacher"]
        check_storage(args.working)
        report.update(status="PASS", step1_complete=True)
    except Exception as exc:
        report.update(status="FAIL", step1_complete=False, failing_step=getattr(exc, "step", phase), reason=str(exc))
        report["steps"][phase] = {"status": "FAIL", "reason": str(exc)}
        print(json.dumps(report), file=sys.stderr, flush=True)
    finally:
        try:
            report["artifacts"] = artifact_manifest(output, ledger)
        except Exception as exc:
            report.update(status="FAIL", step1_complete=False, failing_step="artifact_manifest", reason=str(exc))
        finally:
            if budget:
                budget.finish(report["status"])
                # Accounting now includes artifact verification. Refresh its one manifest row.
                account = output / "teachers/gpu_hours.json"
                for row in report.get("artifacts", []):
                    if row["path"] == str(account):
                        row.update(bytes=account.stat().st_size, sha256=hashlib.sha256(account.read_bytes()).hexdigest())
        report["elapsed_seconds"] = time.monotonic() - started
        report["cpu_process_hours"] = (time.process_time() - cpu_started) / 3600
        for name in ("prepare_coco500", "cache_teachers"):
            report["steps"].setdefault(name, {"status": "NOT_RUN"})
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        locks.close()
    print(json.dumps({"status": report["status"], "report": str(report_path)}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
