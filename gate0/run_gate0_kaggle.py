#!/usr/bin/env python3
"""Build and run GATE 0 on Ubuntu; failed builds produce SKIP, never PASS."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "gate0"


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_logged(command: list[str], env: dict[str, str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("ARGV " + json.dumps(command) + "\n")
        log.flush()
        with subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as process:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()


def build_failure(log_path: Path, returncode: int, detail: str = "") -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    failed = re.findall(r"\[GATE0 FAIL\] step=(\S+) exit=(\d+) line=(\d+)", text)
    steps = re.findall(r"\[GATE0 STEP\] (\S+)", text)
    step = failed[-1][0] if failed else (steps[-1] if steps else "launch_build_gate0_stack")
    tail = "\n".join(text.splitlines()[-60:])[-12000:]
    return {"status": "SKIP", "gate0_complete": False, "failing_step": step,
            "exit_code": returncode, "log_tail": tail,
            "reason": f"Build step {step} exited {returncode}. {detail}\nLog tail:\n{tail}"}


def collect_and_calibrate(evidence: Path, run_dir: Path, env: dict[str, str],
                         log: Path, commands: dict) -> None:
    """Separate outcome: measurement errors never rewrite the frozen verdict."""
    collection = [sys.executable, str(REPO / "scripts/spike_gate0.py"),
                  "--output", str(evidence), "--pin", str(GATE / "build-pin.json"),
                  "--collect-telemetry-only"]
    proposal = evidence / "band_calibration_proposal.json"
    calibration = [sys.executable, str(GATE / "calibrate_bands.py"), str(evidence),
                   "--output", str(proposal)]
    commands.update(telemetry_collection=collection, calibration=calibration)
    status = {"measurement_only": True, "status": "ERROR"}
    try:
        for name in ("band_calibration_proposal.json", "band_calibration_proposal.net_effects.csv"):
            (GATE / name).unlink(missing_ok=True)
        status["collection_exit_code"] = run_logged(collection, env, log)
        status["calibration_exit_code"] = run_logged(calibration, env, log)
        if status["calibration_exit_code"] == 0 and proposal.is_file():
            status.update(status="PROPOSAL", proposal=str(proposal))
            # Latest local alias is informational only. The run retains its copy.
            shutil.copyfile(proposal, GATE / proposal.name)
            sidecar = proposal.with_suffix(".net_effects.csv")
            shutil.copyfile(sidecar, GATE / sidecar.name)
    except Exception as exc:
        status["error"] = str(exc)
    write_json(run_dir / "calibration_status.json", status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/gate0_output"))
    parser.add_argument("--quick", action="store_true", default=os.environ.get("GATE0_QUICK") == "1",
                        help="Only map 0 and +6 at QP 32/GOP 10; success is SMOKE_PASS")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = output / "runs" / run_id
    run_dir.mkdir(parents=True)
    write_json(output / "gate0.json", {"status": "RUNNING", "gate0_complete": False, "run_id": run_id})
    # These are runner-owned aliases for the current run; histories stay in runs/.
    for name in ("build-pin.json", "stock-pin.json", "stack.json", "build_log.txt"):
        (output / name).unlink(missing_ok=True)
    env = os.environ.copy()
    env["GATE0_PYTHON"] = sys.executable
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("GATE0_RC", "CRF")  # CONFIG v1.4 primary mode; retain explicit overrides.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(GATE / "build/python-deps"),
                                                    str(REPO), env.get("PYTHONPATH")]))
    build_command = ["bash", str(GATE / "build_gate0_stack.sh")]
    commands = {"build": build_command, "quick": args.quick}
    driver_log = run_dir / "run_log.txt"
    build_log = run_dir / "build_log.txt"
    phase = "build"
    code = 1
    try:
        try:
            code = run_logged(build_command, env, driver_log)
        finally:
            # Driver output also covers failures before bash could open its log.
            if driver_log.exists():
                shutil.copyfile(driver_log, build_log)
                shutil.copyfile(build_log, output / "build_log.txt")
        if code:
            report = build_failure(build_log, code)
        else:
            for name in ("build-pin.json", "stock-pin.json", "stack.json",
                         "x265_residual_patch.patch", "x265_residual_telemetry.patch", "PATCH_AUDIT.md"):
                shutil.copyfile(GATE / name, run_dir / name)
                shutil.copyfile(GATE / name, output / name)
            stack = json.loads((run_dir / "stack.json").read_text(encoding="utf-8"))
            env["PRISM_X265_SHIM"] = stack["shim"]
            env["PRISM_X265_LIBRARY"] = stack["library"]
            env["LD_LIBRARY_PATH"] = str(Path(stack["library"]).parent)
            env.pop("PRISM_X265_QP_TRACE", None)
            env.pop("PRISM_X265_RC_TELEMETRY", None)  # workers set individual evidence paths
            # The harness sets these same two env vars to stock paths inside a
            # fresh stock worker, preventing same-SONAME loader aliasing.
            reference = [stack["reference_decoder"], "-b", "{bitstream}"]
            evidence = run_dir / "evidence"
            command = [sys.executable, str(REPO / "scripts/spike_gate0.py"),
                       "--output", str(evidence),
                       "--pin", str(GATE / "build-pin.json"),
                       "--stock-pin", str(GATE / "stock-pin.json"),
                       "--stock-shim", stack["stock_shim"],
                       "--stock-library", stack["stock_library"],
                       "--reference-command-json", json.dumps(reference)]
            if args.quick:
                command.append("--quick")
            commands.update({"spike": command, "environment": {
                "PRISM_X265_SHIM": stack["shim"], "PRISM_X265_LIBRARY": stack["library"],
                "GATE0_RC": env["GATE0_RC"], "GATE0_CRF": env.get("GATE0_CRF", "20"),
                "PRISM_X265_RC_TELEMETRY": "<evidence>/qp<QP>_gop<GOP>_<map>.telemetry.csv"},
                "stock_worker_environment": {"PRISM_X265_SHIM": stack["stock_shim"],
                                             "PRISM_X265_LIBRARY": stack["stock_library"]}})
            write_json(run_dir / "commands.json", commands)
            phase = "spike"
            code = run_logged(command, env, driver_log)
            report_path = evidence / "gate0.json"
            if not report_path.is_file():
                report = {"status": "FAIL", "gate0_complete": False,
                          "reason": f"Spike exited {code} without a report; see {driver_log}"}
                code = code or 1
            else:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if code and report.get("status") in {"PASS", "SMOKE_PASS"}:
                    report.update(status="FAIL", gate0_complete=False,
                                  reason=f"Spike exited {code} despite its success report")
                if report.get("status") not in {"PASS", "SMOKE_PASS"}:
                    code = code or 1
                if args.quick and report.get("status") == "PASS":
                    report.update(status="FAIL", gate0_complete=False,
                                  reason="Quick mode cannot report full GATE 0 PASS")
                    code = code or 1
            if not args.quick:
                collect_and_calibrate(evidence, run_dir, env, driver_log, commands)
    except Exception as exc:
        code = code or 1
        report = (build_failure(build_log, code, str(exc)) if phase == "build" else
                  {"status": "FAIL", "gate0_complete": False, "reason": f"{phase}: {exc}"})
    report.update(run_id=run_id, artifact_directory=str(run_dir), quick=args.quick,
                  exit_code=code, patch_review_required=True)
    write_json(run_dir / "commands.json", commands)
    write_json(run_dir / "gate0.json", report)
    write_json(output / "gate0.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
