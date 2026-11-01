#!/usr/bin/env python3
"""Tier 1a Step 2: train-only, single-encode oracle proposals at intra CRF20.

Council visibility: per-image AP50 follows oracle_detector.py's exact convention.
J = AP50 + lambda_R * R0/R is MAXIMIZED; R0 and R are whole-stream bits, including
headers. This user-requested utility differs from inherited CONFIG C's minimized
distortion + lambda_R * R/R0. Zero is candidate 0, encoded/scored only once.
The first 10 frames are real K=8 work, not extra trials. If the conservative full
projection exceeds six hours, later frames use candidates 0..5 (K=6); pilot K=8
labels are retained. K=6 is a recorded exception to CONFIG C's K>=8, never hidden.
The 10-frame planning estimate does not satisfy CONFIG F's 200-image measurement;
an updated 200-image estimate is recorded when available. No raw YUV disk writes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from prism.data.splits import SplitLedger
from prism.training.manifest import EncodeManifest
from prism.training.oracle_sweep import rank_task_candidates, validate_candidate_maps
from tier1a.common import check_storage, exclusive, read_json, sha256, write_json
from tier1a.oracle_candidates import NAMES, RECIPE, cached_candidates, candidates_hash

REPO = Path(__file__).resolve().parents[1]
COUNCIL = [
    "D: per-image AP50 excludes classes without evaluable GT; see oracle_detector docstring",
    "J: maximize AP50 + lambda_R * R0/R; differs from CONFIG C distortion + lambda_R * R/R0",
    "DINO: Step 1 has no attention; use explicitly named patch-to-CLS cosine saliency proxy",
    "Candidate h interpreted as foreground -4/background +1; unknown crop region gets zero",
    "K=6 fallback violates inherited K>=8; retained K=8 pilot labels create mixed K if triggered",
    "10-frame planning pilot requested; CONFIG F 200-image throughput prerequisite remains visible",
    "Detector supervision is YOLO11-L on new reconstructions; no DINO/SAM reconstruction rescoring",
]


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def cpu_seconds() -> float:
    usage = os.times()
    return usage.user + usage.system + usage.children_user + usage.children_system


class FrameSweep:
    """Durable per-frame state, also usable with an offline stub encoder/scorer.

    Persist RUNNING before every encode. Matching PASS and FAIL entries are both
    idempotent; an interrupted entry becomes FAIL without retry. Changed hashes
    are explicit oracle_refresh work, with old entries retained in history.
    """
    def __init__(self, output: Path, ledger: SplitLedger, manifest: EncodeManifest,
                 contract: dict, *, lambda_rate: float = .05):
        self.output, self.ledger, self.manifest = output, ledger, manifest
        self.lambda_rate = lambda_rate
        self.contract_sha = digest_json(dict(contract, lambda_rate=lambda_rate))
        self.path = output / "oracle_labels.json"
        self.data = read_json(self.path) if self.path.exists() else {
            "schema_version": 1, "ledger_sha256": ledger.digest, "contract": contract,
            "contract_sha256": self.contract_sha, "lambda_R": lambda_rate,
            "objective": "maximize D + lambda_R * R0/R (whole-bitstream bits)",
            "labels": {}, "history": [], "effective_k": 8, "council_visibility": COUNCIL}
        if (self.data["ledger_sha256"] != ledger.digest
                or self.data["contract_sha256"] != self.contract_sha):
            raise ValueError("Resume contract changed; use a separate output directory")
        self.flush()

    def flush(self) -> None:
        write_json(self.path, self.data)

    def process(self, frame: dict, payload: bytes, maps: np.ndarray, encoder, scorer, *,
                smoke: bool = False, before_candidate=lambda: None) -> tuple[dict, bool]:
        from PIL import Image
        sample_id = frame["sample_id"]
        entry = self.ledger.require(sample_id, "train")
        if entry.source != "train2017" or frame["source_sha256"] != entry.sha256:
            raise ValueError("Require source-bound train2017 frames")
        validate_candidate_maps(maps, distinct=False)
        if len(maps) not in (6, 8) or np.any(maps[0]):
            raise ValueError("Require K=8 (or recorded K=6 scope) with zero-map candidate 0")
        candidate_sha = candidates_hash(maps)
        input_sha = hashlib.sha256(payload).hexdigest()
        binding = digest_json({"candidates": candidate_sha, "input": input_sha,
                               "geometry": frame["geometry"], "contract": self.contract_sha})
        old = self.data["labels"].get(sample_id)
        if old and old.get("binding_sha256") == binding:
            if old["status"] == "RUNNING":
                old.update(status="FAIL", reason="Interrupted frame; no candidate retried", winner_map_index=None)
                for row in old["candidates"]:
                    if row["status"] == "RUNNING":
                        row.update(status="FAIL", reason="Interrupted encode/decode/score")
                self.flush()
            if old["status"] == "PASS":
                winner = Path(old["winner_png"])
                if not winner.is_file() or sha256(winner) != old["winner_png_sha256"]:
                    old.update(status="FAIL", reason="Winner PNG missing/corrupt; no encode retried",
                               winner_map_index=None)
                    self.flush()
            return old, True
        if old:
            self.data["history"].append(old)
        row = {"sample_id": sample_id, "image_id": frame["image_id"], "status": "RUNNING",
               "source_sha256": entry.sha256, "input_yuv_sha256": input_sha,
               "binding_sha256": binding, "candidates_sha256": candidate_sha,
               "candidate_count": len(maps), "lambda_R": self.lambda_rate,
               "candidates": [{"index": i, "name": NAMES[i], "status": "NOT_ATTEMPTED",
                    "map_sha256": hashlib.sha256(m.tobytes()).hexdigest(), "D": None,
                    "J": None, "bits": None, "frame_bits": None, "decode_ok": False}
                    for i, m in enumerate(maps)], "winner_map_index": None,
               "runner_up_map_index": None, "candidate_provenance": frame.get("candidate_provenance", {})}
        self.data["labels"][sample_id] = row
        self.flush()
        directory = self.output / "frames" / str(frame["image_id"]) / binding[:16]
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "candidate_maps.npz", maps=maps)
        row["candidate_maps_path"] = str(directory / "candidate_maps.npz")
        started = time.monotonic()
        for index, offsets in enumerate(maps):
            result = row["candidates"][index]
            candidate_dir = directory / f"candidate_{index}"
            try:
                before_candidate()
                result["status"] = "RUNNING"
                self.flush()
                self.manifest.record("oracle_refresh" if old else "oracle_initial", sample_id,
                    candidate=index, candidates_sha256=candidate_sha, lambda_R=self.lambda_rate)
                attempt_started = time.monotonic()
                try:
                    result.update(encoder(payload, offsets, frame, candidate_dir))
                finally:
                    result["encode_attempt_seconds"] = time.monotonic() - attempt_started
                # Never infer CSV frame bits from stream bytes: both are required.
                if (result["bits"] <= 0 or result["frame_bits"] <= 0
                        or result["whole_bitstream_bytes"] * 8 != result["bits"]):
                    raise ValueError("Invalid measured whole-stream/per-frame bits")
                if smoke and index == 0:
                    identity = self.data.get("zero_stock_smoke")
                    if identity is None:
                        self.data["zero_stock_smoke"] = {"status": "RUNNING", "sample_id": sample_id}
                        self.flush()
                        self.manifest.record("gate0", sample_id, check="first-frame zero vs stock; one stock encode")
                        stock = encoder(payload, None, frame, directory / "stock", stock=True)
                        identity = {"status": "PASS" if stock["bitstream_sha256"] == result["bitstream_sha256"]
                                    else "FAIL", "sample_id": sample_id,
                                    "zero_sha256": result["bitstream_sha256"],
                                    "stock_sha256": stock["bitstream_sha256"], "stock_encode": stock}
                        self.data["zero_stock_smoke"] = identity
                    if identity["status"] != "PASS":
                        raise ValueError("First-frame zero-map vs stock identity failed/interrupted")
                rgb = encoder.decode(result, frame)
                if rgb.shape != (frame["geometry"]["height"], frame["geometry"]["width"], 3):
                    raise ValueError("Decoded geometry mismatch")
                result["decode_ok"] = True
                png = candidate_dir / "decoded.png"
                png.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(png)
                result["decoded_png"] = str(png)
                result["decoded_png_sha256"] = sha256(png)
                score_started = time.monotonic()
                result["D"] = float(scorer(rgb, frame))
                result["score_seconds"] = time.monotonic() - score_started
                if not math.isfinite(result["D"]) or not 0 <= result["D"] <= 1:
                    raise ValueError("Detector D must be finite in [0,1]")
                result["status"] = "PASS"
            except Exception as exc:
                result.update(status="FAIL", reason=f"{type(exc).__name__}: {exc}")
                # JSON never serializes NaN or infinity, even in failed measurements.
                if result.get("D") is not None and not math.isfinite(result["D"]):
                    result["D"] = None
            self.flush()
            if smoke and index == 0 and self.data.get("zero_stock_smoke", {}).get("status") != "PASS":
                break
        valid = [c["status"] == "PASS" for c in row["candidates"]]
        if all(valid):
            objectives, order = rank_task_candidates([c["D"] for c in row["candidates"]],
                [c["bits"] for c in row["candidates"]], self.lambda_rate)
            for c, value in zip(row["candidates"], objectives):
                c["J"] = value
            winner = row["candidates"][order[0]]
            winner_path = self.output / "winners" / f"{frame['image_id']}.png"
            winner_path.parent.mkdir(parents=True, exist_ok=True)
            # Candidate PNG is already a decoded reconstruction; no extra encode.
            import shutil
            shutil.copyfile(winner["decoded_png"], winner_path)
            row.update(status="PASS", winner_map_index=order[0], runner_up_map_index=order[1],
                       J_table=objectives, R0=row["candidates"][0]["bits"],
                       winner_png=str(winner_path), winner_png_sha256=sha256(winner_path))
        else:
            r0 = row["candidates"][0].get("bits")
            if r0:
                row["R0"] = r0
                for c in row["candidates"]:
                    if c["status"] == "PASS":
                        c["J"] = c["D"] + self.lambda_rate * r0 / c["bits"]
            row.update(status="FAIL", reason="One or more candidates failed; no partial oracle winner",
                       J_table=[c["J"] for c in row["candidates"]])
        row["wall_seconds"] = time.monotonic() - started
        row["encode_seconds"] = sum(c.get("encode_seconds", 0) for c in row["candidates"])
        self.flush()
        # PNGs are materialized for every reconstruction, then only the winner
        # is retained. Thousands of 1080p PNGs would violate the 16 GB workspace.
        for candidate in row["candidates"]:
            if candidate.get("decoded_png"):
                Path(candidate["decoded_png"]).unlink(missing_ok=True)
                candidate["decoded_png_retained"] = False
        self.flush()
        return row, False


def projection(rows: list[dict], *, total: int = 500, safety: float = 1.25) -> dict:
    measured = [r for r in rows if r.get("status") == "PASS" and r.get("wall_seconds") and r.get("candidate_count")
                and all(c.get("encode_seconds") is not None for c in r["candidates"])]
    if not measured:
        raise ValueError("No fully measured frame timings; refusing invented projection")
    per_candidate = sum(r["wall_seconds"] for r in measured) / sum(r["candidate_count"] for r in measured)
    encode_per_candidate = sum(r["encode_seconds"] for r in measured) / sum(r["candidate_count"] for r in measured)
    return {"measured_frames": len(measured), "safety_factor": safety,
            "encode_seconds_per_frame_k8": 8 * encode_per_candidate,
            "end_to_end_seconds_per_candidate": per_candidate,
            "projected_500x8_encode_hours": total * 8 * encode_per_candidate / 3600,
            "projected_500x8_wall_hours": total * 8 * per_candidate * safety / 3600,
            "projected_500x6_wall_hours": total * 6 * per_candidate * safety / 3600,
            "config_f_200_image_measurement": len(measured) >= 200}


def choose_scope(forecast: dict, setup_seconds: float, wall_hours: float) -> int:
    """Only scope change authorized by this kernel: K8 -> K6 after the pilot."""
    return 6 if forecast["projected_500x8_wall_hours"] + setup_seconds / 3600 > wall_hours - 90 / 3600 else 8


def load_inputs(root: Path, ledger_path: Path, annotations: Path, expected_sha: str) -> tuple[SplitLedger, list[dict], str]:
    ledger = SplitLedger.read(ledger_path)
    selection_path = root / "coco500_selection.json"
    selection = read_json(selection_path)
    manifest = read_json(root / "coco500/manifest.json")
    if (not ledger.frozen or len(ledger.entries) != 500 or not ledger.digest.startswith(expected_sha)
            or selection["source"] != "train2017" or manifest["ledger_sha256"] != ledger.digest
            or sha256(annotations) != selection["annotations_sha256"]):
        raise ValueError("Require frozen COCO500 train carve-out and original full annotation SHA")
    rows = manifest["images"]
    if (len(rows) != 500 or {r["sample_id"] for r in rows} != set(ledger.by_id)
            or len(selection["images"]) != 500
            or {r["sample_id"] for r in selection["images"]} != set(ledger.by_id)):
        raise ValueError("Selection, manifest and ledger identities disagree")
    for row in rows:
        entry = ledger.require(row["sample_id"], "train")
        if entry.source != "train2017" or row["source_sha256"] != entry.sha256:
            raise ValueError("Manifest contains a non-training or mismatched source")
    return ledger, sorted(rows, key=lambda r: r["image_id"]), sha256(selection_path)


def load_frame(root: Path, row: dict, ledger: SplitLedger, selection_sha: str) -> tuple[dict, bytes, np.ndarray]:
    from tier1a.yuv import load_yuv
    path = root / "coco500/yuv" / f"{row['image_id']}.yuv.gz"
    meta = read_json(path.with_name(path.name + ".json"))
    if (meta != row["yuv_metadata"] or meta["source_sha256"] != row["source_sha256"]
            or sha256(path) != meta["sha256"]):
        raise ValueError("YUV metadata/source/compressed SHA mismatch")
    geometry = meta["geometry"]
    if (geometry["width"], geometry["height"], geometry["matrix"], geometry["range"]) != (1920, 1080, "bt709", "limited"):
        raise ValueError("Require Step 1 1080p BT.709 limited-range geometry")
    payload = load_yuv(path)
    if hashlib.sha256(payload).hexdigest() != geometry["raw_sha256"]:
        raise ValueError("Raw input SHA mismatch")
    maps, provenance = cached_candidates(root / "teachers", row["sample_id"], geometry,
        source_sha=row["source_sha256"], ledger_sha=ledger.digest, selection_sha=selection_sha)
    return dict(row, geometry=geometry, candidate_provenance=provenance), payload, maps


def ensure_stack(gate: Path, *, build: bool, deadline: float) -> dict:
    ready = (gate / "stack.json").is_file()
    if ready:
        stack = read_json(gate / "stack.json")
        ready = all(Path(stack[k]).is_file() for k in ("library", "shim", "stock_library", "stock_shim"))
        ready = ready and all((gate / name).is_file() for name in ("build-pin.json", "stock-pin.json"))
    if not ready:
        if not build:
            raise FileNotFoundError("No local GATE-0 stack; --no-build forbids downloads/build")
        env = dict(os.environ, GATE0_JOBS="4", GATE0_PYTHON=sys.executable)
        subprocess.run(["bash", str(gate / "build_gate0_stack.sh")], cwd=REPO, env=env,
                       check=True, timeout=max(.01, deadline - time.monotonic()))
    return read_json(gate / "stack.json")


def codec_dry_run(args) -> int:
    """Two synthetic 64x64 frames x three real encodes; no AP or label claims."""
    from tier1a.oracle_codec import Gate0Encoder
    from tier1a.yuv import letterbox_yuv
    from PIL import Image
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "kind": "synthetic codec-only smoke; no detector/COCO claims",
              "frames": [], "raw_yuv_disk_writes": False}
    try:
        ensure_stack(args.gate, build=False, deadline=time.monotonic() + 600)
    except FileNotFoundError as exc:
        report.update(status="SKIP", reason=str(exc), projection_500x8=None)
        write_json(destination / "codec_dry_run_report.json", report)
        print(json.dumps(report))
        return 0
    encoder = Gate0Encoder(args.gate, ffmpeg=args.ffmpeg, deadline=time.monotonic() + 600)
    try:
        for index in range(2):
            rgb = np.random.default_rng(index).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            payload, geometry = letterbox_yuv(rgb, 64, 64)
            frame = {"image_id": index, "geometry": geometry}
            table = []
            for candidate, value in enumerate((0, 3, -3)):
                result = encoder(payload, np.full((4, 4), value, dtype=np.int8), frame,
                                 destination / f"frame_{index}" / f"candidate_{candidate}")
                decoded = encoder.decode(result, frame)
                Image.fromarray(decoded).save(destination / f"frame_{index}_candidate_{candidate}.png")
                result.update(candidate=candidate, decode_ok=True)
                table.append(result)
                if index == 0 and candidate == 0:
                    stock = encoder(payload, None, frame, destination / "stock", stock=True)
                    report["zero_stock_identity"] = stock["bitstream_sha256"] == result["bitstream_sha256"]
                    if not report["zero_stock_identity"]:
                        raise ValueError("Synthetic zero-map/stock identity failed")
            report["frames"].append(table)
        report.update(status="PASS", candidate_encodes=6, stock_encodes=1,
                      projection_500x8=None, projection_note="64x64 codec smoke cannot establish 1080p detector throughput")
    except Exception as exc:
        report.update(status="FAIL", reason=f"{type(exc).__name__}: {exc}")
    write_json(destination / "codec_dry_run_report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


def report_for(sweep: FrameSweep, started: float, cpu_started: float, prior_gpu_hours: float) -> dict:
    rows = list(sweep.data["labels"].values())
    candidates = [c for r in rows for c in r.get("candidates", [])]
    return {"schema_version": 1, "status": "RUNNING", "expected_frames": 500,
        "pass_frames": sum(r["status"] == "PASS" for r in rows),
        "fail_frames": sum(r["status"] == "FAIL" for r in rows),
        "running_frames": sum(r["status"] == "RUNNING" for r in rows),
        "pending_frames": 500 - len(rows), "wall_seconds": time.monotonic() - started,
        "cpu_hours": (cpu_seconds() - cpu_started) / 3600,
        "cpu_hours_definition": "process + waited child user/system CPU time; separate from GPU ceiling",
        "gpu_hours": 0, "prior_pilot_gpu_hours": prior_gpu_hours,
        "total_pilot_gpu_hours": prior_gpu_hours, "gpu_hour_ceiling": 60,
        "encode_manifest_counts": sweep.manifest.counts(), "lambda_R": sweep.lambda_rate,
        "current_label_candidate_totals": {
            "pass": sum(c["status"] == "PASS" for c in candidates),
            "fail": sum(c["status"] == "FAIL" for c in candidates),
            "not_attempted": sum(c["status"] == "NOT_ATTEMPTED" for c in candidates),
            "decode_ok": sum(c.get("decode_ok", False) for c in candidates),
            "whole_bitstream_bytes": sum(c.get("whole_bitstream_bytes", 0) for c in candidates),
            "whole_bitstream_bits": sum(c.get("bits") or 0 for c in candidates),
            "frame_bits": sum(c.get("frame_bits") or 0 for c in candidates)},
        "ledger_sha256": sweep.ledger.digest, "effective_k": sweep.data["effective_k"],
        "k_counts": {str(k): sum(r.get("candidate_count") == k for r in rows) for k in (6, 8)},
        "pilot_projection": sweep.data.get("pilot_projection"),
        "measurement_200": sweep.data.get("measurement_200"),
        "reduction": sweep.data.get("reduction"), "zero_stock_smoke": sweep.data.get("zero_stock_smoke"),
        "council_visibility": COUNCIL,
        "honesty_checks": {"train_only": True, "raw_yuv_disk_writes": False,
            "detector_input": "every candidate decoded RGB; zero candidate reused as reference",
            "single_encode_per_candidate": True, "stock_encodes_expected": 1,
            "failed_candidates_never_win": True, "gpu_inference": False,
            "tier1a_gate_claimed": False, "zero_sum_at_crf": False}}


def run_job(args) -> int:
    from tier1a.oracle_codec import Gate0Encoder
    from tier1a.oracle_detector import CocoAP50
    started, cpu_started = time.monotonic(), cpu_seconds()
    deadline = started + args.wall_hours * 3600 - 90
    output = args.output.resolve()
    report = {"status": "FAIL", "reason": "Setup did not finish", "council_visibility": COUNCIL,
              "expected_frames": 500, "pass_frames": 0, "fail_frames": 0, "pending_frames": 500,
              "gpu_hours": 0, "prior_pilot_gpu_hours": args.prior_gpu_hours, "gpu_hour_ceiling": 60}
    sweep = None
    with exclusive(output / ".writer.lock"):
        try:
            check_storage(args.working, 100_000_000)
            stack = ensure_stack(args.gate, build=not args.no_build, deadline=deadline)
            ledger, rows, selection_sha = load_inputs(args.input, args.ledger, args.annotations, args.expected_ledger_sha)
            scorer = CocoAP50(args.annotations, args.weights, {r["image_id"] for r in rows})
            encoder = Gate0Encoder(args.gate, ffmpeg=args.ffmpeg, deadline=deadline)
            try:
                repo_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                repo_sha = "UNAVAILABLE"
            contract = {"recipe": RECIPE, "detector": scorer.provenance, "encoder": asdict(encoder.config),
                "stack": stack, "pin": encoder.pin, "stock_pin": encoder.stock_pin,
                "annotations_sha256": sha256(args.annotations), "seed": 0, "delta_x": "identity/zero",
                "code_sha256": {str(p.relative_to(REPO)): sha256(p) for p in (
                    Path(__file__), REPO / "tier1a/oracle_candidates.py", REPO / "tier1a/oracle_codec.py",
                    REPO / "tier1a/oracle_detector.py", REPO / "prism/training/oracle_sweep.py",
                    REPO / "prism/codec/x265_bridge.py", REPO / "prism/codec/ffmpeg_stream.py",
                    REPO / "tier1a/yuv.py", REPO / "tier1a/common.py")},
                "config_sha256": {name: sha256(REPO / "docs" / name) for name in ("CONFIG_V1_3.md", "CONFIG_V1_4.md")}}
            manifest = EncodeManifest(output / "encode_manifest.jsonl", repo_sha=repo_sha,
                config_sha=digest_json(contract), build=encoder.pin, seed=0)
            sweep = FrameSweep(output, ledger, manifest, contract, lambda_rate=args.lambda_rate)
            pilot_rows = []
            for number, row in enumerate(rows):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Job allowance exhausted; remaining frames not attempted")
                check_storage(args.working, 90_000_000)
                def before_candidate():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Deadline before candidate; no encode attempted")
                try:
                    frame, payload, maps = load_frame(args.input, row, ledger, selection_sha)
                    # Freeze each frame's original K on resume, including the K=8 pilot.
                    prior = sweep.data["labels"].get(row["sample_id"], {})
                    k = prior.get("candidate_count", 8 if number < 10 else sweep.data["effective_k"])
                    result, skipped = sweep.process(frame, payload, maps[:k], encoder, scorer,
                                                    smoke=number == 0, before_candidate=before_candidate)
                except Exception as exc:
                    result = {"sample_id": row["sample_id"], "image_id": row["image_id"], "status": "FAIL",
                              "reason": f"{type(exc).__name__}: {exc}", "winner_map_index": None,
                              "candidates": [], "phase": "input_or_frame_processing"}
                    old = sweep.data["labels"].get(row["sample_id"])
                    if old:
                        sweep.data["history"].append(old)
                    sweep.data["labels"][row["sample_id"]] = result
                    sweep.flush()
                    skipped = False
                pilot_rows.append(result)
                if number == 0 and sweep.data.get("zero_stock_smoke", {}).get("status") != "PASS":
                    raise ValueError("First-frame stock identity precondition failed; remaining encodes blocked")
                if number == 9 and not sweep.data.get("pilot_projection"):
                    forecast = projection(pilot_rows)
                    forecast["setup_seconds"] = max(0, time.monotonic() - started - sum(r.get("wall_seconds", 0) for r in pilot_rows))
                    sweep.data["pilot_projection"] = forecast
                    if choose_scope(forecast, forecast["setup_seconds"], args.wall_hours) == 6:
                        sweep.data["effective_k"] = 6
                        sweep.data["reduction"] = {"from": 8, "to": 6, "after_frames": 10,
                            "reason": "Measured K8 projection exceeds job allowance", "dropped_indices": [6, 7],
                            "pilot_k8_retained": True, "config_c_k_min_exception": True}
                    sweep.flush()
                if number == 199:
                    sweep.data["measurement_200"] = projection(pilot_rows)
                    sweep.flush()
                report = report_for(sweep, started, cpu_started, args.prior_gpu_hours)
                write_json(output / "oracle_sweep_report.json", report)
                print(json.dumps({"frame": number + 1, "status": result["status"], "skipped": skipped,
                                  "K": result.get("candidate_count"), "wall_seconds": report["wall_seconds"]}), flush=True)
                if number >= 9:
                    estimate = projection(pilot_rows)
                    pending = sum(r["sample_id"] not in sweep.data["labels"] for r in rows[number + 1:])
                    remaining = pending * sweep.data["effective_k"] * estimate["end_to_end_seconds_per_candidate"] * 1.25
                    if time.monotonic() + remaining > deadline:
                        raise TimeoutError("Even the active K scope projects beyond remaining job allowance; resume later")
            report = report_for(sweep, started, cpu_started, args.prior_gpu_hours)
            report["status"] = "PASS" if report["pass_frames"] == 500 else "FAIL"
        except Exception as exc:
            if sweep is not None:
                report = report_for(sweep, started, cpu_started, args.prior_gpu_hours)
            report.update(status="FAIL", reason=f"{type(exc).__name__}: {exc}")
        finally:
            report.update(wall_seconds=time.monotonic() - started,
                          cpu_hours=(cpu_seconds() - cpu_started) / 3600)
            write_json(output / "oracle_sweep_report.json", report)
    return 0 if report["status"] == "PASS" else 1


def supervise(args) -> int:
    """Linux process-group watchdog covers builder, model calls and child codecs."""
    if sys.platform != "linux":
        raise RuntimeError("Kernel execution requires Linux; offline tests run on Windows too")
    gpu_account = args.input / "teachers/gpu_hours.json"
    prior_account = read_json(gpu_account) if gpu_account.exists() else {}
    inherited_gpu = prior_account.get("prior_gpu_hours", 0) + prior_account.get("gpu_hours", 0)
    inherited_gpu += prior_account.get("active", {}).get("reserved_gpu_hours", 0)
    if args.prior_gpu_hours is None:
        if not prior_account:
            raise ValueError("Supply --prior-gpu-hours or preserve Step 1 teachers/gpu_hours.json")
        args.prior_gpu_hours = inherited_gpu
    if args.prior_gpu_hours < inherited_gpu:
        raise ValueError("Prior GPU usage cannot be below the Step 1 cumulative account")
    if not (0 < args.wall_hours <= 6 and math.isfinite(args.prior_gpu_hours)
            and 0 <= args.prior_gpu_hours <= 60 and math.isfinite(args.lambda_rate) and args.lambda_rate >= 0):
        raise ValueError("Invalid wall/GPU/lambda limits")
    # CPU inference does not make an allocated accelerator free. Require a CPU
    # kernel so the reported zero GPU-hour charge is honest.
    try:
        devices = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                         stderr=subprocess.DEVNULL, timeout=10, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        devices = ""
    if devices:
        raise RuntimeError("Select a CPU-only kernel (accelerator None) for this CPU-accounted step")
    args.output.mkdir(parents=True, exist_ok=True)
    with exclusive(args.output / ".supervisor.lock"):
        account_path = args.output / "cpu_hours.json"
        account = read_json(account_path) if account_path.exists() else {"cpu_hours": 0, "wall_hours": 0, "jobs": []}
        if account.get("active"):
            active = account.pop("active")
            account["cpu_hours"] += active["reserved_cpu_hours"]
            account["wall_hours"] += active["reserved_wall_hours"]
            account["jobs"].append(dict(active, status="INTERRUPTED_CHARGED_RESERVATION"))
        if args.prior_gpu_hours < account.get("prior_pilot_gpu_hours", 0):
            raise ValueError("Prior GPU usage cannot decrease")
        account["prior_pilot_gpu_hours"] = args.prior_gpu_hours
        account["active"] = {"reserved_cpu_hours": 4 * args.wall_hours, "reserved_wall_hours": args.wall_hours}
        write_json(account_path, account)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="4", YOLO_AUTOINSTALL="false",
                   YOLO_CONFIG_DIR=str(args.output / "yolo_config"))
        started, cpu_started = time.monotonic(), cpu_seconds()
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                   "--prior-gpu-hours", str(args.prior_gpu_hours), "--run-job"]
        proc = subprocess.Popen(command, env=env, start_new_session=True)
        timed_out = False
        try:
            code = proc.wait(timeout=max(1., args.wall_hours * 3600 - 60))
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code = 2
        elapsed, measured_cpu = time.monotonic() - started, (cpu_seconds() - cpu_started) / 3600
        # Reaped child accounting can omit grandchildren orphaned by a group
        # kill. Charge the 4-vCPU elapsed allowance in that case, explicitly.
        cpu = max(measured_cpu, 4 * elapsed / 3600) if timed_out else measured_cpu
        report_path = args.output / "oracle_sweep_report.json"
        report = read_json(report_path) if report_path.exists() else {"status": "FAIL", "reason": "Job exited before report"}
        if timed_out:
            report.update(status="FAIL", reason="Supervisor deadline/interrupt; process group terminated")
        if code:
            report["status"] = "FAIL"
        labels_path = args.output / "oracle_labels.json"
        if labels_path.exists():
            data = read_json(labels_path)
            for row in data["labels"].values():
                if row["status"] == "RUNNING":
                    row.update(status="FAIL", reason="Interrupted by supervisor; no retries", winner_map_index=None)
                    for candidate in row["candidates"]:
                        if candidate["status"] == "RUNNING":
                            candidate.update(status="FAIL", reason="Interrupted by supervisor")
            write_json(labels_path, data)
            rows = list(data["labels"].values())
            report.update(pass_frames=sum(r["status"] == "PASS" for r in rows),
                          fail_frames=sum(r["status"] == "FAIL" for r in rows), pending_frames=500-len(rows), running_frames=0)
        account.pop("active")
        account["cpu_hours"] += cpu
        account["wall_hours"] += elapsed / 3600
        account["jobs"].append({"status": report["status"], "cpu_hours": cpu,
                                "measured_cpu_hours": measured_cpu, "wall_seconds": elapsed,
                                "conservative_termination_charge": timed_out})
        write_json(account_path, account)
        report.update(wall_seconds=elapsed, cpu_hours=cpu, cumulative_cpu_hours=account["cpu_hours"],
                      gpu_hours=0, prior_pilot_gpu_hours=args.prior_gpu_hours, gpu_hour_ceiling=60,
                      cumulative_cpu_account=str(account_path), measured_cpu_hours=measured_cpu,
                      cpu_accounting_method="4-vCPU elapsed allowance after termination" if timed_out
                      else "process + waited child user/system time")
        write_json(report_path, report)
        write_json(args.output.parent / "step2_report.json", report)
        print(json.dumps(report, indent=2))
        # A stopped child cannot still own the lock after wait().
        if timed_out:
            (args.output / ".writer.lock").unlink(missing_ok=True)
        return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/kaggle/working/tier1a"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/tier1a/oracle"))
    parser.add_argument("--ledger", type=Path, default=REPO / "prism/data/splits/coco500_ledger.json")
    parser.add_argument("--annotations", type=Path, default=Path("/kaggle/working/cache/instances_train2017.json"))
    parser.add_argument("--weights", type=Path, default=Path("/kaggle/working/cache/weights/yolo11l.pt"))
    parser.add_argument("--expected-ledger-sha", default="d96dde11", help="Full SHA recommended; supplied prefix is checked")
    parser.add_argument("--gate", type=Path, default=REPO / "gate0")
    parser.add_argument("--working", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--lambda-rate", type=float, default=.05)
    parser.add_argument("--wall-hours", type=float, default=6.)
    parser.add_argument("--prior-gpu-hours", type=float, help="Cumulative pilot GPU usage; default inherits Step 1 account")
    parser.add_argument("--no-build", action="store_true", help="Require provisioned GATE-0 binaries; no build/network")
    parser.add_argument("--codec-dry-run", action="store_true", help="2x3 synthetic codec-only local smoke, no build/network")
    parser.add_argument("--encode-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-job", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.encode_worker:
        from tier1a.oracle_codec import worker
        return worker(args.encode_worker)
    if args.run_job:
        return run_job(args)
    if args.codec_dry_run:
        return codec_dry_run(args)
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
