"""Publication outputs and offline coverage of the Kaggle harvesting lifecycle."""
from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import auto_harvester as harvester
from scripts import generate_paper_artifacts as paper
from scripts import merge_pool_5k as merger

REPO = Path(__file__).resolve().parents[1]


def final_report():
    """Monotone curves with a known 20% rate reduction, over three seeds."""
    metrics = {}
    for det in paper.DETECTORS:
        metrics[det] = {}
        for model in paper.MODELS:
            for seed in ([None] if model == "M1" else [0, 1, 2]):
                for i, crf in enumerate(paper.CRFS):
                    key = f"{model}_crf{crf}" if seed is None else f"{model}_seed{seed}_crf{crf}"
                    metrics[det][key] = {
                        "bytes": (100_000 - i * 15_000) * (1 if model == "M1" else 0.8),
                        "ap_50_95": 0.6 - i * 0.06, "ap50": 0.8 - i * 0.05,
                        "ap_small": 0.4 - i * 0.04,
                    }
    return {"summary": {"total_unique_images": 5000, "shards_found": 19}, "results": {"metrics": metrics}}


def assert_artifacts(directory):
    for name in paper.ARTIFACT_NAMES:
        data = (directory / name).read_bytes()
        assert len(data) > 100
        if name.endswith(".png"):
            assert data.startswith(b"\x89PNG\r\n\x1a\n")
        elif name.endswith(".pdf"):
            assert data.startswith(b"%PDF-")
        else:
            assert b"\\toprule" in data and b"\\bottomrule" in data


def test_artifacts_cli_on_real_step3_report(tmp_path):
    output = tmp_path / "paper artifacts"
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts/generate_paper_artifacts.py"),
         "--report", str(REPO / "docs/reports/step3_report.json"), "--output-dir", str(output)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert_artifacts(output)
    table = (output / "table_m1_m4_coco.tex").read_text()
    assert "50 images" in table and "9,830,883" in table
    assert "59.40 & 74.98" in table  # Pooled AP, not mean per-image AP50.
    assert r"\Delta_x=0" in table
    for model in paper.MODELS:
        label = r"\textbf{M4}" if model == "M4" else model
        for crf in paper.CRFS:
            assert f"{label} & {crf} &" in table


def test_tier1b_artifacts_and_known_bd_reduction(tmp_path):
    report = final_report()
    curves = paper.normalize_report(report)
    rates = paper.compute_bd_rates(curves, report)
    for det in paper.DETECTORS:
        for model in paper.MODELS[1:]:
            assert list(rates[det][model].values()) == pytest.approx([-20] * 4, abs=1e-7)
    report["results"]["bd_rates"] = {
        det: {model: {"mean": values} for model, values in rows.items()} for det, rows in rates.items()
    }
    report["results"]["bd_rates"]["yolo11l"]["M4"]["mean"]["bjontegaard_bd_rate_map"] = None
    path = tmp_path / "tier1b_val5k_final_report.json"
    path.write_text(json.dumps(report))
    paper.generate_artifacts(path, tmp_path / "figures")
    assert_artifacts(tmp_path / "figures")
    table = (tmp_path / "figures/table_m1_m4_coco.tex").read_text()
    assert "5,000 images" in table
    assert r"\textemdash / 20.00" in table  # Recorded invalid BD must stay invalid.
    assert "20.00 / 20.00" in table  # Reductions have the opposite sign to BD-rate.


@pytest.mark.parametrize("case, disclosed", [("disabled", True), ("enabled", False),
                                              ("empty", False), ("legacy", True)])
def test_tables_disclose_effective_residual_policy(tmp_path, case, disclosed):
    report = final_report()
    report["calibration_gate_audit"] = {"delta_x_forced_zero": True}
    if case != "legacy":
        report["residual_evaluation"] = {
            "enabled_requests": 1 if case == "enabled" else 0,
            "disabled_requests": 0 if case == "empty" else 5,
            "changed_payload_requests": 1 if case == "enabled" else 0,
            "m3_aliases_m1": case != "enabled", "m4_aliases_m2": case != "enabled",
        }
        if case == "disabled":
            report["calibration_gate_audit"] = {"verified": False}
    curves = paper.normalize_report(report)
    paper.write_tables(paper.mean_curves(curves), paper.compute_bd_rates(curves, report), report, tmp_path)
    for filename in ("table_m1_m4_coco.tex", "table_cross_detector.tex"):
        assert (r"\Delta_x=0" in (tmp_path / filename).read_text()) is disclosed


def test_seed_averaging_and_undefined_metrics():
    report = final_report()
    report["results"]["metrics"]["yolo11l"]["M2_seed0_crf20"]["ap_small"] = -1
    report["results"]["metrics"]["yolo11l"]["M2_seed0_crf20"]["bytes"] = 100_000
    means = paper.mean_curves(paper.normalize_report(report))
    assert means["yolo11l"]["M2"][20]["bytes"] == pytest.approx(260_000 / 3)
    assert means["yolo11l"]["M2"][20]["ap_small"] is None


def test_missing_crf_rejected():
    report = final_report()
    del report["results"]["metrics"]["faster_rcnn"]["M4_seed2_crf36"]
    with pytest.raises(ValueError, match="all five CRFs"):
        paper.normalize_report(report)


class KernelWorkerStatus(Enum):
    COMPLETE = 1
    RUNNING = 2
    ERROR = 3


@pytest.mark.parametrize(("value", "expected"), [
    ('user/slug has status "KernelWorkerStatus.COMPLETE"', "COMPLETE"),
    ('user/slug has status "running"', "RUNNING"),
    ("KernelWorkerStatus.ERROR", "ERROR"),
    ("KernelWorkerStatus.QUEUED", "QUEUED"),
    ("complete", "COMPLETE"),
    (KernelWorkerStatus.COMPLETE, "COMPLETE"),
    (SimpleNamespace(status=KernelWorkerStatus.RUNNING), "RUNNING"),
    ({"status": "complete"}, "COMPLETE"),
    ('{"status": "KernelWorkerStatus.COMPLETE"}', "COMPLETE"),
    ('user/slug has status "KernelWorkerStatus.ERROR"\nFailure message: "COMPLETE not reached"', "ERROR"),
    ("HTTP error: could not COMPLETE download", "UNKNOWN"),
    ("INCOMPLETE", "UNKNOWN"),
    ("cancelled", "ERROR"),
    (None, "UNKNOWN"),
])
def test_parse_kernel_status(value, expected):
    assert harvester.parse_kernel_status(value) == expected


def shard_report(index=0):
    ids = list(range(index, 5000, 19))
    observations = []
    for image_id in ids:
        for model in paper.MODELS:
            for seed in ([None] if model == "M1" else [0, 1, 2]):
                for crf in paper.CRFS:
                    observations.append({
                        "image_id": image_id, "model": model, "seed": seed, "crf": crf,
                        "whole_bitstream_bytes": 100, "bits": 800,
                        "yolo11l": {"ap50": None, "rows": []}, "faster_rcnn": {"ap50": 0.2, "rows": []},
                    })
    return {"schema_version": 1, "status": "complete", "shard_idx": index, "num_shards": 19,
            "dataset_image_count": 5000, "image_ids": ids, "observations": observations,
            "observation_count": len(observations), "crfs": list(paper.CRFS), "seeds": [0, 1, 2],
            "annotations_sha256": "annotation-hash"}


def write_report(directory, index, report):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"shard_{index}_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


@pytest.mark.parametrize("damage", [None, "bits", "missing", "duplicate", "wrong_shard", "truncated", "prediction"])
def test_validate_shard_report(tmp_path, damage):
    report = shard_report()
    if damage == "bits":
        report["observations"][0]["bits"] = 799
    elif damage == "missing":
        report["observations"].pop()
        report["observation_count"] -= 1
    elif damage == "duplicate":
        report["image_ids"][-1] = report["image_ids"][0]
    elif damage == "wrong_shard":
        report["shard_idx"] = 1
    elif damage == "prediction":
        report["observations"][0]["yolo11l"]["rows"] = [
            {"image_id": 0, "category_id": 1, "score": 2, "bbox": [0, 0, 1, 1]}]
    path = write_report(tmp_path, 0, report)
    if damage == "truncated":
        path.write_text('{"observations": [')
    if damage:
        with pytest.raises(ValueError):
            harvester.validate_shard_report(path, 0)
    else:
        result = harvester.validate_shard_report(path, 0, expected_ids=range(0, 5000, 19),
                                                annotations_sha256="annotation-hash")
        assert result["observation_count"] == 264 * 50
        assert len(result["sha256"]) == 64
        with pytest.raises(ValueError, match="SHA-256"):
            harvester.validate_shard_report(path, 0, annotations_sha256="wrong")


def test_poll_downloads_only_complete_and_retries_corrupt_reports(tmp_path, monkeypatch, capsys):
    accounts = [{"username": "test-user", "key": "test-key", "dir": tmp_path}]
    downloads = []

    def status(index, account):
        if index == 2:
            raise subprocess.TimeoutExpired("kaggle", 60)
        return "COMPLETE" if index == 0 else "ERROR" if index == 1 else "RUNNING"

    def download(index, account, directory):
        downloads.append(index)
        write_report(directory, index, shard_report(index))

    monkeypatch.setattr(harvester, "check_shard_status", status)
    monkeypatch.setattr(harvester, "download_shard_outputs", download)
    states = harvester.poll_shards(accounts, tmp_path)
    assert states[0]["downloaded"] and not any(s["downloaded"] for s in states[1:])
    assert states[1]["status"] == "ERROR" and "timed out" in states[2]["error"]
    assert not harvester.all_shards_complete(states)
    assert downloads == [0]
    harvester.poll_shards(accounts, tmp_path)
    assert downloads == [0]  # Valid cached outputs are not downloaded twice.
    (tmp_path / "shard_0/shard_0_report.json").write_text("broken")
    harvester.poll_shards(accounts, tmp_path)
    assert downloads == [0, 0]
    assert "1/19 shards complete (5.3%)" in capsys.readouterr().out


def test_subprocess_credentials_and_paginated_download(tmp_path, monkeypatch):
    account = {"username": "account", "key": "account-specific-key", "dir": tmp_path}
    monkeypatch.setenv("KAGGLE_API_TOKEN", "unrelated-token")
    monkeypatch.setenv("KAGGLE_KEY", "unrelated-key")
    calls = []

    def run(command, **kwargs):
        assert kwargs["env"]["KAGGLE_API_TOKEN"] == account["key"]
        assert "KAGGLE_KEY" not in kwargs["env"]
        assert kwargs["env"]["KAGGLE_CONFIG_DIR"] == str(tmp_path)
        assert account["key"] not in command
        calls.append(command)
        return SimpleNamespace(stdout="Next page token: page-2\n" if len(calls) == 1 else "Done\n")

    monkeypatch.setattr(harvester.subprocess, "run", run)
    harvester.download_shard_outputs(0, account, tmp_path / "download")
    assert len(calls) == 2 and calls[1][-2:] == ["--page-token", "page-2"]
    assert not (tmp_path / "download/.download_in_progress").exists()


def complete_states():
    return [{"shard_idx": i, "status": "COMPLETE", "downloaded": True,
             "report": {"shard_idx": i, "sha256": f"hash-{i}", "image_ids": list(range(i, 5000, 19)),
                        "provenance": {"repo_commit": "same-commit"}}} for i in range(19)]


def test_identifies_complete_shards_and_requires_downloads():
    states = complete_states()
    assert harvester.all_shards_complete(states)
    assert not harvester.all_shards_complete(states[:-1])
    states[-1]["downloaded"] = False
    assert not harvester.all_shards_complete(states)
    states[-1]["downloaded"] = True
    states[-1]["status"] = "RUNNING"
    assert not harvester.all_shards_complete(states)


@pytest.mark.parametrize("failure", [None, "merge", "artifacts"])
def test_finalization_receipt_and_restart(tmp_path, monkeypatch, failure):
    annotations = tmp_path / "instances_val2017.json"
    annotations.write_text("{}")
    calls = []

    def run(command, **kwargs):
        script = Path(command[1]).name
        calls.append(script)
        if script == "merge_pool_5k.py":
            if failure == "merge":
                raise subprocess.CalledProcessError(1, command)
            report = final_report()
            report["results"]["bd_rates"] = {"yolo11l": {"M4": {"mean": {}}}}
            (tmp_path / harvester.FINAL_REPORT).write_text(json.dumps(report))
        else:
            if failure == "artifacts":
                raise subprocess.CalledProcessError(1, command)
            directory = tmp_path / "paper_artifacts"
            directory.mkdir(exist_ok=True)
            for name in paper.ARTIFACT_NAMES:
                (directory / name).write_text("test artifact")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(harvester.subprocess, "run", run)
    states = complete_states()
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            harvester.finalize_harvest(states, tmp_path, annotations)
        assert not (tmp_path / "harvest_receipt.json").exists()
    else:
        receipt = harvester.finalize_harvest(states, tmp_path, annotations)
        assert receipt["status"] == "complete" and receipt["total_images"] == 5000
        assert calls == ["merge_pool_5k.py", "generate_paper_artifacts.py"]
        assert harvester.finalize_harvest(states, tmp_path, annotations) == receipt
        assert len(calls) == 2
        (tmp_path / "paper_artifacts/rd_curves.png").write_text("corrupted")
        harvester.finalize_harvest(states, tmp_path, annotations)
        assert len(calls) == 4


def test_finalization_rejects_overlap_and_mixed_provenance(tmp_path):
    states = complete_states()
    states[1]["report"]["image_ids"][0] = 0
    with pytest.raises(ValueError, match="distinct"):
        harvester.finalize_harvest(states, tmp_path, tmp_path / "annotations.json")
    states = complete_states()
    states[1]["report"]["provenance"]["repo_commit"] = "different"
    with pytest.raises(ValueError, match="provenance"):
        harvester.finalize_harvest(states, tmp_path, tmp_path / "annotations.json")


def test_once_checks_without_sleep_or_premature_merge(tmp_path, monkeypatch):
    annotations = tmp_path / "instances_val2017.json"
    annotations.write_text(json.dumps({"images": [{"id": i} for i in range(5000)], "categories": [{"id": 1}]}))
    monkeypatch.setattr(harvester, "discover_pool_accounts", lambda path: [{"username": "test"}])
    monkeypatch.setattr(harvester, "poll_shards", lambda *a, **k: [
        {"shard_idx": i, "status": "RUNNING", "downloaded": False} for i in range(19)])
    monkeypatch.setattr(harvester.time, "sleep", lambda *a: pytest.fail("--once must not sleep"))
    monkeypatch.setattr(harvester, "finalize_harvest", lambda *a: pytest.fail("premature merge"))
    assert harvester.main(["--annotations", str(annotations), "--results-dir", str(tmp_path), "--once"]) == 0
    assert not (tmp_path / ".harvester.lock").exists()


def test_merger_bd_api_uses_rdpoints_and_preserves_numeric_results(monkeypatch):
    observations = []
    for model in paper.MODELS:
        for seed in ([None] if model == "M1" else [0, 1, 2]):
            for i, crf in enumerate(paper.CRFS):
                observations.append({"model": model, "seed": seed, "crf": crf,
                    "whole_bitstream_bytes": (100_000 - i * 15_000) * (1 if model == "M1" else 0.8),
                    **{det: {"rows": [{"quality": 0.6 - i * 0.06}]} for det in paper.DETECTORS}})
    monkeypatch.setattr(merger, "evaluate_pooled_coco_metrics", lambda coco, ids, rows: {
        "ap_50_95": rows[0]["quality"], "ap50": rows[0]["quality"] + 0.2, "ap_small": 0.3})
    results = merger.compute_rd_and_bd_rates(observations, None, [0])
    assert results["metrics"]["yolo11l"]["M1_crf20"]["ap_small"] == 0.3
    for det in paper.DETECTORS:
        for model in paper.MODELS[1:]:
            assert list(results["bd_rates"][det][model]["mean"].values()) == pytest.approx([-20] * 4, abs=1e-7)
    json.dumps(results, allow_nan=False)


def test_merger_rejects_missing_shards_and_conflicting_copies(tmp_path):
    with pytest.raises(ValueError, match="exactly shards"):
        merger.merge_shard_data({}, 19)
    write_report(tmp_path / "shard_0", 0, {"value": 1})
    write_report(tmp_path / "shard_0/nested", 0, {"value": 2})
    with pytest.raises(ValueError, match="Conflicting reports"):
        merger.find_shard_reports(tmp_path)
