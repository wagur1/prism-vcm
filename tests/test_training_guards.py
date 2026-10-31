import hashlib
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from prism.data.splits import SplitEntry, SplitLedger
from prism.training.calibration import CalibrationCell, CalibrationGate, check_cell
from prism.training.manifest import EncodeManifest, validate_budget
from prism.training.oracle_sweep import OracleConfig, OracleSweep


def ledger():
    entries = [SplitEntry("train-image", "a" * 64, "train"),
               SplitEntry("cal-image", "b" * 64, "calibration")]
    draft = SplitLedger(entries)
    return SplitLedger(entries, draft.digest)


def cell(passed=True):
    values = np.arange(1, 101, dtype=float)
    return CalibrationCell(32, "P", values, values if passed else values * 1.1,
        values, values, np.full(100, -.2), True, True, True, True)


def test_calibration_two_failures_restore_and_disable():
    gate = CalibrationGate({(32, "P")})
    restores = []
    assert gate.apply_delta_x(np.ones(3), 0).sum() == 0
    assert gate.assess(0, [cell()], lambda: restores.append(1))["passed"]
    assert gate.allow_delta_x(999) and not gate.allow_delta_x(1000)
    gate.assess(1000, [cell(False)], lambda: restores.append(1))
    assert not gate.allow_delta_x(1000)
    gate.assess(2000, [cell(False)], lambda: restores.append(1))
    assert restores == [1] and gate.fine_tune_disabled
    gate.assess(3000, [cell()], lambda: restores.append(1))
    assert not gate.allow_delta_x(3000)


def test_calibration_insufficient_samples_and_missing_cells_fail():
    original = cell()
    data = dict(original.__dict__)
    for key in ("actual_bits", "predicted_bits", "actual_delta_j", "predicted_delta_j", "direction_delta_j"):
        data[key] = data[key][:99]
    assert not check_cell(CalibrationCell(**data))["passed"]
    gate = CalibrationGate({(32, "P"), (37, "P")})
    assert not gate.assess(0, [original], lambda: None)["passed"]


def test_oracle_train_only_refresh_and_encode_manifest(tmp_path):
    manifest = EncodeManifest(tmp_path / "encodes.jsonl", repo_sha="test", config_sha="test", build={}, seed=1)
    sweep = OracleSweep(OracleConfig("locked-D", .01, 1000), ledger(), manifest)
    maps = [np.full((2, 2), float(value)) for value in range(-4, 4)]
    score = lambda candidate: (float((candidate**2).sum()), 100)
    feasible = lambda candidate: True
    first = sweep.label("train-image", "c" * 64, maps, score, feasible)
    assert not np.any(first.delta_qp)
    sweep.label("train-image", "c" * 64, maps, score, feasible)
    sweep.label("train-image", "d" * 64, maps, score, feasible)
    assert manifest.counts()["oracle_initial"] == manifest.counts()["oracle_refresh"] == 8
    with pytest.raises(ValueError):
        sweep.label("cal-image", "d" * 64, maps, score, feasible)


def test_frozen_splits_and_sealed_validation(tmp_path):
    with pytest.raises(ValueError):
        SplitLedger([SplitEntry("a", "a" * 64, "train"), SplitEntry("b", "a" * 64, "dev")])
    with pytest.raises(RuntimeError):
        SplitLedger([SplitEntry("a", "a" * 64, "train")]).require("a", "train")
    entries = [SplitEntry("test", "e" * 64, "test", "val2017")]
    draft = SplitLedger(entries)
    frozen = SplitLedger(entries, draft.digest)
    with pytest.raises(RuntimeError):
        frozen.require("test", "test")
    ledger().freeze(tmp_path / "ledger.json")
    assert SplitLedger.read(tmp_path / "ledger.json").frozen


def test_working_budget_strict_disk_limit():
    validate_budget(gpu_hours=60, cpu_hours=100, job_hours=6, working_gb=15.9, measured_intra_images=200)
    with pytest.raises(RuntimeError):
        validate_budget(gpu_hours=60, cpu_hours=100, job_hours=6, working_gb=16, measured_intra_images=200)
