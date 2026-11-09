from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from prism.data.splits import SplitEntry, SplitLedger
from prism.training.calibration import CalibrationCell, CalibrationGate
from prism.training.gates import require_training_ready
from prism.training.manifest import EncodeManifest
from tier1a.oracle_codec import ClaimConfig
import tier1a.run_tier1a_step3 as step3
from tier1a.oracle_candidates import candidates_hash, generate_candidates
from tier1a.run_tier1a_step3 import (
    build_spatial_oracle_target,
    masked_qg_huber_loss,
    project_zero_mean_qp_map,
    qg_area_weights,
    run_asm_smoke_suite,
    run_calibration_audit,
    select_spatial_winner,
    train_student_seed,
)
from tier1a.step3_splits import build_step3_splits

REPO = Path(__file__).resolve().parents[1]


def test_step3_stratified_split_ledger_and_require_training_ready(tmp_path):
    parent_path = REPO / "prism/data/splits/coco500_ledger.json"
    selection_path = REPO / "prism/data/splits/coco500_selection.json"
    out_ledger = tmp_path / "step3_splits.json"
    out_meta = tmp_path / "step3_meta.json"

    ledger, meta = build_step3_splits(parent_path, selection_path, out_ledger, out_meta)
    assert ledger.frozen and len(ledger.entries) == 500
    assert meta["counts_by_split"]["train"] == {"person": 120, "vehicle": 90, "animal": 90, "total": 300}
    assert meta["counts_by_split"]["calibration"] == {"person": 40, "vehicle": 30, "animal": 30, "total": 100}
    assert meta["counts_by_split"]["dev"] == {"person": 20, "vehicle": 15, "animal": 15, "total": 50}
    assert meta["counts_by_split"]["pilot"] == {"person": 20, "vehicle": 15, "animal": 15, "total": 50}

    # Verify require_training_ready passes with frozen Step 3 ledger
    pin = {"abi": 215, "enable_assembly": True}
    require_training_ready({"status": "PASS", "pin": pin}, ledger, pin)

    # Verify cross-split access is blocked
    pilot_entry = next(e for e in ledger.entries if e.split == "pilot")
    with pytest.raises(ValueError, match="outside the requested split"):
        ledger.require(pilot_entry.sample_id, "train")


def test_spatial_winner_filters_uniform_and_centers_on_known_mask():
    shape = (68, 120)
    known = np.zeros(shape, dtype=bool)
    known[16:52, 30:90] = True
    sam = np.zeros(shape, dtype=bool)
    sam[20:40, 40:70] = True
    saliency = np.where(sam, 1.0, 0.1)

    maps = generate_candidates(1920, 1080, sam, saliency, sample_id="2782", observed=known)
    c_hash = candidates_hash(maps)
    # Candidate 1 (+3_unif) has highest J=0.90, but Candidate 3 (SAM_-3/+2) is best spatial J=0.88
    candidates = [
        {"index": i, "status": "PASS", "decode_ok": True, "bits": 10000 - i * 10, "J": 0.85}
        for i in range(8)
    ]
    candidates[1]["J"] = 0.90  # uniform +3
    candidates[3]["J"] = 0.88  # SAM_-3/+2
    candidates[4]["J"] = 0.87  # DINO_top25_-3/+2

    winner_idx = select_spatial_winner(candidates)
    assert winner_idx == 3

    label_row = {
        "sample_id": "2782",
        "candidates_sha256": c_hash,
        "winner_map_index": 1,
        "candidates": candidates,
    }
    target = build_spatial_oracle_target(maps, known, label_row)
    assert target["winner_index"] == 3
    assert target["raw_winner_index"] == 1
    t_qg = target["target_qg"]
    # Outside known mask must be strictly 0
    assert np.all(t_qg[~known] == 0.0)
    # Inside known mask must have weighted mean ~ 0 and negative values on SAM foreground
    w = qg_area_weights(1920, 1080)
    weighted_mean = float((t_qg[known] * w[known]).sum() / w[known].sum())
    assert abs(weighted_mean) < 1e-4
    assert float(t_qg[sam & known].mean()) < 0.0
    assert float(t_qg[(~sam) & known].mean()) > 0.0


def test_masked_qg_huber_loss_ignores_unseen_outside_crop():
    pred_ctu = torch.zeros((2, 1, 17, 30), requires_grad=True)
    target_qg = torch.zeros((2, 68, 120))
    known = torch.zeros((2, 68, 120), dtype=torch.bool)
    known[:, 20:40, 40:80] = True
    # Put huge corrupt target outside known mask; loss inside known mask is 0
    target_qg[~known] = 100.0
    area_w = torch.from_numpy(qg_area_weights(1920, 1080))
    loss = masked_qg_huber_loss(pred_ctu, target_qg, known, area_w)
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)


def test_fusion_label_can_win_spatial_supervision():
    shape = (4, 4)
    sam = np.eye(4, dtype=bool)
    maps = generate_candidates(64, 64, sam, sam.astype(float),
                               sample_id="joint", include_fusion=True)
    row = {"sample_id": "joint", "candidate_count": 9,
           "candidates_sha256": candidates_hash(maps), "winner_map_index": 8,
           "candidates": [{"index": i, "status": "PASS", "decode_ok": True,
                           "bits": 1000, "J": .9 if i == 8 else .5} for i in range(9)]}
    target = build_spatial_oracle_target(maps, np.ones(shape, bool), row, width=64, height=64)
    assert target["winner_index"] == 8
    assert target["winner_name"] == "sam_dino_fusion_minus4_plus6"
    assert np.all(target["target_qg"][sam] < 0)
    assert abs(float(target["target_qg"].mean())) < 1e-5


def test_student_training_and_calibration_fail_closed(tmp_path):
    # 1. Verify Student seed training decreases masked QG Huber loss
    known = np.zeros((68, 120), dtype=bool)
    known[16:52, 30:90] = True
    target = np.zeros((68, 120), dtype=np.float32)
    target[20:36, 45:75] = -3.0
    target[36:52, 45:75] = 2.0
    samples = [
        {
            "sample_id": str(i),
            "image_id": i,
            "x_low": torch.full((3, 256, 256), 0.5, dtype=torch.float32),
            "target_qg": target,
            "known_mask": known,
        }
        for i in range(6)
    ]
    _, rep = train_student_seed(
        0,
        samples[:4],
        samples[4:],
        tmp_path,
        device="cpu",
        backbone="small",
        allow_fallback=True,
        epochs=6,
        batch_size=2,
        lr=5e-3,
    )
    assert rep["final_train_huber"] < rep["initial_train_huber"]
    assert Path(rep["checkpoint_path"]).is_file()

    # Calibration is covered below with source-bound fake evidence, without training
    # or writing the tracked split ledger.


def calibration_fixture():
    entries, samples, labels = [], {}, {}
    config = ClaimConfig()
    pin = {"library_sha256": "a" * 64, "shim_sha256": "b" * 64,
           "version": "fake", "abi": 1, "picture_size": 1, "enable_assembly": True}
    for i in range(100):
        sid = str(i)
        source_sha = hashlib.sha256(f"source-{sid}".encode()).hexdigest()
        entries.append(SplitEntry(sid, source_sha, "calibration"))
        payload = bytes([i + 16] * 16 + [128] * 8)
        maps = np.stack([np.full((4, 4), v, np.int8) for v in (0, 3, -3, 1, 2, -2)])
        frame = {"sample_id": sid, "image_id": i, "source_sha256": source_sha,
                 "geometry": {"width": 4, "height": 4, "matrix": "bt709", "range": "limited",
                              "raw_sha256": hashlib.sha256(payload).hexdigest()}}
        samples[sid] = (frame, payload, maps)
        labels[sid] = {"sample_id": sid, "image_id": i, "status": "PASS",
                       "source_sha256": source_sha, "input_yuv_sha256": hashlib.sha256(payload).hexdigest(),
                       "candidate_count": 6, "candidates_sha256": candidates_hash(maps),
                       "candidates": [{"index": k, "status": "PASS", "decode_ok": True,
                           "bits": (i + 16) * 16, "whole_bitstream_bytes": (i + 16) * 2,
                           "J": .5, "map_sha256": hashlib.sha256(m.tobytes()).hexdigest(),
                           "encoder_parameters": config.options(Path(f"{sid}-{k}.csv")), "build_pin": pin}
                           for k, m in enumerate(maps)]}
    draft = SplitLedger(entries)
    ledger = SplitLedger(entries, draft.digest)
    contract = {"encoder": asdict(config), "pin": pin}
    contract_sha = step3.digest_json(dict(contract, lambda_rate=.05))
    for sid, row in labels.items():
        row["binding_sha256"] = step3.digest_json({"candidates": row["candidates_sha256"],
            "input": row["input_yuv_sha256"], "geometry": samples[sid][0]["geometry"], "contract": contract_sha})
    return list(labels), ledger, {"contract": contract, "contract_sha256": contract_sha,
                                 "lambda_R": .05, "labels": labels}, samples


class PixelRateProxy:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, image, offsets, frame_type, qp):
        return {"rate": image.flatten(1).sum(1), "reconstruction": image}


def test_calibration_uses_corresponding_full_resolution_luma(monkeypatch):
    ids, ledger, labels, samples = calibration_fixture()
    monkeypatch.setattr(step3, "ProxyCodec", PixelRateProxy)
    report = run_calibration_audit(ids[::-1], ledger, labels, device="cpu", sample_loader=samples.__getitem__)
    assert report["rate_metrics"]["mape"] == pytest.approx(0.0)
    assert [m["predicted_bits"] for m in report["measurements"]] == [float((i + 16) * 16) for i in range(100)]
    assert [m["sample_id"] for m in report["measurements"]] == ids
    assert report["verified"] is False
    assert "direction_delta_j" in report["unavailable_evidence"]
    assert report["cell_metrics"]["passed"] is False
    assert "assess_step1000" not in report
    assert "m3_aliases_m1" not in report
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("bad", ["missing", "swapped", "source", "map", "geometry", "geometry_binding", "codec", "nonfinite"])
def test_calibration_rejects_missing_or_misaligned_samples(monkeypatch, bad):
    ids, ledger, labels, samples = calibration_fixture()
    monkeypatch.setattr(step3, "ProxyCodec", PixelRateProxy)
    if bad == "missing":
        samples.pop("0")
    elif bad == "swapped":
        samples["0"] = samples["1"]
    elif bad == "source":
        samples["0"][0]["source_sha256"] = "c" * 64
    elif bad == "map":
        samples["0"][2][0, 0, 0] = 1
    elif bad == "geometry":
        samples["0"][0]["geometry"]["width"] = 8
    elif bad == "geometry_binding":
        samples["0"][0]["geometry"].update(width=2, height=8)
    elif bad == "codec":
        labels["labels"]["0"]["candidates"][0]["encoder_parameters"] = "crf=36"
    else:
        labels["labels"]["0"]["candidates"][0]["bits"] = float("nan")
    report = run_calibration_audit(ids, ledger, labels, device="cpu", sample_loader=samples.__getitem__)
    assert report["verified"] is False
    assert report["rate_metrics"]["passed"] is False
    assert report["sample_failures"]
    json.dumps(report, allow_nan=False)


def test_calibration_without_images_is_explicitly_unavailable():
    ids, ledger, labels, _ = calibration_fixture()
    report = run_calibration_audit(ids, ledger, labels, device="cpu")
    assert report["verified"] is False
    assert report["measurements"] == []
    assert "sample_inputs" in report["unavailable_evidence"]
    assert report["consecutive_failures"] == 1


def verified_gate():
    gate = CalibrationGate({(20, "I")})
    values = np.arange(1., 101.)
    cell = CalibrationCell(20, "I", values, values, values, values, np.full(100, -.2), True, True, True, True)
    gate.assess(0, [cell], lambda: None)
    return gate


@pytest.mark.parametrize("policy", [None, {}, "PASS", "failed", "expired", "wrong_cell", "valid"])
def test_encoder_payload_boundary_fails_closed_or_preserves_verified_residual(policy):
    gate = verified_gate() if isinstance(policy, str) and policy in {"failed", "expired", "wrong_cell", "valid"} else policy
    step, crf = 0, 20
    if policy == "failed":
        gate.verified = False
    elif policy == "expired":
        step = 1000
    elif policy == "wrong_cell":
        crf = 24
    payload = bytes([80] * 16 + [128] * 8)
    result, report = step3.prepare_residual_payload(payload, np.full((4, 4), 2.), 4, 4,
                                                  calibration_gate=gate, step=step, crf=crf)
    assert result == (bytes([82] * 16 + [128] * 8) if policy == "valid" else payload)
    assert report["residual_enabled"] is (policy == "valid")
    assert report["payload_changed"] is (policy == "valid")
    assert report["reason"]


@pytest.mark.parametrize("bad", ["tampered", None, [], "invalid"])
def test_calibration_rejects_unbound_or_malformed_contract(monkeypatch, bad):
    ids, ledger, labels, samples = calibration_fixture()
    monkeypatch.setattr(step3, "ProxyCodec", PixelRateProxy)
    if bad == "tampered":
        labels["contract"]["encoder"]["crf"] = 36.0
        config = ClaimConfig(**labels["contract"]["encoder"])
        for row in labels["labels"].values():
            row["candidates"][0]["encoder_parameters"] = config.options(Path("changed.csv"))
    else:
        labels["contract"] = bad
    report = run_calibration_audit(ids, ledger, labels, device="cpu", sample_loader=samples.__getitem__)
    assert report["verified"] is False
    assert report["rate_metrics"]["passed"] is False
    assert report["measurements"] == []
    assert "codec_configuration" in report["unavailable_evidence"]


@pytest.mark.parametrize("field", ["semantic_passed", "same_codec_state", "feasible_plus_minus_one", "dependent_frames_measured"])
@pytest.mark.parametrize("value", [float("nan"), "true", 1, None])
def test_invalid_gate_flag_cannot_authorize_encoder_payload(field, value):
    from dataclasses import replace

    values = np.arange(1., 101.)
    cell = CalibrationCell(20, "I", values, values, values, values, np.full(100, -.2), True, True, True, True)
    gate = CalibrationGate({(20, "I")})
    gate.assess(0, [replace(cell, **{field: value})], lambda: None)
    payload = bytes([80] * 16 + [128] * 8)
    result, report = step3.prepare_residual_payload(payload, np.full((4, 4), 2.), 4, 4, calibration_gate=gate)
    assert result == payload
    assert report["residual_enabled"] is False


@pytest.mark.parametrize("raw", [np.full((4, 4), np.nan), np.ones((2, 2)), np.full((4, 4), "bad")])
def test_invalid_residual_is_zero_at_encoder_payload_boundary(raw):
    payload = bytes([80] * 16 + [128] * 8)
    result, report = step3.prepare_residual_payload(payload, raw, 4, 4, calibration_gate=verified_gate(), crf=20)
    assert result == payload
    assert report["residual_enabled"] is False


@pytest.mark.parametrize("gate_passes", [False, True])
def test_main_enforces_calibration_at_actual_encoder_boundary(tmp_path, monkeypatch, gate_passes):
    ids, ledger, labels, samples = calibration_fixture()
    extra = [SplitEntry(sid, hashlib.sha256(sid.encode()).hexdigest(), split)
             for sid, split in (("100", "train"), ("101", "dev"), ("102", "pilot"))]
    draft = SplitLedger(list(ledger.entries) + extra)
    ledger = SplitLedger(list(draft.entries), draft.digest)
    frame, payload, maps = samples["0"]
    payload = bytes([80] * 16 + [128] * 8)
    for entry in extra:
        samples[entry.sample_id] = (dict(frame, sample_id=entry.sample_id, image_id=int(entry.sample_id)), payload, maps)
        labels["labels"][entry.sample_id] = labels["labels"]["0"]
    rows = [{"sample_id": e.sample_id, "image_id": int(e.sample_id)} for e in ledger.entries]
    captured, calls, h1_evidence = {}, [], []
    original_h1_pass = step3.h1_pass

    def record_h1(comparisons, **kwargs):
        h1_evidence.append(comparisons)
        return original_h1_pass(comparisons, **kwargs)

    monkeypatch.setattr(step3, "h1_pass", record_h1)

    class Encoder:
        pin = stock_pin = labels["contract"]["pin"]
        stack = {"enable_assembly": True}

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, data, offsets, frame, directory, *, stock=False, crf=20.):
            calls.append((data, None if offsets is None else offsets.copy(), stock, crf))
            size = int(1000 - crf * 10 - (20 if offsets is not None else 0))
            return {"bits": size * 8, "whole_bitstream_bytes": size, "encode_seconds": .01,
                    "bitstream_sha256": hashlib.sha256(data + str(stock).encode()).hexdigest()}

        def decode(self, result, frame):
            score_level = 20 + (result["whole_bitstream_bytes"] - 620) // 40
            return np.full((4, 4, 3), score_level, np.uint8)

    class Scorer:
        coco = None
        provenance = {}

        def __init__(self, *args, **kwargs):
            pass

        def score_both(self, rgb, frame):
            return {k: {"ap50": float(rgb[0, 0, 0]) / 100, "rows": [{"score": float(rgb[0, 0, 0]) / 100}]}
                    for k in ("yolo11l", "faster_rcnn")}

    import tier1a.step3_detectors as detectors
    monkeypatch.setattr(detectors, "DualDetectorScorer", Scorer)
    monkeypatch.setattr(detectors, "evaluate_pooled_coco_metrics", lambda coco, ids, rows: {
        "ap50": float(np.mean([r["score"] for r in rows])),
        "ap_50_95": float(np.mean([r["score"] for r in rows])) - .01, "ap_small": .3})
    monkeypatch.setattr(step3, "ensure_stack", lambda *args, **kwargs: {})
    monkeypatch.setattr(step3, "load_inputs", lambda *args: (ledger, rows, "selection"))
    monkeypatch.setattr(step3, "build_step3_splits", lambda *args: (ledger, {"counts_by_split": {}}))
    monkeypatch.setattr(step3, "load_frame", lambda root, row, *args, **kwargs: samples[row["sample_id"]])
    monkeypatch.setattr(step3, "extract_known_mask", lambda *args: np.ones((4, 4), bool))
    monkeypatch.setattr(step3, "build_spatial_oracle_target", lambda *args: {"winner_index": 0, "target_qg": maps[0].astype(np.float32), "known_mask": np.ones((4, 4), bool)})
    monkeypatch.setattr(step3, "train_student_seed", lambda *args, **kwargs: (object(), {}))
    monkeypatch.setattr(step3, "Gate0Encoder", Encoder)
    monkeypatch.setattr(step3, "run_asm_smoke_suite", lambda *args: {"status": "PASS", "physical_encodes": 0})
    monkeypatch.setattr(step3, "read_json", lambda path: labels)
    monkeypatch.setattr(step3, "write_json", lambda path, value: captured.update({path.name: value}))
    monkeypatch.setattr(step3, "SEEDS", (0,))
    monkeypatch.setattr(step3, "infer_student_qp_map", lambda *args, **kwargs: np.ones((4, 4), np.int8))
    monkeypatch.setattr(step3, "infer_student_residual", lambda *args, **kwargs: np.full((4, 4), 2.))
    monkeypatch.setattr(step3, "paired_bd_bootstrap", lambda *args, **kwargs: {"valid": False, "ci95": None})

    def audit(*args, **kwargs):
        gate = kwargs.get("calibration_gate")
        if gate_passes and gate is not None:
            values = np.arange(1., 101.)
            cells = [CalibrationCell(q, "I", values, values, values, values, np.full(100, -.2), True, True, True, True)
                     for q in step3.CRF_GRID]
            gate.assess(0, cells, lambda: None)
        return {"verified": gate_passes, "status": "PASS" if gate_passes else "UNAVAILABLE"}

    monkeypatch.setattr(step3, "run_calibration_audit", audit)
    monkeypatch.setattr(sys, "argv", ["step3", "--annotations", str(tmp_path / "annotations"), "--weights", str(tmp_path / "weights"),
                        "--output", str(tmp_path / "out"), "--scratch-dir", str(tmp_path / "scratch"),
                        "--working", str(tmp_path), "--eval-split", "pilot", "--no-build"])
    assert step3.main() == 0
    assert len(calls) == (20 if gate_passes else 10)
    for crf in step3.CRF_GRID:
        at_qp = [c for c in calls if c[3] == crf]
        assert at_qp[0][0] == at_qp[1][0] == payload
        assert at_qp[0][1] is None and at_qp[0][2] is True
        np.testing.assert_array_equal(at_qp[1][1], np.ones((4, 4), np.int8))
        assert at_qp[1][2] is False
        if gate_passes:
            assert at_qp[2][0] == at_qp[3][0] == bytes([82] * 16 + [128] * 8)
            assert at_qp[2][1] is None and at_qp[2][2] is True
            np.testing.assert_array_equal(at_qp[3][1], at_qp[1][1])
            assert at_qp[3][2] is False
    effective = captured["step3_report.json"]["residual_evaluation"]
    assert effective["enabled_requests"] == (5 if gate_passes else 0)
    assert effective["disabled_requests"] == (0 if gate_passes else 5)
    assert effective["m3_aliases_m1"] is (not gate_passes)
    assert effective["m4_aliases_m2"] is (not gate_passes)
    if gate_passes:
        assert all(e["M4-vs-M2"]["ci95"] is None for e in h1_evidence)


def test_asm_smoke_suite_and_manifest_accounting(tmp_path):
    class StubEncoder:
        stack = {"enable_assembly": True, "nasm_version": "NASM version 2.15.05"}

        def __call__(self, payload, offsets, frame, directory, *, stock=False, crf=20.0):
            is_nonzero = offsets is not None and bool(np.any(offsets))
            tag = f"crf_{int(crf)}_{'plus3' if is_nonzero else 'stock_or_zero'}"
            b_bytes = 50000 - int(crf) * 500 - (5000 if is_nonzero else 0)
            return {
                "whole_bitstream_bytes": b_bytes,
                "bits": b_bytes * 8,
                "bitstream_sha256": f"sha_{tag}",
                "encode_seconds": 0.65,
            }

    manifest = EncodeManifest(
        tmp_path / "manifest.jsonl",
        repo_sha="abc1234",
        config_sha="cfg1234",
        build={"patched": {}, "stock": {}},
        seed=0,
    )
    frame = {"sample_id": "2782", "image_id": 2782, "geometry": {"width": 1920, "height": 1080}}
    smoke = run_asm_smoke_suite(StubEncoder(), frame, b"yuv", tmp_path, manifest)
    assert smoke["status"] == "PASS"
    assert smoke["enable_assembly"] is True
    assert smoke["physical_encodes"] == 11
    assert manifest.counts()["gate0"] == 11

