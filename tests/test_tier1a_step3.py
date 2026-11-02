from pathlib import Path
import numpy as np
import pytest
import torch

from prism.training.gates import require_training_ready
from tier1a.common import read_json
from tier1a.oracle_candidates import candidates_hash, generate_candidates
from tier1a.run_tier1a_step3 import (
    build_spatial_oracle_target,
    masked_qg_huber_loss,
    project_zero_mean_qp_map,
    qg_area_weights,
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

    # 2. Verify CalibrationGate audit on the real 100 calibration images in oracle_labels.json
    ledger, _ = build_step3_splits(
        REPO / "prism/data/splits/coco500_ledger.json",
        REPO / "prism/data/splits/coco500_selection.json",
        REPO / "prism/data/splits/coco500_tier1a_splits.json",
    )
    cal_ids = [e.sample_id for e in ledger.entries if e.split == "calibration"]
    oracle_labels = read_json(REPO / "tier1a/oracle/oracle_labels.json")
    cal_rep = run_calibration_audit(cal_ids, ledger, oracle_labels, device="cpu")
    assert cal_rep["status"] == "FAIL_CLOSED_DELTA_X_ZERO"
    assert cal_rep["verified"] is False
    assert cal_rep["fine_tune_disabled"] is True
    assert cal_rep["delta_x_forced_zero"] is True
    assert cal_rep["m3_aliases_m1"] is True
    assert cal_rep["m4_aliases_m2"] is True
