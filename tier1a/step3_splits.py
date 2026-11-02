"""Deterministic stratified split partition of COCO500 for Tier 1a Step 3.

Per Council Escalation #12 (6/6 unanimous Option 2A):
Partitions the 500 train2017 images from coco500_ledger.json into:
  - train:       300 images (120 person / 90 vehicle / 90 animal)
  - calibration: 100 images ( 40 person / 30 vehicle / 30 animal)
  - dev:          50 images ( 20 person / 15 vehicle / 15 animal)
  - pilot:        50 images ( 20 person / 15 vehicle / 15 animal)

Sorting within each stratum uses only SHA256("tier1a_step3_split_v1:" + sample_id),
never consulting oracle winners, AP scores, or bitstreams.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from prism.data.splits import SplitEntry, SplitLedger
from tier1a.common import read_json, sha256, write_json

SPLIT_RECIPE_TAG = "tier1a_step3_split_v1"
STRATUM_QUOTAS = {
    "person": {"train": 120, "calibration": 40, "dev": 20, "pilot": 20},
    "vehicle": {"train": 90, "calibration": 30, "dev": 15, "pilot": 15},
    "animal": {"train": 90, "calibration": 30, "dev": 15, "pilot": 15},
}


def deterministic_key(sample_id: str) -> tuple[str, int]:
    digest = hashlib.sha256(f"{SPLIT_RECIPE_TAG}:{sample_id}".encode("utf-8")).hexdigest()
    return digest, int(sample_id)


def build_step3_splits(
    parent_ledger_path: Path,
    selection_path: Path,
    output_ledger_path: Path,
    output_meta_path: Path | None = None,
) -> tuple[SplitLedger, dict]:
    parent_ledger = SplitLedger.read(parent_ledger_path)
    if not parent_ledger.frozen or len(parent_ledger.entries) != 500:
        raise ValueError("Require frozen 500-entry parent COCO500 ledger")
    selection = read_json(selection_path)
    images = selection.get("images", [])
    if len(images) != 500:
        raise ValueError("Require 500-image selection metadata")

    by_stratum: dict[str, list[str]] = {"person": [], "vehicle": [], "animal": []}
    stratum_of: dict[str, str] = {}
    for row in images:
        sid = str(row["sample_id"])
        stratum = row["stratum"]
        if stratum not in by_stratum:
            raise ValueError(f"Unknown stratum: {stratum}")
        if sid not in parent_ledger.by_id:
            raise ValueError(f"Selection sample_id {sid} missing from parent ledger")
        by_stratum[stratum].append(sid)
        stratum_of[sid] = stratum

    for stratum, quotas in STRATUM_QUOTAS.items():
        if len(by_stratum[stratum]) != sum(quotas.values()):
            raise ValueError(f"Stratum {stratum} count mismatch: {len(by_stratum[stratum])}")

    new_entries: list[SplitEntry] = []
    counts_by_split: dict[str, dict[str, int]] = {
        s: {"person": 0, "vehicle": 0, "animal": 0, "total": 0}
        for s in ("train", "calibration", "dev", "pilot")
    }

    for stratum in ("person", "vehicle", "animal"):
        ordered = sorted(by_stratum[stratum], key=deterministic_key)
        quotas = STRATUM_QUOTAS[stratum]
        cursor = 0
        for split_name in ("train", "calibration", "dev", "pilot"):
            count = quotas[split_name]
            for sid in ordered[cursor : cursor + count]:
                old = parent_ledger.by_id[sid]
                new_entries.append(
                    SplitEntry(
                        sample_id=old.sample_id,
                        sha256=old.sha256,
                        split=split_name,
                        source=old.source,
                        sequence_id=old.sequence_id,
                    )
                )
                counts_by_split[split_name][stratum] += 1
                counts_by_split[split_name]["total"] += 1
            cursor += count

    new_entries.sort(key=lambda e: int(e.sample_id))
    if not output_ledger_path.exists():
        unfrozen = SplitLedger(new_entries, protocol_frozen=True, pilot_passed=False)
        output_ledger_path.parent.mkdir(parents=True, exist_ok=True)
        unfrozen.freeze(output_ledger_path)

    frozen_ledger = SplitLedger.read(output_ledger_path)
    if not frozen_ledger.frozen or len(frozen_ledger.entries) != 500:
        raise ValueError("Failed to freeze 500-entry Step 3 split ledger")

    meta = {
        "schema_version": 1,
        "recipe_tag": SPLIT_RECIPE_TAG,
        "council_authority": "Escalation #12 (2026-11-02) 6/6 unanimous Option 2A",
        "exploratory_note": (
            "Derived from COCO500 after Step 2 aggregate oracle audit; prevents direct "
            "train-on-eval while keeping val2017 100% sealed."
        ),
        "parent_ledger_path": str(parent_ledger_path.name),
        "parent_ledger_sha256": parent_ledger.digest,
        "selection_sha256": sha256(selection_path),
        "step3_ledger_sha256": frozen_ledger.digest,
        "counts_by_split": counts_by_split,
        "stratum_by_sample_id": stratum_of,
    }
    if output_meta_path is not None:
        write_json(output_meta_path, meta)
    return frozen_ledger, meta


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    splits_dir = repo / "prism/data/splits"
    ledger, meta = build_step3_splits(
        splits_dir / "coco500_ledger.json",
        splits_dir / "coco500_selection.json",
        splits_dir / "coco500_tier1a_splits.json",
        splits_dir / "coco500_tier1a_splits_meta.json",
    )
    print(json.dumps({
        "step3_ledger_sha256": ledger.digest,
        "parent_ledger_sha256": meta["parent_ledger_sha256"],
        "counts_by_split": meta["counts_by_split"],
    }, indent=2))
