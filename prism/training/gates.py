"""Prerequisites for any real training or primary benchmark run."""
from __future__ import annotations

from prism.data.splits import SplitLedger


def require_training_ready(gate0: dict, ledger: SplitLedger, build_pin: dict) -> None:
    if gate0.get("status") != "PASS" or gate0.get("pin") != build_pin:
        raise RuntimeError("Training locked: GATE 0 must PASS on this exact pinned build")
    if not ledger.frozen or not ledger.entries:
        raise RuntimeError("Training locked: real split SHA lists are not frozen")
    splits = {entry.split for entry in ledger.entries}
    if not {"train", "calibration", "dev", "pilot"}.issubset(splits):
        raise RuntimeError("Train, calibration, dev and pilot must be separate frozen splits")
