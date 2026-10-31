"""Split identities are SHA256s of source files, frozen before training."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class SplitEntry:
    sample_id: str
    sha256: str
    split: str
    source: str = "train2017"
    sequence_id: str | None = None


class SplitLedger:
    def __init__(self, entries: list[SplitEntry], frozen_sha256: str | None = None,
                 protocol_frozen: bool = False, pilot_passed: bool = False):
        self.entries = tuple(entries)
        self.protocol_frozen, self.pilot_passed = protocol_frozen, pilot_passed
        self.by_id = {entry.sample_id: entry for entry in entries}
        if len(self.by_id) != len(entries) or len({e.sha256 for e in entries}) != len(entries):
            raise ValueError("Duplicate sample identity or content across splits")
        sequences: dict[str, str] = {}
        for entry in entries:
            if entry.split not in {"train", "calibration", "dev", "pilot", "test"}:
                raise ValueError("Unknown split")
            if len(entry.sha256) != 64 or any(c not in "0123456789abcdef" for c in entry.sha256):
                raise ValueError("Expected lowercase source SHA256")
            if entry.split != "test" and entry.source not in {"train2017", "K400", "MOT17"}:
                raise ValueError("COCO calibration/holdout must come from train2017")
            if entry.source == "val2017" and entry.split != "test":
                raise ValueError("val2017 is sealed test data")
            if entry.sequence_id:
                key = entry.source + ":" + entry.sequence_id
                if key in sequences and sequences[key] != entry.split:
                    raise ValueError("Video sequence crosses split boundaries")
                sequences[key] = entry.split
        self.digest = hashlib.sha256(self.canonical_bytes()).hexdigest()
        self.frozen = frozen_sha256 is not None
        if self.frozen and self.digest != frozen_sha256:
            raise ValueError("Frozen split ledger digest mismatch")

    def canonical_bytes(self) -> bytes:
        return json.dumps([entry.__dict__ for entry in sorted(self.entries,
                          key=lambda entry: entry.sample_id)], sort_keys=True,
                          separators=(",", ":")).encode()

    def require(self, sample_id: str, split: str) -> SplitEntry:
        if not self.frozen or not self.entries:
            raise RuntimeError("Populate and freeze real source SHA lists before use")
        entry = self.by_id[sample_id]
        if entry.split != split:
            raise ValueError("Sample is outside the requested split")
        if split == "test" and not (self.protocol_frozen and self.pilot_passed):
            raise RuntimeError("Test is sealed until protocol freeze and pilot PASS")
        return entry

    @classmethod
    def read(cls, path: Path) -> SplitLedger:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls([SplitEntry(**entry) for entry in data["entries"]],
                   data.get("frozen_sha256"), data.get("protocol_frozen", False),
                   data.get("pilot_passed", False))

    def freeze(self, path: Path) -> None:
        if path.exists() or not self.entries:
            raise ValueError("Freeze requires a new file and nonempty real identities")
        path.write_text(json.dumps({"entries": [entry.__dict__ for entry in self.entries],
            "frozen_sha256": self.digest, "protocol_frozen": self.protocol_frozen,
            "pilot_passed": self.pilot_passed}, indent=2), encoding="utf-8")
