"""Read offline DINOv2-Base/SAM 2.1-Base targets from NPZ shards or LMDB."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .splits import SplitLedger


class CachedDataset:
    def __init__(self, index_path: Path, ledger: SplitLedger, split: str = "train",
                 lmdb_path: Path | None = None):
        self.root = index_path.parent.resolve()
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        self.ledger, self.split = ledger, split
        if self.index["teachers"] != ["DINOv2-Base", "SAM 2.1-Base"]:
            raise ValueError("Unexpected teacher identities")
        if self.index.get("split_ledger_sha256") != ledger.digest:
            raise ValueError("Cache index must bind to the frozen split ledger digest")
        self.records = [row for row in self.index["samples"]
                        if ledger.by_id[row["sample_id"]].split == split]
        for row in self.records:
            entry = ledger.require(row["sample_id"], split)
            if row["source_sha256"] != entry.sha256:
                raise ValueError("Cached sample does not match frozen source hash")
        self.environment = None
        if lmdb_path is not None:
            import lmdb
            self.environment = lmdb.open(str(lmdb_path), readonly=True, lock=False,
                                         readahead=False, max_readers=128)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        self.ledger.require(row["sample_id"], self.split)
        if self.environment is not None:
            with self.environment.begin() as txn:
                payload = txn.get(row["key"].encode())
            if payload is None:
                raise KeyError(row["key"])
        else:
            path = (self.root / row["shard"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Cache shard escapes cache root")
            payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["cache_sha256"]:
            raise ValueError("Teacher cache checksum mismatch")
        with np.load(io.BytesIO(payload), allow_pickle=False) as shard:
            prefix = row.get("prefix", "")
            result = {name: shard[prefix + name].copy()
                      for name in ("x_low", "dino_features", "sam_masks", "rois")}
        if result["x_low"].shape != (3, 256, 256) or result["dino_features"].shape[-1] != 768:
            raise ValueError("Invalid student input or DINOv2-Base target shape")
        return dict(result, sample_id=row["sample_id"])

    def close(self) -> None:
        if self.environment is not None:
            self.environment.close()
            self.environment = None
