"""Offline extraction from locally supplied teacher adapters; never downloads weights."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from prism.data.splits import SplitLedger


def extract(ledger: SplitLedger, sources: dict[str, str], output: Path,
            teacher_adapter: Any, *, split: str = "train", teacher_size: int = 518) -> None:
    """Adapter owns locally loaded DINO/SAM weights and model-specific normalization.

    Adapter returns dense dino_features [B,768,h,w], sam_masks [B,K,h,w], and
    original-image-coordinate rois [N,5]. Both teachers receive multiples of 14.
    """
    import torch
    from PIL import Image
    from prism.data.transforms import resize_inputs, sample_roi_targets

    output.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for sample_id, filename in sources.items():
            entry = ledger.require(sample_id, split)
            source = Path(filename)
            if hashlib.sha256(source.read_bytes()).hexdigest() != entry.sha256:
                raise ValueError("Source no longer matches frozen split")
            with Image.open(source) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.0
            x_low, teacher_input = resize_inputs(tensor, teacher_size)
            targets = teacher_adapter(teacher_input, source_size=rgb.shape[:2])
            dino = targets["dino_features"]
            if dino.shape[1] != 768:
                raise ValueError("Use DINOv2-Base, not a different teacher size")
            sampled = sample_roi_targets(dino, targets["rois"], image_height=rgb.shape[0],
                                         image_width=rgb.shape[1])
            # Content hashes are safe filenames independent of arbitrary sample IDs.
            shard = output / f"{entry.sha256}.npz"
            np.savez_compressed(shard, x_low=x_low[0].cpu().numpy(),
                dino_features=sampled[..., 0, 0].cpu().numpy(),
                sam_masks=targets["sam_masks"][0].cpu().numpy(), rois=targets["rois"].cpu().numpy())
            records.append({"sample_id": sample_id, "source_sha256": entry.sha256,
                            "shard": shard.name, "cache_sha256": hashlib.sha256(shard.read_bytes()).hexdigest()})
    (output / "index.json").write_text(json.dumps({
        "teachers": ["DINOv2-Base", "SAM 2.1-Base"], "teacher_size": teacher_size,
        "split_ledger_sha256": ledger.digest, "samples": records}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True, help="JSON sample_id -> local image path")
    parser.add_argument("--adapter", required=True, help="Local module:factory returning teacher callable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "calibration", "dev", "pilot"])
    args = parser.parse_args()
    module, factory = args.adapter.split(":")
    adapter = getattr(importlib.import_module(module), factory)()
    extract(SplitLedger.read(args.ledger), json.loads(args.sources.read_text()),
            args.output, adapter, split=args.split)


if __name__ == "__main__":
    main()
