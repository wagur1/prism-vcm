"""Extraction through teacher adapters, which own model provisioning and normalization."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any
from collections.abc import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from prism.data.splits import SplitLedger


def extract(ledger: SplitLedger, sources: dict[str, str], output: Path,
            teacher_adapter: Any, *, split: str = "train", teacher_size: int = 518,
            center_crop: bool = False, fp16: bool = False, resume: bool = False,
            before_sample: Callable[[str], None] | None = None,
            after_sample: Callable[[str, bool], None] | None = None) -> dict:
    """Adapter owns locally loaded DINO/SAM weights and model-specific normalization.

    Adapter returns dense dino_features [B,768,h,w], sam_masks [B,K,h,w], and
    original-image-coordinate rois [N,5]. Both teachers receive multiples of 14.
    """
    import torch
    from PIL import Image
    from prism.data.transforms import resize_inputs, sample_roi_targets

    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "index.json"
    settings = {"teacher_size": teacher_size, "center_crop": center_crop, "fp16": fp16}
    previous = {}
    if resume and index_path.exists():
        old = json.loads(index_path.read_text(encoding="utf-8"))
        if (old.get("split_ledger_sha256") != ledger.digest or
                old.get("extraction_settings") != settings or
                old.get("teachers") != ["DINOv2-Base", "SAM 2.1-Base"]):
            raise ValueError("Cannot resume a cache with different provenance/settings")
        previous = {row["sample_id"]: row for row in old["samples"]}
    records = dict(previous)
    created = skipped = 0

    def save_index() -> None:
        temporary = index_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "teachers": ["DINOv2-Base", "SAM 2.1-Base"], "teacher_size": teacher_size,
            "extraction_settings": settings, "split_ledger_sha256": ledger.digest,
            "samples": [records[key] for key in sorted(records)]}, indent=2), encoding="utf-8")
        temporary.replace(index_path)

    with torch.inference_mode():
        for sample_id, filename in sources.items():
            if before_sample:
                before_sample(sample_id)
            entry = ledger.require(sample_id, split)
            source = Path(filename)
            if hashlib.sha256(source.read_bytes()).hexdigest() != entry.sha256:
                raise ValueError("Source no longer matches frozen split")
            if sample_id in previous:
                row = previous[sample_id]
                path = (output / row["shard"]).resolve()
                if not path.is_relative_to(output.resolve()) or row["source_sha256"] != entry.sha256:
                    raise ValueError("Invalid resumed cache identity/path")
                if path.exists():
                    if hashlib.sha256(path.read_bytes()).hexdigest() != row["cache_sha256"]:
                        raise ValueError("Resumed teacher cache checksum mismatch")
                    valid = (teacher_adapter.validate_cached(sample_id, entry.sha256)
                             if hasattr(teacher_adapter, "validate_cached") else True)
                    if valid:
                        skipped += 1
                        if after_sample:
                            after_sample(sample_id, False)
                        continue
            with Image.open(source) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()[None] / 255.0
            x_low, teacher_input = resize_inputs(tensor, teacher_size, center_crop=center_crop)
            if hasattr(teacher_adapter, "set_sample"):
                teacher_adapter.set_sample(sample_id, entry.sha256)
            targets = teacher_adapter(teacher_input, source_size=rgb.shape[:2])
            dino = targets["dino_features"]
            if dino.shape[1] != 768:
                raise ValueError("Use DINOv2-Base, not a different teacher size")
            sampling_h, sampling_w = targets.get("sampling_size", rgb.shape[:2])
            sampled = sample_roi_targets(dino, targets.get("sampling_rois", targets["rois"]),
                                         image_height=sampling_h, image_width=sampling_w)
            # Content hashes are safe filenames independent of arbitrary sample IDs.
            shard = output / f"{entry.sha256}.npz"
            arrays = {"x_low": x_low[0].cpu().numpy(), "dino_features": sampled[..., 0, 0].cpu().numpy(),
                      "sam_masks": targets["sam_masks"][0].cpu().numpy(), "rois": targets["rois"].cpu().numpy()}
            if fp16:
                arrays = {key: value.astype(np.float16) if np.issubdtype(value.dtype, np.floating)
                          else value for key, value in arrays.items()}
            temporary = shard.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(shard)
            records[sample_id] = {"sample_id": sample_id, "source_sha256": entry.sha256,
                                 "shard": shard.name, "cache_sha256": hashlib.sha256(shard.read_bytes()).hexdigest()}
            save_index()  # Resume at image granularity, including after interruption.
            created += 1
            if after_sample:
                after_sample(sample_id, True)
    save_index()
    return {"created": created, "skipped": skipped, "samples": len(records)}


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
