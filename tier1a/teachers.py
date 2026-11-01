"""The two pinned teacher identities and per-image resumable NPZ artifacts."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np

from tier1a.common import StepFailure, download, read_json, sha256, write_json

DINO_REPO_URL = "https://github.com/facebookresearch/dinov2/zipball/main"
DINO_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
SAM_SOURCE = "https://github.com/facebookresearch/sam2.git"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
RECIPE = {"version": 1, "teacher_size": 518, "center_crop": True,
          "sam_max_masks": 32, "sam_prompt_order": "area descending, annotation ID ascending",
          "sam_internal_size": 1024, "sam_checkpoint": "sam2.1_hiera_base_plus.pt",
          "dino_model": "dinov2_vitb14", "floating_cache_dtype": "float16"}


def save_npz(path: Path, arrays: dict[str, np.ndarray], binding: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    write_json(path.with_suffix(".json"), {"sha256": sha256(path), "binding": binding})


def load_npz(path: Path, binding: dict) -> dict[str, np.ndarray] | None:
    receipt = path.with_suffix(".json")
    if not path.exists() or not receipt.exists():
        return None  # Interrupted write, safely regenerate; no completion flag.
    meta = read_json(receipt)
    if meta["binding"] != binding or meta["sha256"] != sha256(path):
        raise StepFailure("teacher_cache", f"Checksum or provenance mismatch: {path}")
    with np.load(path, allow_pickle=False) as shard:
        return {key: shard[key].copy() for key in shard.files}


def crop_prompts(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    from prism.data.transforms import center_crop_box

    left, top, right, bottom = center_crop_box(row["height"], row["width"])
    source, boxes, ids = [], [], []
    scale = 518 / (right - left)
    for annotation in row["annotations"]:
        x, y, w, h = annotation["bbox"]
        x1, y1, x2, y2 = max(left, x), max(top, y), min(right, x + w), min(bottom, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        source.append([0, x1, y1, x2, y2])
        boxes.append([(x1 - left) * scale, (y1 - top) * scale,
                      (x2 - left) * scale, (y2 - top) * scale])
        ids.append(annotation["id"])
        if len(boxes) == 32:
            break
    return (np.asarray(source, dtype=np.float32).reshape(-1, 5),
            np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(ids, dtype=np.int64), (left, top, right, bottom))


class TeacherAdapter:
    def __init__(self, output: Path, weights: Path, rows: list[dict], ledger_sha: str,
                 selection_sha: str, *, device: str, offline: bool = False):
        self.output, self.weights, self.device, self.offline = output, weights, device, offline
        self.rows = {row["sample_id"]: row for row in rows}
        self.base_binding = {"ledger_sha256": ledger_sha, "selection_sha256": selection_sha, "recipe": RECIPE}
        self.dino: Any = None
        self.sam: Any = None
        self.sample_id = ""
        self.source_sha = ""
        self.stats = {"dinov2_created": 0, "dinov2_skipped": 0, "sam21_created": 0, "sam21_skipped": 0}

    def binding(self, source_sha: str) -> dict:
        return dict(self.base_binding, source_sha256=source_sha)

    def validate_cached(self, sample_id: str, source_sha: str) -> bool:
        for teacher in ("dinov2", "sam21"):
            if load_npz(self.output / teacher / f"{sample_id}.npz", self.binding(source_sha)) is None:
                return False
        for teacher in ("dinov2", "sam21"):
            self.stats[teacher + "_skipped"] += 1
        return True

    def set_sample(self, sample_id: str, source_sha: str) -> None:
        self.sample_id, self.source_sha = sample_id, source_sha

    def _provenance(self, name: str, value: dict) -> None:
        path = self.output / "model_provenance.json"
        data = read_json(path) if path.exists() else {}
        if name in data and data[name] != value:
            raise StepFailure("teacher_weights", f"Refusing changed {name} model provenance on resume")
        data[name] = value
        write_json(path, data)

    def _load_dino(self) -> Any:
        import hashlib
        import torch

        checkpoint = self.weights / "dinov2_vitb14_pretrain.pth"
        download(DINO_URL, checkpoint, offline=self.offline)
        hub = self.weights / "torch_hub"
        torch.hub.set_dir(str(hub))
        source = hub / "facebookresearch_dinov2_main"
        try:
            if source.exists():
                model = torch.hub.load(str(source), "dinov2_vitb14", source="local", pretrained=False)
            elif self.offline:
                raise StepFailure("dinov2_source", f"Offline source cache miss; URL={DINO_REPO_URL}")
            else:
                model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=False,
                                       trust_repo=True, skip_validation=True)
            source_digest = hashlib.sha256()
            for path in sorted(source.rglob("*.py")):
                source_digest.update(path.relative_to(source).as_posix().encode())
                source_digest.update(path.read_bytes())
            self._provenance("dinov2", {"url": DINO_URL, "checkpoint_sha256": sha256(checkpoint),
                "hub_source_url": DINO_REPO_URL, "source_sha256": source_digest.hexdigest()})
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            return model.eval().to(self.device)
        except Exception as exc:
            raise StepFailure("dinov2_load", f"source URL={DINO_REPO_URL}; weights URL={DINO_URL}; {exc}") from exc

    def _load_sam(self) -> Any:
        checkpoint = self.weights / "sam2.1_hiera_base_plus.pt"
        download(SAM_URL, checkpoint, offline=self.offline)
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            distribution = importlib.metadata.distribution("SAM-2")
            direct = json.loads(distribution.read_text("direct_url.json") or "{}")
            self._provenance("sam21", {"url": SAM_URL, "checkpoint_sha256": sha256(checkpoint),
                "config": SAM_CONFIG, "package_version": distribution.version, "package_origin": direct})
            model = build_sam2(SAM_CONFIG, str(checkpoint), device=self.device, mode="eval",
                               apply_postprocessing=False)
            return SAM2ImagePredictor(model)
        except Exception as exc:
            raise StepFailure("sam21_load", f"Meta package URL={SAM_SOURCE}; weights URL={SAM_URL}; {exc}") from exc

    def __call__(self, teacher_input: Any, *, source_size: tuple[int, int]) -> dict:
        import torch

        row = self.rows[self.sample_id]
        rois, boxes, annotation_ids, crop = crop_prompts(row)
        binding = self.binding(self.source_sha)
        dino_path = self.output / "dinov2" / f"{self.sample_id}.npz"
        dino = load_npz(dino_path, binding)
        if dino is None:
            if self.dino is None:
                self.dino = self._load_dino()
            tensor = teacher_input.to(self.device)
            mean = tensor.new_tensor([.485, .456, .406])[None, :, None, None]
            std = tensor.new_tensor([.229, .224, .225])[None, :, None, None]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                result = self.dino.forward_features((tensor - mean) / std)
            dino = {"cls_token": result["x_norm_clstoken"][0].cpu().numpy().astype(np.float16),
                    "patch_tokens": result["x_norm_patchtokens"][0].cpu().numpy().astype(np.float16),
                    "crop_box": np.asarray(crop, dtype=np.int32)}
            save_npz(dino_path, dino, binding)
            self.stats["dinov2_created"] += 1
        else:
            self.stats["dinov2_skipped"] += 1
        if dino["cls_token"].shape != (768,) or dino["patch_tokens"].shape != (1369, 768):
            raise StepFailure("dinov2_shape", "Expected Base CLS[768] and patch tokens[1369,768]")
        sam_path = self.output / "sam21" / f"{self.sample_id}.npz"
        sam = load_npz(sam_path, binding)
        if sam is None:
            masks = np.zeros((0, 518, 518), dtype=bool)
            scores = np.zeros(0, dtype=np.float16)
            if len(boxes):
                if self.sam is None:
                    self.sam = self._load_sam()
                view = np.rint(teacher_input[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                # The API accepts the shared 518 view; official SAM internally resizes to 1024.
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                    self.sam.set_image(view)
                    raw_masks, raw_scores, _ = self.sam.predict(box=boxes, multimask_output=False)
                masks = np.asarray(raw_masks).reshape(len(boxes), 518, 518).astype(bool)
                scores = np.asarray(raw_scores).reshape(len(boxes)).astype(np.float16)
            sam = {"masks": masks, "scores": scores, "rois": rois.astype(np.float16),
                   "prompt_boxes": boxes.astype(np.float16), "annotation_ids": annotation_ids,
                   "crop_box": np.asarray(crop, dtype=np.int32), "source_size": np.asarray(source_size, dtype=np.int32)}
            save_npz(sam_path, sam, binding)
            self.stats["sam21_created"] += 1
        else:
            self.stats["sam21_skipped"] += 1
        dense = torch.from_numpy(dino["patch_tokens"].astype(np.float32)).T.reshape(1, 768, 37, 37)
        sampling_rois = np.column_stack((np.zeros(len(boxes), dtype=np.float32), boxes))
        return {"dino_features": dense, "sam_masks": torch.from_numpy(sam["masks"])[None],
                "rois": torch.from_numpy(rois), "sampling_rois": torch.from_numpy(sampling_rois),
                "sampling_size": (518, 518)}
