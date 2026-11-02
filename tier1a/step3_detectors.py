"""Dual-detector scoring (YOLOv11-L + Faster R-CNN ResNet50-FPN v2) for Tier 1a Step 3.

Decodes each reconstructed frame once and evaluates both detectors on the same
1080p letterboxed RGB image, mapping predicted boxes back to source image
coordinates via source_box() before running pycocotools.cocoeval.COCOeval.

Computes both:
  1. Per-image AP50 (for paired_bd_bootstrap with unit='image' and per-CRF Holm tests).
  2. Pooled dataset-level COCO AP@[.50:.95], AP50, and AP_small across the 50 pilot images.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import importlib.metadata
import io
from pathlib import Path

import numpy as np

from tier1a.common import sha256
from tier1a.oracle_detector import COCO_IDS, CocoAP50, source_box

COCO_ID_SET = frozenset(COCO_IDS)


def evaluate_per_image_ap50(coco, image_id: int, rows: list[dict], *,
                           allow_undefined: bool = False) -> float | None:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with redirect_stdout(io.StringIO()):
        if rows:
            detections = coco.loadRes(rows)
        else:
            detections = COCO()
            detections.dataset = {
                "images": [coco.imgs[image_id]],
                "categories": list(coco.cats.values()),
                "annotations": [],
            }
            detections.createIndex()
        evaluator = COCOeval(coco, detections, "bbox")
        evaluator.params.imgIds = [image_id]
        evaluator.params.catIds = list(COCO_IDS)
        evaluator.params.iouThrs = np.array([0.5])
        evaluator.params.recThrs = np.linspace(0, 1, 101)
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][0, :, :, 0, 2]
    valid = precision[precision >= 0]
    if not len(valid):
        if allow_undefined:
            return None
        raise ValueError(f"Image {image_id} has no evaluable non-ignored GT; D undefined")
    return float(valid.mean())


def evaluate_pooled_coco_metrics(coco, image_ids: list[int], all_rows: list[dict]) -> dict[str, float]:
    """Run standard pycocotools COCOeval across a pooled set of images (e.g. 50 pilot images)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ordered_ids = sorted(set(int(i) for i in image_ids))
    with redirect_stdout(io.StringIO()):
        if all_rows:
            detections = coco.loadRes(all_rows)
        else:
            detections = COCO()
            detections.dataset = {
                "images": [coco.imgs[iid] for iid in ordered_ids],
                "categories": list(coco.cats.values()),
                "annotations": [],
            }
            detections.createIndex()
        evaluator = COCOeval(coco, detections, "bbox")
        evaluator.params.imgIds = ordered_ids
        evaluator.params.catIds = list(COCO_IDS)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    return {
        "ap_50_95": float(stats[0]),
        "ap50": float(stats[1]),
        "ap75": float(stats[2]),
        "ap_small": float(stats[3]),
        "ap_medium": float(stats[4]),
        "ap_large": float(stats[5]),
    }


class FasterRCNNScorer:
    """Faster R-CNN ResNet50-FPN v2 (torchvision COCO_V1) detector for cross-detector transfer."""

    def __init__(self, coco, image_ids: set[int], *, device: str | None = None, weights_path: Path | None = None):
        import torch
        import torchvision
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )

        self.coco = coco
        self.image_ids = set(image_ids)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        weights_enum = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        if weights_path is not None and weights_path.is_file():
            self.model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            w_sha = sha256(weights_path)
        else:
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights_enum)
            # Hash serialized state dict deterministically for provenance
            buf = io.BytesIO()
            torch.save(self.model.state_dict(), buf)
            w_sha = hashlib.sha256(buf.getvalue()).hexdigest()

        self.model.to(self.device)
        self.model.eval()
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.provenance = {
            "model": "fasterrcnn_resnet50_fpn_v2",
            "weights": "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
            "device": self.device,
            "score_thresh": 0.001,
            "max_det": 100,
            "parameters": self.parameters,
            "weights_sha256": w_sha,
            "versions": {
                "torch": importlib.metadata.version("torch"),
                "torchvision": importlib.metadata.version("torchvision"),
            },
        }

    def predict_rows(self, rgb: np.ndarray, frame: dict) -> list[dict]:
        import torch

        image_id, geometry = int(frame["image_id"]), frame["geometry"]
        if image_id not in self.image_ids:
            raise ValueError(f"Image {image_id} outside allowed evaluation split")
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(
            device=self.device, dtype=torch.float32
        ) / 255.0
        with torch.no_grad():
            out = self.model([tensor])[0]
        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()
        order = np.argsort(-scores, kind="stable")[:100]
        rows = []
        for idx in order:
            score = float(scores[idx])
            if score < 0.001:
                continue
            cat_id = int(labels[idx])
            if cat_id not in COCO_ID_SET:
                continue
            bbox = source_box(boxes[idx].tolist(), geometry)
            if bbox[2] > 0 and bbox[3] > 0:
                rows.append({
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": bbox,
                    "score": score,
                })
        return rows

    def score_frame(self, rgb: np.ndarray, frame: dict) -> tuple[float, list[dict]]:
        rows = self.predict_rows(rgb, frame)
        ap50 = evaluate_per_image_ap50(self.coco, int(frame["image_id"]), rows)
        return ap50, rows


class DualDetectorScorer:
    """Wraps primary YOLOv11-L and secondary Faster R-CNN ResNet50-FPN v2 on shared decoded RGB."""

    def __init__(
        self,
        annotations: Path,
        yolo_weights: Path,
        image_ids: set[int],
        *,
        device: str | None = None,
        frcnn_weights: Path | None = None,
    ):
        self.yolo = CocoAP50(annotations, yolo_weights, image_ids, device=device)
        self.frcnn = FasterRCNNScorer(
            self.yolo.coco,
            image_ids,
            device=self.yolo.device,
            weights_path=frcnn_weights,
        )
        self.coco = self.yolo.coco
        self.provenance = {
            "primary_detector": self.yolo.provenance,
            "secondary_detector": self.frcnn.provenance,
        }

    def predict_yolo_rows(self, rgb: np.ndarray, frame: dict) -> list[dict]:
        image_id, geometry = int(frame["image_id"]), frame["geometry"]
        if image_id not in self.yolo.image_ids:
            raise ValueError("Scoring outside allowed image_ids")
        prediction = self.yolo.model.predict(
            source=np.ascontiguousarray(rgb[..., ::-1]),
            device=self.yolo.device,
            imgsz=640,
            conf=0.001,
            iou=0.7,
            max_det=100,
            augment=False,
            half=False,
            verbose=False,
            save=False,
            batch=1,
        )[0]
        rows = []
        for box, score, cls in zip(
            prediction.boxes.xyxy.cpu().numpy(),
            prediction.boxes.conf.cpu().numpy(),
            prediction.boxes.cls.cpu().numpy(),
        ):
            bbox = source_box(box, geometry)
            if bbox[2] and bbox[3]:
                rows.append({
                    "image_id": image_id,
                    "category_id": COCO_IDS[int(cls)],
                    "bbox": bbox,
                    "score": float(score),
                })
        return rows

    def score_both(self, rgb: np.ndarray, frame: dict, *, allow_undefined_ap50: bool = False) -> dict:
        image_id = int(frame["image_id"])
        yolo_rows = self.predict_yolo_rows(rgb, frame)
        yolo_ap50 = evaluate_per_image_ap50(self.coco, image_id, yolo_rows,
                                           allow_undefined=allow_undefined_ap50)
        frcnn_rows = self.frcnn.predict_rows(rgb, frame)
        frcnn_ap50 = evaluate_per_image_ap50(self.coco, image_id, frcnn_rows,
                                            allow_undefined=allow_undefined_ap50)
        return {
            "yolo11l": {"ap50": yolo_ap50, "rows": yolo_rows},
            "faster_rcnn": {"ap50": frcnn_ap50, "rows": frcnn_rows},
        }
