"""CPU YOLO11-L scoring on decoded RGB, using only frozen train identities.

COUNCIL VISIBILITY: D is per-image COCO bbox AP@IoU=.50, NOT dataset mAP.
COCOeval uses 101 recall points, useCats=1, area=all, maxDets=100, all 80
categories. Mean only precision entries >=0 (categories with no non-ignored
GT in this image are excluded by COCOeval). Thus detections of absent classes
do not reduce D. Crowd/ignore and matching follow pycocotools. Empty detections
score zero if evaluable GT exists; no evaluable GT is a FAIL, not a zero label.
Predictions on the full decoded letterbox are inverse-transformed and clipped
to original source coordinates before comparison to ALL image annotations.
This convention, detector confidence/NMS settings, and the utility direction
must be visible to council; it is not a claim about held-out detection quality.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.metadata
import io
from pathlib import Path

import numpy as np

from tier1a.common import sha256

COCO_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
            39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75,
            76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90)
SETTINGS = {"model": "YOLO11-L", "device": "cpu", "imgsz": 640, "conf": .001,
            "iou": .7, "max_det": 100, "augment": False, "half": False,
            "batch": 1, "metric": "per-image COCO bbox AP50; 101 recall points",
            "area": "all", "useCats": 1, "absent_classes": "excluded", "seed": 0}


def source_box(xyxy: list[float], geometry: dict) -> list[float]:
    x1, y1, x2, y2 = xyxy
    x1, x2 = (np.clip((x - geometry["left"]) / geometry["scale_x"],
                     0, geometry["source_width"]) for x in (x1, x2))
    y1, y2 = (np.clip((y - geometry["top"]) / geometry["scale_y"],
                     0, geometry["source_height"]) for y in (y1, y2))
    return [float(x1), float(y1), float(max(0, x2 - x1)), float(max(0, y2 - y1))]


class CocoAP50:
    def __init__(self, annotations: Path, weights: Path, image_ids: set[int], *, device: str | None = None):
        if not weights.is_file():
            raise FileNotFoundError(f"Provision local YOLO11-L weights first: {weights}")
        import torch
        from ultralytics import YOLO
        from pycocotools.coco import COCO
        torch.set_num_threads(4)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        with redirect_stdout(io.StringIO()):
            self.coco = COCO(str(annotations))
        if not image_ids <= set(self.coco.imgs) or tuple(sorted(self.coco.cats)) != COCO_IDS:
            raise ValueError("Require selected train IDs and standard COCO80 categories")
        self.image_ids = image_ids
        self.model = YOLO(str(weights), task="detect")
        if len(self.model.names) != 80:
            raise ValueError("YOLO weights must use COCO80 class order")
        # Reject differently sized YOLO models even if renamed yolo11l.pt.
        self.parameters = sum(p.numel() for p in self.model.model.parameters())
        yaml_file = str(getattr(self.model.model, "yaml", {}).get("yaml_file", ""))
        scale = getattr(self.model.model, "yaml", {}).get("scale")
        if ("yolo11" not in yaml_file.lower() or scale != "l"
                or not 20_000_000 < self.parameters < 30_000_000):
            raise ValueError(f"Expected YOLO11-L architecture; got {yaml_file}, {scale}, {self.parameters}")
        for index, category_id in enumerate(COCO_IDS):
            if self.model.names[index] != self.coco.cats[category_id]["name"]:
                raise ValueError("Detector COCO class names/order mismatch")
        self.provenance = {**SETTINGS, "device": self.device, "weights_sha256": sha256(weights),
                           "parameters": self.parameters,
                           "versions": {name: importlib.metadata.version(name)
                                        for name in ("torch", "ultralytics", "pycocotools", "numpy", "Pillow")}}

    def __call__(self, rgb: np.ndarray, frame: dict) -> float:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        image_id, geometry = frame["image_id"], frame["geometry"]
        if image_id not in self.image_ids:
            raise ValueError("Scoring outside the frozen training selection")
        # Ultralytics ndarray input is BGR. Its boxes are returned in full input
        # pixel coordinates after the detector's own resize/letterbox inversion.
        prediction = self.model.predict(source=np.ascontiguousarray(rgb[..., ::-1]),
            device=self.device, imgsz=640, conf=.001, iou=.7, max_det=100, augment=False,
            half=False, verbose=False, save=False, batch=1)[0]
        rows = []
        for box, score, cls in zip(prediction.boxes.xyxy.cpu().numpy(),
                                   prediction.boxes.conf.cpu().numpy(),
                                   prediction.boxes.cls.cpu().numpy()):
            bbox = source_box(box, geometry)
            if bbox[2] and bbox[3]:
                rows.append({"image_id": image_id, "category_id": COCO_IDS[int(cls)],
                             "bbox": bbox, "score": float(score)})
        with redirect_stdout(io.StringIO()):
            if rows:
                detections = self.coco.loadRes(rows)
            else:
                detections = COCO()
                detections.dataset = {"images": [self.coco.imgs[image_id]],
                                      "categories": list(self.coco.cats.values()), "annotations": []}
                detections.createIndex()
            evaluator = COCOeval(self.coco, detections, "bbox")
            evaluator.params.imgIds = [image_id]
            evaluator.params.catIds = list(COCO_IDS)
            evaluator.params.iouThrs = np.array([.5])
            evaluator.params.recThrs = np.linspace(0, 1, 101)
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate()
            evaluator.accumulate()
        precision = evaluator.eval["precision"][0, :, :, 0, 2]
        valid = precision[precision >= 0]
        if not len(valid):
            raise ValueError("Image has no evaluable non-ignored GT; D undefined")
        return float(valid.mean())
