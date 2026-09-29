"""YOLO-based waste detectors.

Two backends, both wrapping Ultralytics; the `ultralytics` import is deferred so
the package imports fine without the heavy ML extra installed.

- ``YOLODetector``      — a standard closed-vocabulary model (needs waste weights
                          trained/fine-tuned on the project classes).
- ``YOLOWorldDetector`` — an *open-vocabulary* model (YOLO-World). Detects the
                          classes you name as text prompts, **no training needed**
                          — this is what lets RiverGuard run zero-shot on real
                          footage before a fine-tuned detector exists.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from riverguard.types import Detection


def _resolve_weights(weights: str) -> str:
    """Use the local weights file if present, else fall back to the bare name.

    A bare Ultralytics model name (e.g. ``yolov8s-world.pt``) is auto-downloaded,
    so a fresh clone (where ``models/`` is gitignored) still works out of the box.
    """
    return weights if Path(weights).exists() else Path(weights).name


class YOLODetector:
    """Closed-vocabulary Ultralytics YOLO (v8/v11) / RT-DETR wrapper."""

    def __init__(self, weights: str, conf: float = 0.35, iou: float = 0.5,
                 classes: list[str] | None = None, imgsz: int = 640) -> None:
        self.weights = weights
        self.conf = conf
        self.iou = iou
        self.classes = classes or []
        self.imgsz = imgsz
        self._model = None

    def _lazy_load(self):
        if self._model is None:
            from ultralytics import YOLO  # deferred import

            self._model = YOLO(_resolve_weights(self.weights))
        return self._model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        model = self._lazy_load()
        results = model.predict(frame, conf=self.conf, iou=self.iou,
                                imgsz=self.imgsz, verbose=False)
        out: list[Detection] = []
        for r in results:
            names = r.names  # index -> class name from the model itself
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_idx = int(box.cls[0])
                name = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) \
                    else (self.classes[cls_idx] if cls_idx < len(self.classes) else str(cls_idx))
                out.append(Detection((x1, y1, x2, y2), float(box.conf[0]), name))
        return out


class YOLOWorldDetector:
    """Open-vocabulary detector (YOLO-World).

    Detects whatever you list in ``prompts`` — e.g. ``["plastic bottle",
    "plastic bag", "styrofoam", "floating trash"]`` — without any training.
    """

    def __init__(self, weights: str = "yolov8s-world.pt", prompts: list[str] | None = None,
                 conf: float = 0.03, iou: float = 0.5, imgsz: int = 1280) -> None:
        self.weights = weights
        self.prompts = prompts or ["floating trash", "plastic bottle", "plastic bag"]
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self._model = None

    def _lazy_load(self):
        if self._model is None:
            from ultralytics import YOLO  # deferred import

            self._model = YOLO(_resolve_weights(self.weights))
            self._model.set_classes(self.prompts)  # open-vocabulary prompts
        return self._model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        model = self._lazy_load()
        results = model.predict(frame, conf=self.conf, iou=self.iou,
                                imgsz=self.imgsz, verbose=False)
        out: list[Detection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_idx = int(box.cls[0])
                name = self.prompts[cls_idx] if cls_idx < len(self.prompts) else str(cls_idx)
                out.append(Detection((x1, y1, x2, y2), float(box.conf[0]), name))
        return out


def build_detector(cfg: dict):
    """Factory: build a detector from the ``detection`` config block."""
    model = str(cfg.get("model", "yolo")).lower()
    if model in {"yoloworld", "yolo-world", "world"}:
        return YOLOWorldDetector(
            weights=cfg.get("weights", "yolov8s-world.pt"),
            prompts=cfg.get("prompts"),
            conf=cfg.get("conf_threshold", 0.03),
            iou=cfg.get("iou_threshold", 0.5),
            imgsz=cfg.get("imgsz", 1280),
        )
    return YOLODetector(
        weights=cfg["weights"],
        conf=cfg.get("conf_threshold", 0.35),
        iou=cfg.get("iou_threshold", 0.5),
        classes=cfg.get("classes"),
        imgsz=cfg.get("imgsz", 640),
    )
