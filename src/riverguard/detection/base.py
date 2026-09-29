"""Detector interface. Concrete backends (YOLO, RT-DETR) implement `detect`."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from riverguard.types import Detection


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return waste detections for a single BGR frame."""
        raise NotImplementedError
