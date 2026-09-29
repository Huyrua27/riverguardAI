"""Shared dataclasses passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2


def bbox_center(bbox: BBox) -> tuple[float, float]:
    """Centroid (cx, cy) of an (x1, y1, x2, y2) box."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class Detection:
    """A single waste object detected in one frame."""

    bbox: BBox  # x1, y1, x2, y2
    score: float
    cls: str

    @property
    def center(self) -> tuple[float, float]:
        return bbox_center(self.bbox)


@dataclass
class Track:
    """A detection associated with a persistent object ID over time."""

    track_id: int
    bbox: BBox
    cls: str
    score: float
    age: int = 0          # frames since first seen
    hits: int = 1         # number of frames matched

    @property
    def center(self) -> tuple[float, float]:
        return bbox_center(self.bbox)


@dataclass
class MotionVector:
    """Estimated drift of a tracked object (pixels / frame)."""

    track_id: int
    dx: float
    dy: float
    speed: float


@dataclass
class HotspotForecast:
    """Predicted risk of waste accumulation over a spatial grid."""

    grid: list[list[float]]  # risk in [0, 1] per cell
    horizon_seconds: int
    frame_shape: tuple[int, int] | None = None  # (h, w) the grid maps onto


@dataclass
class Decision:
    """Actionable output for a region."""

    region_id: str
    risk_score: float
    priority: str  # LOW | MEDIUM | HIGH
    probability: float
    lead_time_seconds: int
    meta: dict = field(default_factory=dict)
