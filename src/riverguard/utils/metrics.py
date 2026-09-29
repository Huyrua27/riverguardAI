"""Evaluation metrics used across detection / tracking / prediction."""
from __future__ import annotations


def iou(box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mae(pred: list[float], target: list[float]) -> float:
    if not pred:
        return 0.0
    return sum(abs(p - t) for p, t in zip(pred, target, strict=False)) / len(pred)


def rmse(pred: list[float], target: list[float]) -> float:
    if not pred:
        return 0.0
    return (sum((p - t) ** 2 for p, t in zip(pred, target, strict=False)) / len(pred)) ** 0.5
