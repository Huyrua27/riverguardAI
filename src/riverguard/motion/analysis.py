"""Motion analysis: drift direction & speed from track trajectories.

Estimates each object's drift (dx, dy, speed) in pixels/frame by smoothing the
per-step displacement of its centroid over a sliding window. Trajectory-based;
swap in dense optical flow (Farneback / RAFT) for camera-motion compensation as
the model matures.
"""
from __future__ import annotations

from collections import defaultdict, deque

from riverguard.types import MotionVector, Track


class MotionAnalyzer:
    def __init__(self, smoothing_window: int = 5) -> None:
        self.smoothing_window = max(2, smoothing_window)
        # track_id -> recent centroids
        self._history: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.smoothing_window)
        )

    @staticmethod
    def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def update(self, tracks: list[Track]) -> list[MotionVector]:
        seen = {t.track_id for t in tracks}
        vectors: list[MotionVector] = []
        for t in tracks:
            cx, cy = self._center(t.bbox)
            hist = self._history[t.track_id]
            hist.append((cx, cy))
            if len(hist) < 2:
                continue  # need at least two observations for a vector
            # Mean per-step displacement over the window -> robust drift estimate.
            pts = list(hist)
            steps = len(pts) - 1
            dx = (pts[-1][0] - pts[0][0]) / steps
            dy = (pts[-1][1] - pts[0][1]) / steps
            speed = (dx**2 + dy**2) ** 0.5
            vectors.append(MotionVector(t.track_id, dx, dy, speed))
        # Forget tracks that disappeared this frame.
        for tid in list(self._history.keys()):
            if tid not in seen:
                del self._history[tid]
        return vectors
