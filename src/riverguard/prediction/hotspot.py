"""Hotspot prediction: forecast where waste will accumulate.

Baseline is a grid-based spatio-temporal accumulator: each tracked object is
projected forward along its drift vector, then stamped onto a coarse grid with a
small Gaussian footprint (waste clusters, it is not a point). The result is
normalised to a [0, 1] risk map. Swap for ConvLSTM / temporal heatmap
forecasting as the model matures.

Two different time scales are kept apart on purpose:

- ``lookahead_seconds`` — how far objects are *kinematically* extrapolated
  inside the camera view (seconds). Linear extrapolation is only meaningful
  over a short window; projecting 20 minutes ahead pushes every moving object
  to the frame border and creates spurious edge hotspots.
- ``horizon_seconds`` — the alert lead time reported to operators
  ("zone B will accumulate within ~20 min"). It labels the forecast; it is not
  used to extrapolate pixels.

Projections that leave the frame are kept at the object's current position:
the camera cannot see where it goes, so we do not invent a border hotspot.
"""
from __future__ import annotations

from riverguard.types import HotspotForecast, MotionVector, Track

# 3x3 Gaussian stamp — waste spreads to neighbouring cells, not a single pixel.
_STAMP = [
    [0.25, 0.5, 0.25],
    [0.50, 1.0, 0.50],
    [0.25, 0.5, 0.25],
]


class HotspotPredictor:
    def __init__(self, grid_size: tuple[int, int] = (32, 32),
                 horizon_seconds: int = 1200, fps: float = 30.0,
                 lookahead_seconds: float = 10.0) -> None:
        self.gh, self.gw = grid_size
        self.horizon_seconds = horizon_seconds
        self.fps = fps                    # processed frames per second (src_fps / stride)
        self.lookahead_seconds = lookahead_seconds

    def predict(
        self,
        tracks: list[Track],
        motion: list[MotionVector],
        frame_shape: tuple[int, int],
    ) -> HotspotForecast:
        h, w = frame_shape
        grid = [[0.0 for _ in range(self.gw)] for _ in range(self.gh)]
        motion_by_id = {m.track_id: m for m in motion}
        steps = self.lookahead_seconds * self.fps  # motion vectors are px / processed frame

        for t in tracks:
            cx, cy = t.center
            m = motion_by_id.get(t.track_id)
            if m is not None:
                px, py = cx + m.dx * steps, cy + m.dy * steps
                if 0 <= px < w and 0 <= py < h:  # only forecast inside the view
                    cx, cy = px, py
            gx = min(self.gw - 1, max(0, int(cx / w * self.gw)))
            gy = min(self.gh - 1, max(0, int(cy / h * self.gh)))
            # Stamp a small Gaussian footprint weighted by detection confidence.
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = gy + dy, gx + dx
                    if 0 <= yy < self.gh and 0 <= xx < self.gw:
                        grid[yy][xx] += t.score * _STAMP[dy + 1][dx + 1]

        # Normalise to [0, 1].
        peak = max((max(row) for row in grid), default=0.0)
        if peak > 0:
            grid = [[round(v / peak, 6) for v in row] for row in grid]
        return HotspotForecast(grid=grid, horizon_seconds=self.horizon_seconds,
                               frame_shape=(h, w))
