"""Multi-object tracking — assigns persistent IDs across frames.

A lightweight SORT-family tracker (no Kalman filter, dependency-free,
deterministic):

1. **Predict** — each track carries a smoothed velocity; its box is shifted
   forward by ``velocity × frames_since_seen`` before matching (constant-velocity
   motion model), so drifting waste and slow camera pans still overlap.
2. **Associate** — detections are matched greedily to predicted boxes, first by
   IoU (``match_thresh``), then — for small objects whose boxes stop overlapping
   between sampled frames — by centre distance, normalised by the box diagonal
   (``max_center_dist``). IoU matches always outrank distance matches.
3. **Manage** — unmatched detections start new tracks; tracks unseen for
   ``track_buffer`` frames are retired.

Set ``method`` to ``bytetrack`` / ``botsort`` to defer to Ultralytics' trackers
once a fine-tuned detector is wired in.
"""
from __future__ import annotations

from dataclasses import dataclass

from riverguard.types import BBox, Detection, Track, bbox_center
from riverguard.utils.metrics import iou


@dataclass
class _State:
    track: Track
    lost: int = 0          # frames since last matched
    vx: float = 0.0        # smoothed centre velocity, px / processed frame
    vy: float = 0.0


def _shift(bbox: BBox, dx: float, dy: float) -> BBox:
    x1, y1, x2, y2 = bbox
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def _diag(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(1.0, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)


class Tracker:
    def __init__(self, method: str = "iou", track_thresh: float = 0.3,
                 match_thresh: float = 0.3, track_buffer: int = 30,
                 max_center_dist: float = 1.5, velocity_smoothing: float = 0.5,
                 **kwargs) -> None:
        self.method = method
        self.track_thresh = track_thresh        # min detection score to start a track
        self.match_thresh = match_thresh        # min IoU to associate det <-> track
        self.track_buffer = track_buffer        # frames a lost track survives
        self.max_center_dist = max_center_dist  # centre gate, in box diagonals
        self.alpha = velocity_smoothing         # EMA weight of the newest velocity
        self.kwargs = kwargs
        self._next_id = 0
        self._states: dict[int, _State] = {}

    def reset(self) -> None:
        self._next_id = 0
        self._states.clear()

    def _predicted(self, st: _State) -> BBox:
        steps = st.lost + 1
        return _shift(st.track.bbox, st.vx * steps, st.vy * steps)

    def update(self, detections: list[Detection]) -> list[Track]:
        """Associate this frame's detections with existing tracks."""
        dets = [d for d in detections if d.score >= self.track_thresh]
        track_ids = list(self._states.keys())

        # Candidate (score, track_id, det_idx). IoU matches score in (1, 2],
        # centre-distance matches in [0, 1) — so overlap always wins.
        candidates: list[tuple[float, int, int]] = []
        for tid in track_ids:
            pred = self._predicted(self._states[tid])
            pcx, pcy = bbox_center(pred)
            gate = self.max_center_dist * _diag(pred)
            for di, det in enumerate(dets):
                overlap = iou(pred, det.bbox)
                if overlap >= self.match_thresh:
                    candidates.append((1.0 + overlap, tid, di))
                    continue
                dcx, dcy = det.center
                dist = ((dcx - pcx) ** 2 + (dcy - pcy) ** 2) ** 0.5
                if dist < gate:
                    candidates.append((1.0 - dist / gate, tid, di))
        candidates.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        active: list[Track] = []

        for _score, tid, di in candidates:
            if tid in matched_tracks or di in matched_dets:
                continue
            det = dets[di]
            st = self._states[tid]
            steps = st.lost + 1
            ocx, ocy = st.track.center
            ncx, ncy = det.center
            vx, vy = (ncx - ocx) / steps, (ncy - ocy) / steps
            if st.track.hits == 1:  # first velocity sample: take it as-is
                st.vx, st.vy = vx, vy
            else:
                st.vx = self.alpha * vx + (1 - self.alpha) * st.vx
                st.vy = self.alpha * vy + (1 - self.alpha) * st.vy
            st.track = Track(track_id=tid, bbox=det.bbox, cls=det.cls, score=det.score,
                             age=st.track.age + 1, hits=st.track.hits + 1)
            st.lost = 0
            matched_tracks.add(tid)
            matched_dets.add(di)
            active.append(st.track)

        # Unmatched detections -> new tracks.
        for di, det in enumerate(dets):
            if di in matched_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            new = Track(track_id=tid, bbox=det.bbox, cls=det.cls, score=det.score,
                        age=0, hits=1)
            self._states[tid] = _State(track=new)
            active.append(new)

        # Age unmatched tracks; retire the stale ones.
        for tid in track_ids:
            if tid in matched_tracks:
                continue
            st = self._states[tid]
            st.lost += 1
            st.track.age += 1
            if st.lost > self.track_buffer:
                del self._states[tid]

        active.sort(key=lambda t: t.track_id)
        return active
