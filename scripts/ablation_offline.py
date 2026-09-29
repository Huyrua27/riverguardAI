"""Offline ablation on stored detections — no GPU, fully reproducible.

Re-runs only the downstream stages on the detections already saved in
``outputs/video/*/results.json`` so every variant sees *identical* input:

Tracking (label-free proxies — IDF1/MOTA need ground-truth IDs we do not have yet):
  A. IoU-only greedy tracker (original baseline)
  B. A + constant-velocity prediction
  C. B + centre-distance gating (current default)
  Metrics: #tracks, mean track length, #tracks lasting >= 3 frames,
           share of detections linked into such tracks, #motion vectors.

Hotspot prediction:
  old = extrapolate pixels over the full 20-min alert horizon
  new = 10 s kinematic lookahead, out-of-view projections kept in place
  Metric: share of predicted risk mass on the outer ring of the grid
          (spurious border hotspots).

    python scripts/ablation_offline.py --outputs outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from riverguard.motion import MotionAnalyzer
from riverguard.prediction import HotspotPredictor
from riverguard.tracking import Tracker
from riverguard.types import Detection, MotionVector, Track
from riverguard.utils.metrics import iou


class IoUOnlyTracker:
    """The original baseline: greedy IoU association, no motion model."""

    def __init__(self, match_thresh: float = 0.3, track_buffer: int = 30) -> None:
        self.match_thresh, self.track_buffer = match_thresh, track_buffer
        self._next, self._tracks = 0, {}

    def update(self, dets: list[Detection]) -> list[Track]:
        cands = sorted(((iou(t.bbox, d.bbox), tid, i)
                        for tid, (t, _) in self._tracks.items() for i, d in enumerate(dets)
                        if iou(t.bbox, d.bbox) >= self.match_thresh), reverse=True)
        used_t, used_d, out = set(), set(), []
        for _, tid, i in cands:
            if tid in used_t or i in used_d:
                continue
            prev = self._tracks[tid][0]
            tr = Track(tid, dets[i].bbox, dets[i].cls, dets[i].score, prev.age + 1, prev.hits + 1)
            self._tracks[tid] = [tr, 0]
            used_t.add(tid)
            used_d.add(i)
            out.append(tr)
        for i, d in enumerate(dets):
            if i not in used_d:
                tr = Track(self._next, d.bbox, d.cls, d.score)
                self._tracks[self._next] = [tr, 0]
                self._next += 1
                out.append(tr)
        for tid in [t for t in self._tracks if t not in used_t]:
            self._tracks[tid][1] += 1
            if self._tracks[tid][1] > self.track_buffer:
                del self._tracks[tid]
        return out


class ClampingPredictor(HotspotPredictor):
    """The original predictor: extrapolate over the whole alert horizon at an
    assumed 30 fps and clamp off-frame projections onto the border cells."""

    def predict(self, tracks, motion, frame_shape):
        h, w = frame_shape
        grid = [[0.0] * self.gw for _ in range(self.gh)]
        mv = {m.track_id: m for m in motion}
        for t in tracks:
            cx, cy = t.center
            m = mv.get(t.track_id)
            if m is not None:
                cx += m.dx * self.horizon_seconds * 30.0
                cy += m.dy * self.horizon_seconds * 30.0
            gx = min(self.gw - 1, max(0, int(cx / w * self.gw)))
            gy = min(self.gh - 1, max(0, int(cy / h * self.gh)))
            grid[gy][gx] += t.score
        peak = max(max(r) for r in grid)
        grid = [[v / peak for v in r] for r in grid] if peak > 0 else grid
        return type("F", (), {"grid": grid})()


def _dets(frame: dict) -> list[Detection]:
    return [Detection(tuple(d["bbox"]), d["score"], d["cls"]) for d in frame["detections"]]


def track_stats(frames: list[dict], tracker) -> dict:
    motion = MotionAnalyzer(smoothing_window=5)
    lengths: dict[int, int] = {}
    n_det = n_vec = 0
    for fr in frames:
        tracks = tracker.update(_dets(fr))
        n_vec += len(motion.update(tracks))
        n_det += len(tracks)
        for t in tracks:
            lengths[t.track_id] = lengths.get(t.track_id, 0) + 1
    long_ids = {k for k, v in lengths.items() if v >= 3}
    linked = sum(v for k, v in lengths.items() if k in long_ids)
    return {
        "tracks": len(lengths),
        "mean_len": round(sum(lengths.values()) / max(1, len(lengths)), 2),
        "tracks_ge3": len(long_ids),
        "linked_pct": round(100 * linked / max(1, n_det), 1),
        "motion_vectors": n_vec,
    }


def border_share(frames: list[dict], predictor: HotspotPredictor) -> float:
    total = border = 0.0
    g = predictor.gh
    for fr in frames:
        tracks = [Track(**t) for t in fr["tracks"]]
        motion = [MotionVector(**m) for m in fr["motion"]]
        grid = predictor.predict(tracks, motion, (fr["height"], fr["width"])).grid
        for y, row in enumerate(grid):
            for x, v in enumerate(row):
                total += v
                if y in (0, g - 1) or x in (0, g - 1):
                    border += v
    return round(100 * border / total, 1) if total else 0.0


def main() -> None:
    from riverguard.utils.console import utf8_console

    utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--save", default="outputs/ablation_offline.json")
    args = ap.parse_args()

    report = {}
    for rf in sorted(Path(args.outputs, "video").glob("*/results.json")):
        r = json.loads(rf.read_text(encoding="utf-8"))
        frames, fps = r["frames"], (r.get("source_fps") or 30.0) / (r.get("stride") or 1)
        report[rf.parent.name] = {
            "frames": len(frames),
            "A_iou_only": track_stats(frames, IoUOnlyTracker()),
            "B_plus_velocity": track_stats(frames, Tracker(track_thresh=0.0, max_center_dist=0.0)),
            "C_plus_centre_gate": track_stats(frames, Tracker(track_thresh=0.0)),
            "border_pct_old": border_share(frames, ClampingPredictor(horizon_seconds=1200)),
            "border_pct_new": border_share(frames, HotspotPredictor(fps=fps, lookahead_seconds=10)),
        }

    keys = ["tracks", "mean_len", "tracks_ge3", "linked_pct", "motion_vectors"]
    print(f"{'video':<8}{'variant':<20}" + "".join(f"{k:>15}" for k in keys))
    for vid, rep in report.items():
        for var in ("A_iou_only", "B_plus_velocity", "C_plus_centre_gate"):
            print(f"{vid:<8}{var:<20}" + "".join(f"{rep[var][k]:>15}" for k in keys))
    print("\nRisk mass on grid border (%):  old (20-min extrapolation) -> new (10 s lookahead)")
    for vid, rep in report.items():
        print(f"  {vid}: {rep['border_pct_old']}% -> {rep['border_pct_new']}%")
    Path(args.save).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {args.save}")


if __name__ == "__main__":
    main()
