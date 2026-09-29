"""End-to-end orchestration: Observe → Understand → Predict → Act.

Reads a source (image / folder / video / stream) frame-by-frame and runs the
full RiverGuard pipeline, emitting ranked decisions plus visual deliverables
(overlay + Pollution Risk Map) and a machine-readable JSON report.

Each stage is pluggable and configured via YAML. The ``stages`` argument selects
an ablation level so the same code path serves the ablation matrix in
``docs/EVALUATION.md``.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from loguru import logger

from riverguard.config import load_config
from riverguard.decision import DecisionEngine
from riverguard.detection import build_detector
from riverguard.motion import MotionAnalyzer
from riverguard.prediction import HotspotPredictor
from riverguard.sources import count_frames_hint, iter_frames
from riverguard.tracking import Tracker

# Ablation levels — each adds one reasoning stage on top of the previous.
STAGE_PRESETS = {
    "detection_only": {"track": False, "motion": False, "predict": False, "decide": False},
    "det_track": {"track": True, "motion": False, "predict": False, "decide": False},
    "det_track_motion": {"track": True, "motion": True, "predict": False, "decide": False},
    "det_track_motion_pred": {"track": True, "motion": True, "predict": True, "decide": False},
    "proposed": {"track": True, "motion": True, "predict": True, "decide": True},
}


class Pipeline:
    """Wires the five stages and runs them per frame at a chosen ablation level."""

    def __init__(self, cfg: dict, stages: str = "proposed") -> None:
        self.cfg = cfg
        self.stages = STAGE_PRESETS[stages]
        self.stage_name = stages

        self.detector = build_detector(cfg["detection"])
        self.tracker = Tracker(
            method=cfg["tracking"]["method"],
            track_thresh=cfg["tracking"].get("track_thresh", 0.3),
            match_thresh=cfg["tracking"].get("match_thresh", 0.3),
            track_buffer=cfg["tracking"].get("track_buffer", 30),
            max_center_dist=cfg["tracking"].get("max_center_dist", 1.5),
        )
        self.motion = MotionAnalyzer(smoothing_window=cfg["motion"]["smoothing_window"])
        self.predictor = HotspotPredictor(
            grid_size=tuple(cfg["prediction"]["grid_size"]),
            horizon_seconds=cfg["prediction"]["horizon_seconds"],
            lookahead_seconds=cfg["prediction"].get("lookahead_seconds", 10.0),
        )
        self.engine = DecisionEngine(
            alert_threshold=cfg["decision"]["alert_threshold"],
            risk_weights=cfg["decision"]["risk_weights"],
        )

    def process_frame(self, frame) -> dict:
        """Run the enabled stages on one BGR frame; return a result dict."""
        h, w = frame.shape[:2]
        detections = self.detector.detect(frame)                       # Observe

        tracks = self.tracker.update(detections) if self.stages["track"] else []
        motion = self.motion.update(tracks) if self.stages["motion"] else []

        forecast = None
        decisions = []
        if self.stages["predict"]:
            forecast = self.predictor.predict(tracks, motion, (h, w))  # Predict
            if self.stages["decide"]:
                decisions = self.engine.rank(forecast)                 # Act

        return {
            "width": w,
            "height": h,
            "detections": [asdict(d) for d in detections],
            "tracks": [asdict(t) for t in tracks],
            "motion": [asdict(m) for m in motion],
            "forecast": (
                {"grid": forecast.grid, "horizon_seconds": forecast.horizon_seconds}
                if forecast is not None else None
            ),
            "decisions": [asdict(d) for d in decisions],
        }


def run(config_path: str, source: str, stages: str = "proposed",
        output_dir: str | None = None, save_vis: bool = True,
        make_report: bool = True, stride: int = 1,
        max_frames: int | None = None) -> dict:
    """Run the pipeline over ``source`` and return a full results dict.

    For a video source, an annotated ``*.mp4`` is written (instead of one JPG
    per frame), a cumulative Pollution Risk Map is accumulated over the whole
    clip, and only a handful of evenly-spaced frames are kept for the report.
    """
    import numpy as np

    from riverguard import visualize
    from riverguard.sources import imwrite, is_video_source, video_fps
    from riverguard.types import Decision, MotionVector, Track

    cfg = load_config(config_path)
    pipe = Pipeline(cfg, stages=stages)
    gh, gw = pipe.predictor.gh, pipe.predictor.gw

    out_dir = Path(output_dir or "outputs")
    vis_dir = out_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    video = is_video_source(source)
    src_fps = video_fps(source) if video else 1.0
    out_fps = max(1.0, src_fps / max(1, stride)) if video else 1.0
    # Motion vectors are px per *processed* frame → predictor needs that rate.
    pipe.predictor.fps = src_fps / max(1, stride) if video else 1.0
    total_hint = count_frames_hint(source)
    logger.info(f"Source={source} stages={stages} video={video} stride={stride} "
                f"frames≈{total_hint or '?'}")

    frames_out: list[dict] = []
    cum_grid = np.zeros((gh, gw), dtype=np.float64)   # cumulative risk map
    base_frame = None                                 # canvas for the risk map
    writer = None
    tmp_video = None
    t0 = time.perf_counter()

    def _overlay(frame, res):
        tracks = [Track(**t) for t in res["tracks"]]
        motion = [MotionVector(**m) for m in res["motion"]]
        ov = visualize.draw_overlay(frame, tracks, motion)
        if res["forecast"]:
            ov = visualize.heatmap_overlay(ov, res["forecast"]["grid"])
            ov = visualize.draw_hotspot_markers(ov, [Decision(**d) for d in res["decisions"]])
        return ov

    for idx, name, frame in iter_frames(source, stride=stride, max_frames=max_frames):
        res = pipe.process_frame(frame)
        res["frame_id"] = idx
        res["name"] = name
        res["t"] = round(idx / src_fps, 3) if video else float(len(frames_out))
        if base_frame is None:
            base_frame = frame.copy()
        if res["forecast"]:
            cum_grid += np.asarray(res["forecast"]["grid"], dtype=np.float64)

        if save_vis:
            overlay = _overlay(frame, res)
            if video:
                if writer is None:
                    import cv2
                    tmp_video = Path(tempfile.mkdtemp()) / "annotated.mp4"
                    h, w = overlay.shape[:2]
                    writer = cv2.VideoWriter(str(tmp_video),
                                             cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
                writer.write(overlay)
            else:
                stem = Path(name).stem or f"frame_{idx:06d}"
                vis_path = vis_dir / f"{stem}.jpg"
                imwrite(vis_path, overlay)
                res["vis"] = str(vis_path.relative_to(out_dir)).replace("\\", "/")

        top = res["decisions"][0] if res["decisions"] else None
        n_det = len(res["detections"])
        if top and top["priority"] == "HIGH":
            logger.warning(
                f"[{name}] {n_det} objects — {top['region_id']}: risk "
                f"{top['risk_score']} ({top['priority']}), lead {top['lead_time_seconds']}s"
            )
        else:
            logger.info(f"[{name}] {n_det} objects detected")
        frames_out.append(res)

    elapsed = time.perf_counter() - t0
    n = len(frames_out)

    # Finalise annotated video (write to ASCII temp, then move — Unicode-safe).
    video_rel = None
    if writer is not None:
        writer.release()
        import shutil
        stem = Path(source).stem or "video"
        dest = out_dir / f"{stem}_annotated.mp4"
        # OpenCV writes MPEG-4 Part 2 ("mp4v"), which browsers won't play —
        # re-encode to H.264 when an ffmpeg binary is available.
        encoded = _to_h264(tmp_video)
        shutil.move(str(encoded or tmp_video), str(dest))
        video_rel = dest.name
        logger.info(f"Annotated video ({'H.264' if encoded else 'mp4v'}): {dest}")

    # Clean still of the scene — the canvas the dashboard draws the risk map on.
    base_rel = None
    if save_vis and base_frame is not None:
        bp = out_dir / "base_frame.jpg"
        imwrite(bp, base_frame)
        base_rel = bp.name

    # Cumulative Pollution Risk Map over the whole run.
    risk_map_rel = None
    risk_grid = None
    if cum_grid.max() > 0:
        risk_grid = np.round(cum_grid / cum_grid.max(), 4).tolist()
    if save_vis and base_frame is not None and risk_grid is not None:
        risk_img = visualize.heatmap_overlay(base_frame, risk_grid, alpha=0.55)
        rp = out_dir / "risk_map.jpg"
        imwrite(rp, risk_img)
        risk_map_rel = rp.name

    # Peak alerts across the whole clip (for the video dashboard).
    peak_alerts = sorted(
        (dec for fr in frames_out for dec in fr["decisions"]),
        key=lambda d: d["risk_score"], reverse=True,
    )
    seen_regions: set[str] = set()
    top_alerts = []
    for dec in peak_alerts:
        if dec["region_id"] in seen_regions:
            continue
        seen_regions.add(dec["region_id"])
        top_alerts.append(dec)
        if len(top_alerts) >= 10:
            break

    results = {
        "source": source,
        "config": config_path,
        "stages": stages,
        "is_video": video,
        "stride": stride,
        "num_frames": n,
        "elapsed_seconds": round(elapsed, 3),
        "fps": round(n / elapsed, 2) if elapsed > 0 else None,
        "detection_prompts": cfg["detection"].get("prompts"),
        "source_fps": src_fps if video else None,
        "frame_size": ([frames_out[0]["width"], frames_out[0]["height"]]
                       if frames_out else None),
        "video": video_rel,
        "base_frame": base_rel,
        "risk_map": risk_map_rel,
        "risk_grid": risk_grid,
        "summary": _summarize(frames_out),
        "top_alerts": top_alerts,
        "frames": frames_out,
    }

    # Keep results.json lean for long videos: the per-frame 32×32 forecast grid
    # is already folded into `risk_grid`, and only the top cells per frame matter.
    if video:
        for fr in frames_out:
            if fr.get("forecast"):
                fr["forecast"] = {"horizon_seconds": fr["forecast"]["horizon_seconds"]}
            fr["decisions"] = fr["decisions"][:5]

    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=None if video else 2)
    logger.info(f"Wrote {results_path} ({n} frames, {results['fps']} FPS)")

    if make_report:
        try:
            from riverguard.dashboard.report import generate_report
            report_path = generate_report(results, out_dir)
            logger.info(f"Report: {report_path}")
            results["report"] = str(report_path)
        except Exception as e:  # pragma: no cover - report is best-effort
            logger.warning(f"Report generation skipped: {e}")

    return results


def _to_h264(src: Path) -> Path | None:
    """Re-encode ``src`` to browser-playable H.264; ``None`` if ffmpeg is missing/fails."""
    import subprocess

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - optional dependency
        return None
    dest = src.with_name(src.stem + "_h264.mp4")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
           "-movflags", "+faststart", str(dest)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    except (subprocess.SubprocessError, OSError) as e:  # pragma: no cover
        logger.warning(f"H.264 re-encode failed, keeping mp4v: {e}")
        return None
    return dest


def _summarize(frames: list[dict]) -> dict:
    class_counts: dict[str, int] = {}
    total_det = 0
    high = med = low = 0
    for fr in frames:
        for d in fr["detections"]:
            class_counts[d["cls"]] = class_counts.get(d["cls"], 0) + 1
            total_det += 1
        for dec in fr["decisions"]:
            if dec["priority"] == "HIGH":
                high += 1
            elif dec["priority"] == "MEDIUM":
                med += 1
            else:
                low += 1
    return {
        "total_detections": total_det,
        "avg_detections_per_frame": round(total_det / len(frames), 2) if frames else 0,
        "class_counts": dict(sorted(class_counts.items(), key=lambda kv: -kv[1])),
        "alert_cells": {"HIGH": high, "MEDIUM": med, "LOW": low},
    }


def main() -> None:
    from riverguard.utils.console import utf8_console

    utf8_console()
    parser = argparse.ArgumentParser(description="RiverGuard AI end-to-end pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", required=True,
                        help="image / folder / video path, camera index or stream URL")
    parser.add_argument("--stages", default="proposed", choices=list(STAGE_PRESETS))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--stride", type=int, default=1,
                        help="giữ mỗi frame thứ N (video) — vd 15 ≈ 2 fps trên clip 30 fps")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="dừng sau N frame đã xử lý")
    parser.add_argument("--no-vis", action="store_true", help="skip saving visual overlays")
    parser.add_argument("--no-report", action="store_true", help="skip HTML report")
    args = parser.parse_args()
    run(args.config, args.source, stages=args.stages, output_dir=args.output_dir,
        save_vis=not args.no_vis, make_report=not args.no_report,
        stride=args.stride, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
