"""System-level tests: detector factory, frame sources, report generation."""
from __future__ import annotations

import numpy as np

from riverguard.dashboard.report import generate_report
from riverguard.detection import YOLODetector, YOLOWorldDetector, build_detector
from riverguard.sources import imwrite, iter_frames


def test_build_detector_open_vocab():
    det = build_detector({"model": "yoloworld", "weights": "w.pt",
                          "prompts": ["plastic bottle"], "conf_threshold": 0.05})
    assert isinstance(det, YOLOWorldDetector)
    assert det.prompts == ["plastic bottle"]
    assert det._model is None  # lazy — no weights loaded on construction


def test_build_detector_closed_vocab():
    det = build_detector({"model": "yolo", "weights": "w.pt",
                          "classes": ["foam"], "conf_threshold": 0.3})
    assert isinstance(det, YOLODetector)


def test_iter_frames_single_image(tmp_path):
    img = np.zeros((32, 48, 3), dtype=np.uint8)
    p = tmp_path / "frame.jpg"
    imwrite(p, img)
    frames = list(iter_frames(str(p)))
    assert len(frames) == 1
    idx, name, frame = frames[0]
    assert idx == 0 and frame.shape == (32, 48, 3)


def test_iter_frames_directory(tmp_path):
    for i in range(3):
        imwrite(tmp_path / f"img_{i}.png", np.zeros((8, 8, 3), dtype=np.uint8))
    frames = list(iter_frames(str(tmp_path)))
    assert len(frames) == 3


def test_generate_report(tmp_path):
    results = {
        "source": "data/raw", "stages": "proposed", "num_frames": 1, "fps": 3.0,
        "detection_prompts": ["plastic bottle"],
        "summary": {"total_detections": 2, "class_counts": {"plastic bottle": 2},
                    "alert_cells": {"HIGH": 1, "MEDIUM": 0, "LOW": 3}},
        "frames": [{
            "name": "a.jpg", "frame_id": 0,
            "detections": [{"cls": "plastic bottle"}, {"cls": "plastic bottle"}],
            "tracks": [{"track_id": 0}],
            "decisions": [{"region_id": "cell_1_2", "priority": "HIGH", "risk_score": 0.9,
                           "probability": 0.9, "lead_time_seconds": 1200,
                           "meta": {"center_px": [10, 20]}}],
            "vis": "vis/a.jpg",
        }],
    }
    path = generate_report(results, tmp_path)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "RiverGuard AI" in text and "cell_1_2" in text
