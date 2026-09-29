"""Zone-level decision analytics."""
from __future__ import annotations

from riverguard.analytics import analyze, categorize, compass

CFG = {
    "prediction": {"horizon_seconds": 1200, "lookahead_seconds": 10},
    "decision": {"risk_weights": {"density": 0.4, "convergence": 0.4, "confidence": 0.2},
                 "alert_threshold": 0.7},
    "analytics": {"zones": [3, 3], "recency_half_life": 0.33,
                  "compensate_camera_motion": False},
    "categories": {"Plastic": ["bottle", "bag"], "Foam": ["foam"]},
}


def _frame(t, tracks, motion=()):
    return {
        "t": t, "width": 300, "height": 300,
        "detections": [{"bbox": b, "score": 0.5, "cls": c} for _, b, c in tracks],
        "tracks": [{"track_id": i, "bbox": b, "cls": c, "score": 0.5} for i, b, c in tracks],
        "motion": [{"track_id": i, "dx": dx, "dy": dy, "speed": 0.0} for i, dx, dy in motion],
    }


def _results(frames):
    return {"is_video": True, "stride": 1, "source_fps": 1.0, "frame_size": [300, 300],
            "frames": frames}


def test_compass_directions():
    assert compass(0, -1)[1] == "N"
    assert compass(1, 0)[1] == "E"
    assert compass(1, -1)[1] == "NE"
    assert compass(0, -1, heading_deg=90)[1] == "E"   # camera facing east


def test_categorize_maps_open_vocab_labels():
    assert categorize("Bottle", CFG["categories"]) == "Plastic"
    assert categorize("styrofoam", CFG["categories"]) == "Other"


def test_busiest_zone_becomes_hotspot():
    box_a = (10, 10, 30, 30)       # top-left → Zone A
    frames = [_frame(t, [(0, box_a, "bottle"), (1, box_a, "foam")]) for t in range(5)]
    out = analyze(_results(frames), CFG)
    assert out["hotspot"]["id"] == "A"
    assert out["queue"][0]["zone"] == "A"
    assert all(z["risk"] == 0 for z in out["zones"] if z["id"] != "A")
    assert {c["category"] for c in out["composition"]} == {"Plastic", "Foam"}


def test_inflow_drives_convergence_and_eta():
    # Object in Zone A drifting right at 10 px/s; 10 s lookahead → lands in Zone B.
    frames = [_frame(t, [(7, (40 + 10 * t, 40, 60 + 10 * t, 60), "bag")],
                     motion=[(7, 10.0, 0.0)] if t else []) for t in range(3)]
    out = analyze(_results(frames), CFG)
    zone_b = next(z for z in out["zones"] if z["id"] == "B")
    assert zone_b["inflow"] >= 1 and zone_b["convergence"] > 0
    assert zone_b["eta_s"] is not None and zone_b["eta_s"] > 0
    assert out["motion"]["compass"] == "E"
    assert "7" in out["trajectories"]


def test_confidence_reflects_detector_quality_not_just_clip_length():
    box = (10, 10, 30, 30)

    def run(score):
        frames = [_frame(t, [(t, box, "bottle")]) for t in range(40)]   # 40 distinct tracks
        for f in frames:
            f["tracks"][0]["score"] = score
        return analyze(_results(frames), CFG)["hotspot"]

    weak, strong = run(0.03), run(0.6)       # zero-shot-like vs fine-tuned-like
    assert strong["confidence"] > 0.9
    assert weak["confidence"] < 0.15
    assert weak["evidence"] == strong["evidence"]


def test_density_change_needs_a_baseline():
    box = (10, 10, 30, 30)
    # 1 early observation, many late ones → "new activity", not "+900%".
    frames = [_frame(0, [(0, box, "bag")])] + [_frame(t, [(t, box, "bag")]) for t in range(10, 20)]
    zone_a = analyze(_results(frames), CFG)["zones"][0]
    assert zone_a["density_change_pct"] is None and zone_a["new_activity"]


def test_empty_run_has_no_hotspot():
    out = analyze(_results([]), CFG)
    assert out["hotspot"] is None and out["queue"] == []
