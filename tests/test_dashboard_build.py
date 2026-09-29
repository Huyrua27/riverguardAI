"""Decision dashboard build: registry join, media paths, network queue, HTML."""
from __future__ import annotations

import json
import re
from pathlib import Path

from riverguard.dashboard.build import build_dashboard, build_manifest

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _write_run(out: Path, cam_id: str, raw: Path) -> None:
    run = out / "video" / cam_id
    run.mkdir(parents=True)
    (run / "base_frame.jpg").write_bytes(b"jpg")
    box = (10, 10, 30, 30)
    frames = [{
        "t": float(t), "frame_id": t, "width": 300, "height": 300,
        "detections": [{"bbox": box, "score": 0.5, "cls": "bottle"}],
        "tracks": [{"track_id": 0, "bbox": box, "cls": "bottle", "score": 0.5}],
        "motion": [], "decisions": [],
    } for t in range(4)]
    (run / "results.json").write_text(json.dumps({
        "source": str(raw), "is_video": True, "stride": 1, "source_fps": 1.0,
        "frame_size": [300, 300], "num_frames": 4, "fps": 2.0,
        "base_frame": "base_frame.jpg", "risk_grid": [[1.0]], "frames": frames,
    }), encoding="utf-8")


def test_manifest_joins_registry_and_builds_queue(tmp_path):
    raw = tmp_path / "raw" / "cam1.mp4"
    raw.parent.mkdir()
    raw.write_bytes(b"mp4")
    out = tmp_path / "outputs"
    _write_run(out, "cam1", raw)
    cams = tmp_path / "cameras.yaml"
    cams.write_text(
        "cameras:\n"
        "  - {id: cam1, name: Cầu A, lat: 10.7, lon: 106.6}\n"
        "  - {id: drone01, name: Drone 01, type: drone}\n", encoding="utf-8")

    m = build_manifest(out, CONFIG, cams)
    by_id = {c["id"]: c for c in m["cameras"]}
    assert by_id["cam1"]["status"] == "online" and by_id["cam1"]["a"]["hotspot"]["id"] == "A"
    assert by_id["drone01"]["status"] == "offline" and by_id["drone01"]["a"] is None
    assert by_id["cam1"]["media"]["raw"] == "../raw/cam1.mp4"
    assert by_id["cam1"]["media"]["base"] == "video/cam1/base_frame.jpg"
    assert m["queue"][0]["cam"] == "cam1" and m["queue"][0]["zone"] == "A"

    html = build_dashboard(out, CONFIG, cams).read_text(encoding="utf-8")
    assert "/*__DATA__*/" not in html
    data = json.loads(re.search(r"const RG = (.*?);\n", html).group(1))
    assert data["cameras"][0]["name"] == "Cầu A"
