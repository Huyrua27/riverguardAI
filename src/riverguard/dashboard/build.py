"""Build the RiverGuard **Decision Support Dashboard** — the demo product.

Collects every processed run under ``<output_dir>/video/*/results.json``, joins it
with the camera registry (``configs/cameras.yaml``), runs the zone-level
analytics (:mod:`riverguard.analytics`) and writes one self-contained
``dashboard.html`` next to the outputs.

The page opens straight from disk (no server — handy for judging); the GIS view
additionally needs internet for the Leaflet library and map tiles. The same page
is served live by :mod:`riverguard.dashboard.app`.

    python -m riverguard.dashboard.build --output-dir outputs
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import yaml

from riverguard.analytics import analyze
from riverguard.config import load_config

TEMPLATE = Path(__file__).parent / "templates" / "dashboard.html"


def _rel(target: Path, base: Path) -> str | None:
    """Path of ``target`` relative to ``base`` (URL form), or None if missing."""
    if not target.exists():
        return None
    return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()


def _load_cameras(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("cameras") or [])


def _camera_entry(meta: dict, run_dir: Path | None, output_dir: Path, cfg: dict) -> dict:
    """One camera: registry metadata + media paths + analytics (if it has data)."""
    entry = {
        "id": meta["id"],
        "name": meta.get("name", meta["id"]),
        "type": meta.get("type", "fixed"),
        "lat": meta.get("lat"),
        "lon": meta.get("lon"),
        "heading_deg": meta.get("heading_deg"),
        "placeholder": bool(meta.get("placeholder_location", False)),
        "a": None,
        "media": {},
        "meta": {},
        "grid": None,
    }
    results = None
    if run_dir is not None and (run_dir / "results.json").exists():
        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))

    entry["status"] = meta.get("status") or ("online" if results else "offline")
    if not results:
        return entry

    src = Path(str(results.get("source", "")))
    base = results.get("base_frame") or results.get("risk_map")
    entry["media"] = {
        "raw": _rel(src, output_dir) if results.get("is_video") else None,
        "annotated": _rel(run_dir / results["video"], output_dir) if results.get("video") else None,
        "base": _rel(run_dir / base, output_dir) if base else None,
    }
    entry["grid"] = results.get("risk_grid")
    entry["a"] = analyze(results, cfg, heading_deg=meta.get("heading_deg"))
    entry["meta"] = {
        "duration": entry["a"]["duration_s"],
        "n_frames": results.get("num_frames"),
        "stride": results.get("stride"),
        "pipeline_fps": results.get("fps"),
    }
    return entry


def build_manifest(output_dir: Path, config_path: str | Path,
                   cameras_path: str | Path | None) -> dict:
    cfg = load_config(config_path)
    registry = _load_cameras(Path(cameras_path) if cameras_path else None)
    runs = {p.parent.name: p.parent for p in sorted((output_dir / "video").glob("*/results.json"))}

    cameras = []
    seen: set[str] = set()
    for meta in registry:
        cameras.append(_camera_entry(meta, runs.get(meta["id"]), output_dir, cfg))
        seen.add(meta["id"])
    for run_id, run_dir in runs.items():  # runs not in the registry still show up
        if run_id not in seen:
            cameras.append(_camera_entry({"id": run_id}, run_dir, output_dir, cfg))

    queue = sorted(
        ({"cam": c["id"], "cam_name": c["name"], **q}
         for c in cameras if c["a"] for q in c["a"]["queue"]),
        key=lambda q: q["risk"], reverse=True,
    )
    det = cfg["detection"]
    return {
        "generated_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "detector": {
            "model": f"{det.get('model', 'yolo')} ({Path(str(det.get('weights', ''))).name})",
            "zero_shot": str(det.get("model", "")).lower() in {"yoloworld", "yolo-world", "world"},
            "prompts": det.get("prompts") or det.get("classes") or [],
        },
        "horizon_s": cfg["prediction"]["horizon_seconds"],
        "alert_threshold": cfg["decision"]["alert_threshold"],
        "weights": cfg["decision"]["risk_weights"],
        "categories": cfg.get("categories", {}),
        "cameras": cameras,
        "queue": queue,
        "totals": {
            "detections": sum(c["a"]["kpis"]["detections"] for c in cameras if c["a"]),
        },
    }


def build_dashboard(output_dir: str | Path, config_path: str | Path = "configs/default.yaml",
                    cameras_path: str | Path | None = "configs/cameras.yaml") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(output_dir, config_path, cameras_path)
    # "</" inside a <script> block would end it early — escape it.
    data = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", data)
    out = output_dir / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    from riverguard.utils.console import utf8_console

    utf8_console()
    import argparse

    parser = argparse.ArgumentParser(description="Build the RiverGuard decision dashboard")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cameras", default="configs/cameras.yaml")
    args = parser.parse_args()
    out = build_dashboard(args.output_dir, args.config, args.cameras)
    print(f"Built {out}. Mở trực tiếp trong trình duyệt, hoặc chạy "
          f"`python -m riverguard.dashboard.app` để phục vụ qua web.")


if __name__ == "__main__":
    main()
