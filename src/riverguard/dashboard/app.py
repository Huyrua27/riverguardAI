"""Dashboard server (FastAPI) — serves the Decision Support Dashboard live.

Rebuilds ``dashboard.html`` on every page load (so newly processed videos show up)
and serves it together with the media it references: the outputs folder and the
raw videos the live overlay is drawn on. Requires the ``dashboard`` extra
(``pip install -e '.[dashboard]'``). For a zero-dependency demo you can instead
open ``outputs/dashboard.html`` directly.

    python -m riverguard.dashboard.app --output-dir outputs
    # → http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    from riverguard.utils.console import utf8_console

    utf8_console()
    parser = argparse.ArgumentParser(description="RiverGuard AI dashboard server")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cameras", default="configs/cameras.yaml")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--raw-dir", default="data/raw", help="thư mục video gốc")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    try:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse, RedirectResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Install dashboard extra: pip install -e '.[dashboard]'") from e

    from riverguard.config import load_config
    from riverguard.dashboard.build import build_dashboard, build_manifest

    dcfg = load_config(args.config).get("dashboard", {})
    host = args.host or dcfg.get("host", "127.0.0.1")
    port = args.port or dcfg.get("port", 8000)
    out_dir, raw_dir = Path(args.output_dir), Path(args.raw_dir)

    app = FastAPI(title="RiverGuard AI")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/queue")
    def queue() -> JSONResponse:
        """Network-wide collection priority queue (zone-level decisions)."""
        m = build_manifest(out_dir, args.config, args.cameras)
        return JSONResponse(m["queue"])

    @app.get("/api/cameras")
    def cameras() -> JSONResponse:
        m = build_manifest(out_dir, args.config, args.cameras)
        return JSONResponse([
            {"id": c["id"], "name": c["name"], "status": c["status"],
             "hotspot": c["a"]["hotspot"] if c["a"] else None,
             "kpis": c["a"]["kpis"] if c["a"] else None}
            for c in m["cameras"]
        ])

    @app.get("/")
    def index() -> RedirectResponse:
        build_dashboard(out_dir, args.config, args.cameras)  # always current
        return RedirectResponse(f"/{out_dir.name}/dashboard.html")

    # dashboard.html lives in outputs/ and references ../data/raw/*.mp4 — mount
    # both at matching URL paths so those relative links resolve.
    app.mount(f"/{out_dir.name}", StaticFiles(directory=str(out_dir)), name="outputs")
    if raw_dir.exists():
        app.mount("/" + raw_dir.as_posix().strip("/"), StaticFiles(directory=str(raw_dir)),
                  name="raw")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
