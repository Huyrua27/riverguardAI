"""Static HTML report generator — the Decision Dashboard, offline edition.

Builds a single ``report.html`` from a pipeline results dict: per-frame overlay +
Pollution Risk Map, a ranked Priority List and summary stats. Needs no web
server (good for demos and for attaching to the competition submission); the
FastAPI app in ``app.py`` serves the same data live.
"""
from __future__ import annotations

import html
from pathlib import Path

_PRIORITY_BADGE = {
    "HIGH": "#e5484d",
    "MEDIUM": "#f5a623",
    "LOW": "#e0c000",
}


def _priority_rows(frame: dict, top_k: int = 5) -> str:
    rows = []
    for d in frame.get("decisions", [])[:top_k]:
        color = _PRIORITY_BADGE.get(d["priority"], "#888")
        center = d.get("meta", {}).get("center_px", ["-", "-"])
        rows.append(
            f"<tr><td>{html.escape(d['region_id'])}</td>"
            f"<td><span class='badge' style='background:{color}'>{d['priority']}</span></td>"
            f"<td>{d['risk_score']:.3f}</td>"
            f"<td>{int(d['probability'] * 100)}%</td>"
            f"<td>{d['lead_time_seconds']}s</td>"
            f"<td>{center[0]}, {center[1]}</td></tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='6' class='muted'>Không có vùng cảnh báo</td></tr>")
    return "\n".join(rows)


def _frame_section(frame: dict) -> str:
    name = html.escape(str(frame.get("name", frame.get("frame_id"))))
    vis = frame.get("vis")
    img = (f"<img src='{html.escape(vis)}' alt='overlay'/>" if vis
           else "<p class='muted'>no overlay</p>")
    n_det = len(frame.get("detections", []))
    n_track = len(frame.get("tracks", []))
    return f"""
    <section class="card">
      <h3>{name}</h3>
      <div class="grid2">
        <div>{img}</div>
        <div>
          <p><b>{n_det}</b> vật thể phát hiện · <b>{n_track}</b> track</p>
          <table>
            <thead><tr><th>Vùng</th><th>Ưu tiên</th><th>Risk</th>
              <th>Xác suất</th><th>Lead</th><th>Tâm (px)</th></tr></thead>
            <tbody>{_priority_rows(frame)}</tbody>
          </table>
        </div>
      </div>
    </section>"""


def generate_report(results: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = results.get("summary", {})
    prompts = results.get("detection_prompts") or []
    classes = s.get("class_counts", {})
    alerts = s.get("alert_cells", {})

    class_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in classes.items()
    ) or "<tr><td colspan='2' class='muted'>—</td></tr>"

    # Video source → embed annotated clip + cumulative risk map + peak alerts.
    # Image source → one card per frame.
    if results.get("is_video"):
        video = results.get("video")
        risk_map = results.get("risk_map")
        vid_html = (f"<video src='{html.escape(video)}' controls style='width:100%;"
                    f"border-radius:10px'></video>" if video else "<p class='muted'>—</p>")
        rm_html = (f"<img src='{html.escape(risk_map)}' alt='risk map'/>"
                   if risk_map else "<p class='muted'>—</p>")
        alert_rows = []
        for d in results.get("top_alerts", [])[:10]:
            color = _PRIORITY_BADGE.get(d["priority"], "#888")
            c = d.get("meta", {}).get("center_px", ["-", "-"])
            alert_rows.append(
                f"<tr><td>{html.escape(d['region_id'])}</td>"
                f"<td><span class='badge' style='background:{color}'>{d['priority']}</span></td>"
                f"<td>{d['risk_score']:.3f}</td><td>{int(d['probability']*100)}%</td>"
                f"<td>{d['lead_time_seconds']}s</td><td>{c[0]}, {c[1]}</td></tr>"
            )
        alert_body = "\n".join(alert_rows) or "<tr><td colspan='6' class='muted'>—</td></tr>"
        frames_html = f"""
    <div class="grid2">
      <section class="card"><h3>🎬 Video annotated (overlay + tracking)</h3>{vid_html}</section>
      <section class="card"><h3>🔥 Pollution Risk Map (tích luỹ cả clip)</h3>{rm_html}</section>
    </div>
    <section class="card"><h3>🏆 Vùng ưu tiên (peak alerts toàn clip)</h3>
      <table><thead><tr><th>Vùng</th><th>Ưu tiên</th><th>Risk</th>
        <th>Xác suất</th><th>Lead</th><th>Tâm (px)</th></tr></thead>
      <tbody>{alert_body}</tbody></table></section>"""
    else:
        frames_html = "\n".join(_frame_section(fr) for fr in results.get("frames", []))

    doc = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RiverGuard AI — Decision Dashboard</title>
<style>
  :root {{ --bg:#0f1420; --card:#182031; --ink:#e8edf5; --muted:#8b98ad; --line:#263247; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif; }}
  header {{ padding:24px 20px; border-bottom:1px solid var(--line); }}
  h1 {{ margin:0 0 4px; font-size:22px; }}
  .sub {{ color:var(--muted); font-size:14px; }}
  main {{ max-width:1100px; margin:0 auto; padding:20px; }}
  .kpis {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0 24px; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:14px 18px; min-width:150px; }}
  .kpi b {{ font-size:26px; display:block; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
    padding:16px; margin-bottom:20px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  @media (max-width:760px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
  img {{ width:100%; border-radius:10px; border:1px solid var(--line); }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th,td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; }}
  .badge {{ color:#111; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:700; }}
  .muted {{ color:var(--muted); }}
  code {{ background:#0b0f18; padding:2px 6px; border-radius:6px; }}
</style></head>
<body>
<header>
  <h1>🌊 RiverGuard AI — Decision Dashboard</h1>
  <div class="sub">Observe → Understand → Predict → Act ·
    stages=<code>{html.escape(results.get('stages',''))}</code> ·
    nguồn=<code>{html.escape(str(results.get('source','')))}</code></div>
</header>
<main>
  <div class="kpis">
    <div class="kpi"><b>{results.get('num_frames',0)}</b>frame xử lý</div>
    <div class="kpi"><b>{s.get('total_detections',0)}</b>vật thể phát hiện</div>
    <div class="kpi"><b>{results.get('fps','–')}</b>FPS</div>
    <div class="kpi"><b style="color:#e5484d">{alerts.get('HIGH',0)}</b>ô HIGH</div>
  </div>

  <div class="grid2">
    <div class="card"><h3>Phân bố loại rác</h3>
      <table><thead><tr><th>Class</th><th>Số lượng</th></tr></thead>
      <tbody>{class_rows}</tbody></table></div>
    <div class="card"><h3>Vocabulary (open-vocab prompts)</h3>
      <p class="muted">{html.escape(', '.join(prompts)) if prompts else '—'}</p>
      <p class="muted" style="margin-top:12px">Cảnh báo ô lưới: HIGH {alerts.get('HIGH',0)}
        · MEDIUM {alerts.get('MEDIUM',0)} · LOW {alerts.get('LOW',0)}</p></div>
  </div>

  <h2>{'Kết quả video' if results.get('is_video') else 'Chi tiết theo frame'}</h2>
  {frames_html}
  <p class="muted" style="margin-top:24px">Sinh tự động bởi RiverGuard AI pipeline.</p>
</main></body></html>"""

    report_path = out_dir / "report.html"
    report_path.write_text(doc, encoding="utf-8")
    return report_path
