"""Zone-level decision analytics — turn one pipeline run into operator answers.

The per-frame pipeline scores a fine 32×32 grid. Operators think in *zones*
("send the boat to Zone B"), so this module aggregates a whole run into a coarse
zone grid (default 3×3 → Zone A…I) and answers the three dashboard questions:

- **Perception** — what is there now: overlay boxes, trajectories, composition.
- **Reasoning**  — where is it going: drift speed / direction, inflow between zones.
- **Decision**   — what to do first: per-zone risk, priority, ETA, hotspot + "why".

Risk per zone combines the three factors weighted by ``decision.risk_weights``:

- ``density``     recency-weighted presence of waste in the zone, relative to the
                  busiest zone (recent frames weigh more — early warning is about
                  what happens next, not what happened a minute ago);
- ``convergence`` share of objects whose short-term projected path *enters* the
                  zone vs. leaves it, ``in / (in + out + 1)`` (the +1 keeps a single
                  observation from reading as a certainty);
- ``confidence``  evidence × quality: ``1 − exp(−k / 5)`` over the distinct tracks
                  seen in / flowing into the zone, scaled by the zone's mean detector
                  score relative to ``analytics.confidence_ref``. A zero-shot detector
                  scoring ~0.03 therefore yields a *low* confidence, however long the
                  clip — confidence rises as the detector improves.

The result is a **risk score in [0, 1], not a calibrated probability** — the
dashboard labels it accordingly. ETA is a kinematic estimate (distance to the
zone centre ÷ drift speed) and exists only when inflow was actually observed.

When the camera pans (hand-held footage), every object appears to move with the
camera; with ``analytics.compensate_camera_motion`` the per-frame median motion
is subtracted as an ego-motion estimate before any flow statistic is computed.
"""
from __future__ import annotations

import math
from statistics import median

from riverguard.decision.scoring import DecisionEngine

_COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def compass(dx: float, dy: float, heading_deg: float | None = None) -> tuple[float, str]:
    """Bearing of an image-space vector: 0° = up, clockwise.

    With the camera's ``heading_deg`` (direction the camera faces, 0 = North),
    the bearing becomes geographic; otherwise it is relative to the frame.
    """
    ang = math.degrees(math.atan2(dx, -dy)) % 360.0
    if heading_deg is not None:
        ang = (ang + heading_deg) % 360.0
    return round(ang, 1), _COMPASS[int((ang + 22.5) // 45) % 8]


def zone_name(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else f"Z{index}"


def categorize(cls: str, mapping: dict[str, list[str]]) -> str:
    """Map an open-vocabulary label onto a project waste category."""
    key = cls.strip().lower()
    for category, names in mapping.items():
        if key in {n.strip().lower() for n in names}:
            return category
    return "Other"


def _level(value: float) -> str:
    if value >= 0.6:
        return "HIGH"
    if value >= 0.3:
        return "MEDIUM"
    return "LOW" if value > 0 else "NONE"


def analyze(results: dict, cfg: dict, heading_deg: float | None = None) -> dict:
    """Aggregate a pipeline ``results`` dict into zone-level decision data."""
    frames = results.get("frames", [])
    acfg = cfg.get("analytics", {})
    rows, cols = acfg.get("zones", [3, 3])
    half_life = float(acfg.get("recency_half_life", 0.33))
    compensate = bool(acfg.get("compensate_camera_motion", True))
    conf_ref = float(acfg.get("confidence_ref", 0.25))
    min_half = int(acfg.get("min_obs_for_trend", 3))
    horizon = int(cfg["prediction"]["horizon_seconds"])
    lookahead = float(cfg["prediction"].get("lookahead_seconds", 10.0))
    weights = cfg["decision"]["risk_weights"]
    threshold = float(cfg["decision"]["alert_threshold"])
    mpp = (cfg.get("motion") or {}).get("meters_per_pixel")
    categories = cfg.get("categories", {})

    n_zones = rows * cols
    is_video = bool(results.get("is_video"))
    stride = max(1, int(results.get("stride") or 1))
    src_fps = float(results.get("source_fps") or 30.0) if is_video else 1.0
    proc_fps = src_fps / stride if is_video else 1.0

    if frames:
        w, h = results.get("frame_size") or (frames[0]["width"], frames[0]["height"])
    else:
        w, h = 1, 1
    ts = [
        float(fr["t"]) if "t" in fr
        else (fr.get("frame_id", i) / src_fps if is_video else float(i))
        for i, fr in enumerate(frames)
    ]
    t0 = ts[0] if ts else 0.0
    t_end = ts[-1] if ts else 0.0
    duration = t_end - t0
    hl = max(1e-6, half_life * duration)
    recency = [0.5 ** ((t_end - t) / hl) if duration > 0 else 1.0 for t in ts]
    t_mid = t0 + duration / 2

    def zone_of(nx: float, ny: float) -> int:
        r = min(rows - 1, max(0, int(ny * rows)))
        c = min(cols - 1, max(0, int(nx * cols)))
        return r * cols + c

    presence = [0.0] * n_zones
    n_obs = [0] * n_zones
    first_half = [0] * n_zones
    second_half = [0] * n_zones
    inflow = [0] * n_zones
    outflow = [0] * n_zones
    inflow_eta: list[list[float]] = [[] for _ in range(n_zones)]
    zone_vec = [[0.0, 0.0, 0] for _ in range(n_zones)]
    zone_tracks: list[set] = [set() for _ in range(n_zones)]
    score_sum = [0.0] * n_zones

    speeds: list[float] = []
    vsum = [0.0, 0.0]
    overlay: list[list] = []
    trajectories: dict[int, dict] = {}
    trend: list[list[float]] = []
    composition: dict[str, int] = {}
    total_det = 0

    for i, fr in enumerate(frames):
        t, wt = ts[i], recency[i]
        dets = fr.get("detections", [])
        objs = fr.get("tracks") or [{"track_id": None, **d} for d in dets]
        motion = {m["track_id"]: (m["dx"], m["dy"]) for m in fr.get("motion", [])}
        if compensate and len(motion) >= 3:  # median motion ≈ camera ego-motion
            mdx = median(v[0] for v in motion.values())
            mdy = median(v[1] for v in motion.values())
            motion = {k: (v[0] - mdx, v[1] - mdy) for k, v in motion.items()}

        boxes = []
        for o in objs:
            x1, y1, x2, y2 = o["bbox"]
            nx, ny = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            z = zone_of(nx, ny)
            presence[z] += wt
            n_obs[z] += 1
            (first_half if t < t_mid else second_half)[z] += 1
            score_sum[z] += float(o["score"])
            tid = o.get("track_id")
            zone_tracks[z].add(tid if tid is not None else ("obs", i, len(zone_tracks[z])))
            boxes.append([tid, o["cls"], round(float(o["score"]), 3),
                          round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)])
            if tid is not None:
                tr = trajectories.setdefault(tid, {"cls": o["cls"], "pts": []})
                tr["pts"].append([round(t, 3), round(nx, 4), round(ny, 4)])

            if tid is None or tid not in motion:
                continue
            dx, dy = motion[tid]
            speed = math.hypot(dx, dy) * proc_fps  # px / s
            speeds.append(speed)
            vsum[0] += dx
            vsum[1] += dy
            zone_vec[z][0] += dx
            zone_vec[z][1] += dy
            zone_vec[z][2] += 1
            steps = lookahead * proc_fps
            px, py = nx + dx * steps / w, ny + dy * steps / h
            if 0 <= px < 1 and 0 <= py < 1:
                zp = zone_of(px, py)
                if zp != z:
                    outflow[z] += 1
                    inflow[zp] += 1
                    zr, zc = divmod(zp, cols)
                    dist = math.hypot(((zc + 0.5) / cols - nx) * w, ((zr + 0.5) / rows - ny) * h)
                    if speed > 1e-6:
                        inflow_eta[zp].append(dist / speed)

        overlay.append([round(t, 3), boxes])
        trend.append([round(t, 2), len(dets)])
        total_det += len(dets)
        for d in dets:
            cat = categorize(d["cls"], categories)
            composition[cat] = composition.get(cat, 0) + 1

    # ---- Decision: per-zone risk -------------------------------------------
    wsum = sum(weights.values()) or 1.0
    max_presence = max(presence) if presence else 0.0
    n_frames = max(1, len(frames))
    zones = []
    for z in range(n_zones):
        r, c = divmod(z, cols)
        density = presence[z] / max_presence if max_presence > 0 else 0.0
        convergence = inflow[z] / (inflow[z] + outflow[z] + 1)
        mean_score = score_sum[z] / n_obs[z] if n_obs[z] else 0.0
        evidence = 1.0 - math.exp(-(len(zone_tracks[z]) + inflow[z]) / 5.0)
        confidence = evidence * min(1.0, mean_score / conf_ref) if n_obs[z] else 0.0
        risk = (weights.get("density", 0) * density
                + weights.get("convergence", 0) * convergence
                + weights.get("confidence", 0) * confidence) / wsum
        if n_obs[z] == 0 and inflow[z] == 0:
            risk = 0.0
        flow = None
        zx, zy, zn = zone_vec[z]
        if zn >= 2:
            deg, cp = compass(zx, zy, heading_deg)
            mag = math.hypot(zx, zy) or 1.0
            flow = {"deg": deg, "compass": cp, "n": zn,
                    "ux": round(zx / mag, 3), "uy": round(zy / mag, 3)}
        # % change needs a meaningful baseline; with too few early observations
        # the zone is reported as "new activity" instead of e.g. "+4100%".
        change = None
        if first_half[z] >= min_half:
            change = round((second_half[z] - first_half[z]) / first_half[z] * 100, 1)
        zones.append({
            "id": zone_name(z),
            "row": r, "col": c,
            "bounds": [round(c / cols, 4), round(r / rows, 4),
                       round((c + 1) / cols, 4), round((r + 1) / rows, 4)],
            "risk": round(risk, 3),
            "priority": DecisionEngine._priority(risk, threshold) if risk > 0 else "NONE",
            "density": round(density, 3),
            "objects_per_frame": round(n_obs[z] / n_frames, 2),
            "density_change_pct": change,
            "new_activity": first_half[z] < min_half <= second_half[z],
            "convergence": round(convergence, 3),
            "convergence_level": _level(convergence),
            "inflow": inflow[z], "outflow": outflow[z],
            "confidence": round(confidence, 3),
            "evidence": round(evidence, 3),
            "mean_score": round(mean_score, 3),
            "n_tracks": len(zone_tracks[z]),
            "n_obs": n_obs[z],
            "eta_s": round(median(inflow_eta[z]), 1) if inflow_eta[z] else None,
            "flow": flow,
        })

    queue = sorted((z for z in zones if z["risk"] > 0), key=lambda z: z["risk"], reverse=True)

    # ---- Reasoning: global drift -------------------------------------------
    n_vec = len(speeds)
    motion_summary: dict = {"n_vectors": n_vec, "compensated": compensate,
                            "calibrated": mpp is not None, "enough_data": n_vec >= 5}
    if n_vec:
        mean_speed = sum(speeds) / n_vec
        net = math.hypot(vsum[0] / n_vec, vsum[1] / n_vec) * proc_fps
        deg, cp = compass(vsum[0], vsum[1], heading_deg)
        motion_summary.update({
            "speed_px_s": round(mean_speed, 1),
            "speed_m_s": round(mean_speed * mpp, 3) if mpp else None,
            "deg": deg, "compass": cp,
            "coherence": round(net / mean_speed, 2) if mean_speed > 0 else 0.0,
        })

    hotspot = None
    if queue:
        top = queue[0]
        flow = top["flow"] or ({"compass": motion_summary.get("compass"),
                                "deg": motion_summary.get("deg")} if n_vec else None)
        hotspot = {**top, "why": {
            "density_change_pct": top["density_change_pct"],
            "new_activity": top["new_activity"],
            "convergence_level": top["convergence_level"],
            "flow": flow,
            "confidence": top["confidence"],
        }}

    unique_tracks = len({b[0] for _, bs in overlay for b in bs if b[0] is not None})
    comp_total = sum(composition.values()) or 1
    return {
        "frame_size": [w, h],
        "duration_s": round(duration, 2),
        "proc_fps": round(proc_fps, 3),
        "horizon_s": horizon,
        "lookahead_s": lookahead,
        "grid": [rows, cols],
        "zones": zones,
        "queue": [{"zone": z["id"], "risk": z["risk"], "priority": z["priority"],
                   "eta_s": z["eta_s"]} for z in queue],
        "hotspot": hotspot,
        "motion": motion_summary,
        "overlay": overlay,
        "trajectories": {str(k): v for k, v in trajectories.items() if len(v["pts"]) >= 2},
        "trend": trend,
        "composition": [
            {"category": k, "count": v, "pct": round(v / comp_total * 100, 1)}
            for k, v in sorted(composition.items(), key=lambda kv: -kv[1])
        ],
        "kpis": {
            "detections": total_det,
            "unique_tracks": unique_tracks,
            "density_per_frame": round(total_det / n_frames, 2),
            "hotspots": sum(1 for z in zones if z["priority"] == "HIGH"),
            "max_risk": queue[0]["risk"] if queue else 0.0,
        },
    }
