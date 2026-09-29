"""Rendering helpers — turn pipeline outputs into images a human can read.

Produces the two visual deliverables from the proposal:
- ``draw_overlay``    → live-video overlay (boxes, class, confidence, object ID)
- ``heatmap_overlay`` → Pollution Risk Map blended over the frame
"""
from __future__ import annotations

import numpy as np

# BGR colours.
_PRIORITY_COLOR = {
    "HIGH": (0, 0, 255),      # red
    "MEDIUM": (0, 165, 255),  # orange
    "LOW": (0, 200, 255),     # yellow
}
_BOX_COLOR = (60, 220, 60)    # green


def draw_overlay(frame: np.ndarray, tracks: list, motion: list | None = None) -> np.ndarray:
    """Draw tracked boxes with ID / class / confidence onto a copy of the frame."""
    import cv2

    out = frame.copy()
    motion_by_id = {m.track_id: m for m in (motion or [])}
    h = out.shape[0]
    thick = max(1, round(h / 600))
    font_scale = max(0.4, h / 1600)

    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), _BOX_COLOR, thick)
        label = f"#{t.track_id} {t.cls} {t.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), _BOX_COLOR, -1)
        cv2.putText(out, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thick, cv2.LINE_AA)
        # Draw drift arrow if we have a motion vector.
        m = motion_by_id.get(t.track_id)
        if m is not None and (abs(m.dx) + abs(m.dy)) > 0.01:
            cx, cy = (int(v) for v in t.center)
            scale = 8.0
            cv2.arrowedLine(out, (cx, cy),
                            (int(cx + m.dx * scale), int(cy + m.dy * scale)),
                            (255, 0, 0), thick, tipLength=0.3)
    return out


def heatmap_overlay(frame: np.ndarray, grid: list[list[float]], alpha: float = 0.45) -> np.ndarray:
    """Blend a normalised risk grid over the frame as a JET colour map."""
    import cv2

    h, w = frame.shape[:2]
    arr = np.asarray(grid, dtype=np.float32)
    if arr.size == 0 or float(arr.max()) <= 0.0:
        return frame.copy()
    arr = np.clip(arr, 0.0, 1.0)
    big = cv2.resize((arr * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    color = cv2.applyColorMap(big, cv2.COLORMAP_JET)
    mask = (big.astype(np.float32) / 255.0 * alpha)[..., None]
    out = (frame.astype(np.float32) * (1 - mask) + color.astype(np.float32) * mask)
    return out.astype(np.uint8)


def draw_hotspot_markers(frame: np.ndarray, decisions: list, top_k: int = 5) -> np.ndarray:
    """Mark the top-K priority regions with a labelled circle."""
    import cv2

    out = frame
    h = out.shape[0]
    font_scale = max(0.4, h / 1400)
    thick = max(1, round(h / 600))
    for d in decisions[:top_k]:
        center = d.meta.get("center_px")
        if not center:
            continue
        cx, cy = int(center[0]), int(center[1])
        color = _PRIORITY_COLOR.get(d.priority, (0, 200, 255))
        cv2.circle(out, (cx, cy), max(8, h // 60), color, thick + 1)
        cv2.putText(out, f"{d.priority} {d.risk_score:.2f}", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick, cv2.LINE_AA)
    return out
