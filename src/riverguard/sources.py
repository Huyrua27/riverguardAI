"""Frame sources — feed the pipeline from an image, a folder, a video or a stream.

``iter_frames`` yields ``(frame_index, name, frame_bgr)`` so the same pipeline
runs unchanged whether the input is a single photo, a directory of stills, an
mp4, an RTSP stream or a webcam index.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS


def imread(path: str | Path) -> np.ndarray | None:
    """Unicode-safe image read.

    ``cv2.imread`` cannot open paths with non-ASCII characters on Windows
    (e.g. a Vietnamese username), so we read the bytes ourselves and decode.
    """
    import cv2

    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite(path: str | Path, img: np.ndarray) -> bool:
    """Unicode-safe image write (mirror of :func:`imread`)."""
    import cv2

    path = Path(path)
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def is_video_source(source: str) -> bool:
    """True when ``source`` is a video file, stream URL or webcam index."""
    p = Path(source)
    if p.is_dir():
        return False
    if p.is_file():
        return p.suffix.lower() in VIDEO_EXTS
    return source.isdigit() or "://" in source


def iter_frames(source: str, stride: int = 1,
                max_frames: int | None = None) -> Iterator[tuple[int, str, np.ndarray]]:
    """Yield ``(frame_index, name, frame_bgr)`` from ``source``.

    ``source`` may be: an image file, a directory of images, a video file,
    a webcam index (e.g. ``"0"``) or a stream URL.

    - ``stride``: keep every Nth frame (video only) — e.g. ``stride=15`` on a
      30 fps clip ≈ 2 fps, which is plenty for slow-drifting waste and far
      cheaper to process.
    - ``max_frames``: stop after this many *kept* frames.
    """
    import cv2

    stride = max(1, stride)
    p = Path(source)

    # Directory of images.
    if p.is_dir():
        images = sorted(f for f in p.iterdir() if _is_image(f))
        if not images:
            raise SystemExit(f"No images found in directory: {source}")
        kept = 0
        for i, img_path in enumerate(images):
            frame = imread(img_path)
            if frame is None:
                continue
            yield i, img_path.name, frame
            kept += 1
            if max_frames and kept >= max_frames:
                return
        return

    # Single image.
    if p.is_file() and _is_image(p):
        frame = imread(p)
        if frame is None:
            raise SystemExit(f"Cannot read image: {source}")
        yield 0, p.name, frame
        return

    # Video / stream / webcam index.
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {source}")
    name = p.name or str(source)
    idx = kept = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, f"{name}#{idx}", frame
                kept += 1
                if max_frames and kept >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def video_fps(source: str, default: float = 30.0) -> float:
    """Read a video's native FPS (for stride math / output timing)."""
    import cv2

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    return fps if fps and fps > 0 else default


def count_frames_hint(source: str) -> int | None:
    """Best-effort frame count (images only); ``None`` for streams/videos."""
    p = Path(source)
    if p.is_dir():
        return sum(1 for f in p.iterdir() if _is_image(f))
    if p.is_file() and _is_image(p):
        return 1
    return None
