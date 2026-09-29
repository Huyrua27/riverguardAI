"""Console encoding guard for the command-line entry points.

On Windows, when output is redirected or the console uses a legacy code page
(e.g. cp1258), printing Vietnamese text raises ``UnicodeEncodeError`` and kills
the command. Every CLI calls :func:`utf8_console` first so messages are written
as UTF-8 (unencodable characters are replaced instead of crashing).
"""
from __future__ import annotations

import sys


def utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if enc == "utf8" or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - exotic streams
            pass
