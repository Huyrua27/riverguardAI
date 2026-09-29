import io
import sys

from riverguard.utils.console import utf8_console


def test_legacy_codepage_console_does_not_crash_on_vietnamese(monkeypatch):
    raw = io.BytesIO()
    legacy = io.TextIOWrapper(raw, encoding="cp1258")   # Windows Vietnamese code page
    monkeypatch.setattr(sys, "stdout", legacy)
    utf8_console()
    print("Mở trực tiếp trong trình duyệt")                # 'ở' is not encodable in cp1258
    sys.stdout.flush()
    assert raw.getvalue().decode("utf-8").startswith("Mở trực tiếp")
