from .report import generate_report


def build_dashboard(*args, **kwargs):
    """Lazy re-export (keeps ``python -m riverguard.dashboard.build`` warning-free)."""
    from .build import build_dashboard as _build

    return _build(*args, **kwargs)


__all__ = ["build_dashboard", "generate_report"]
