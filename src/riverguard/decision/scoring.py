"""Risk & priority scoring — turns a forecast into actionable decisions."""
from __future__ import annotations

from riverguard.types import Decision, HotspotForecast


class DecisionEngine:
    def __init__(self, alert_threshold: float = 0.7,
                 risk_weights: dict[str, float] | None = None) -> None:
        self.alert_threshold = alert_threshold
        self.risk_weights = risk_weights or {
            "density": 0.4, "convergence": 0.4, "confidence": 0.2,
        }

    @staticmethod
    def _priority(risk: float, high: float) -> str:
        if risk >= high:
            return "HIGH"
        if risk >= high * 0.6:
            return "MEDIUM"
        return "LOW"

    def rank(self, forecast: HotspotForecast) -> list[Decision]:
        """Emit one ranked Decision per non-trivial grid cell.

        Cells with zero risk are dropped; the rest are sorted by risk descending.
        When the forecast carries a frame shape, each decision's ``meta`` also
        holds the cell's pixel-space centre so the dashboard can place a marker.
        """
        gh = len(forecast.grid)
        gw = len(forecast.grid[0]) if gh else 0
        shape = forecast.frame_shape
        decisions: list[Decision] = []
        for gy, row in enumerate(forecast.grid):
            for gx, risk in enumerate(row):
                if risk <= 0.0:
                    continue
                meta = {"grid": [gy, gx], "grid_size": [gh, gw]}
                if shape is not None and gw and gh:
                    h, w = shape
                    meta["center_px"] = [
                        round((gx + 0.5) / gw * w, 1),
                        round((gy + 0.5) / gh * h, 1),
                    ]
                decisions.append(
                    Decision(
                        region_id=f"cell_{gy}_{gx}",
                        risk_score=round(risk, 3),
                        priority=self._priority(risk, self.alert_threshold),
                        probability=round(risk, 3),
                        lead_time_seconds=forecast.horizon_seconds,
                        meta=meta,
                    )
                )
        decisions.sort(key=lambda d: d.risk_score, reverse=True)
        return decisions
