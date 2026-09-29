from riverguard.decision import DecisionEngine
from riverguard.types import HotspotForecast


def test_ranking_and_priority():
    forecast = HotspotForecast(grid=[[0.9, 0.0], [0.3, 0.5]], horizon_seconds=1200)
    engine = DecisionEngine(alert_threshold=0.7)
    decisions = engine.rank(forecast)
    # zero cells dropped
    assert len(decisions) == 3
    # sorted descending
    assert decisions[0].risk_score == 0.9
    assert decisions[0].priority == "HIGH"
    assert decisions[-1].priority in {"LOW", "MEDIUM"}
