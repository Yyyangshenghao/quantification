from __future__ import annotations

import pandas as pd

from src.strategy.regime import determine_market_regime


def test_market_regime_risk_on(configs: dict) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=200).strftime("%Y-%m-%d"),
            "close": [100] * 199 + [110],
        }
    )
    result = determine_market_regime(frame, configs["strategy"])
    assert result["regime"] == "risk_on"
    assert result["max_total_position"] == 1.0


def test_market_regime_risk_off(configs: dict) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=200).strftime("%Y-%m-%d"),
            "close": [140] * 140 + [100] * 59 + [80],
        }
    )
    result = determine_market_regime(frame, configs["strategy"])
    assert result["regime"] == "risk_off"
    assert result["max_total_position"] == 0.4
