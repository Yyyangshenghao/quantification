from __future__ import annotations

import pandas as pd

from src.strategy.valuation import add_quantile_columns, trailing_quantile


def test_trailing_quantile_is_percentile() -> None:
    assert trailing_quantile([1, 2, 3, 4, 5], current=3) == 60.0


def test_add_quantile_columns_builds_blended_series() -> None:
    frame = pd.DataFrame(
        {
            "code": ["600000.sh"] * 5,
            "date": pd.date_range("2024-01-01", periods=5).strftime("%Y-%m-%d"),
            "value": [5, 4, 3, 2, 1],
        }
    )
    result = add_quantile_columns(
        frame,
        entity_col="code",
        value_col="value",
        date_col="date",
        windows={"q_5y": 5, "q_10y": 5},
        blended_weights={"q_5y": 0.6, "q_10y": 0.4},
    )
    latest = result.iloc[-1]
    assert latest["q_5y"] == 20.0
    assert latest["q_10y"] == 20.0
    assert latest["q_blended"] == 20.0
