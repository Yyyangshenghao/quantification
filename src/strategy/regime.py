from __future__ import annotations

import pandas as pd


def determine_market_regime(index_frame: pd.DataFrame, strategy_cfg: dict) -> dict:
    ordered = index_frame.sort_values("date").copy()
    ma60_window = int(strategy_cfg["market_regime"]["moving_averages"]["ma60"])
    ma200_window = int(strategy_cfg["market_regime"]["moving_averages"]["ma200"])
    if "ma60" not in ordered.columns:
        ordered["ma60"] = ordered["close"].rolling(ma60_window, min_periods=1).mean()
    if "ma200" not in ordered.columns:
        ordered["ma200"] = ordered["close"].rolling(ma200_window, min_periods=1).mean()
    latest = ordered.iloc[-1]
    close = float(latest["close"])
    ma60 = float(latest["ma60"])
    ma200 = float(latest["ma200"])
    if close >= ma200:
        regime = "risk_on"
    elif close < ma60 and ma60 < ma200:
        regime = "risk_off"
    else:
        regime = "neutral"
    return {
        "regime": regime,
        "close": close,
        "ma60": ma60,
        "ma200": ma200,
        "max_total_position": strategy_cfg["market_regime"]["max_total_position"][regime],
    }
