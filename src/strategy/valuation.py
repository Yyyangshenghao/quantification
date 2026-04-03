from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def trailing_quantile(history: Sequence[float], current: float | None = None) -> float:
    values = np.asarray([value for value in history if pd.notna(value)], dtype=float)
    if values.size == 0:
        return np.nan
    current_value = values[-1] if current is None else current
    return float((values <= current_value).sum() / values.size * 100.0)


def add_quantile_columns(
    frame: pd.DataFrame,
    entity_col: str,
    value_col: str,
    date_col: str,
    windows: dict[str, int],
    blended_weights: dict[str, float],
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values([entity_col, date_col]).copy()
    for label, window in windows.items():
        ordered[label] = (
            ordered.groupby(entity_col)[value_col]
            .rolling(window=window, min_periods=1)
            .apply(lambda values: trailing_quantile(values), raw=False)
            .reset_index(level=0, drop=True)
        )
    ordered["q_blended"] = 0.0
    for label, weight in blended_weights.items():
        ordered["q_blended"] += ordered[label] * float(weight)
    return ordered


def latest_quantiles(frame: pd.DataFrame, entity_col: str, date_col: str = "date") -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values([entity_col, date_col])
    latest_index = ordered.groupby(entity_col)[date_col].idxmax()
    return ordered.loc[latest_index].reset_index(drop=True)
