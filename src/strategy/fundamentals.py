from __future__ import annotations

import math

import pandas as pd

from src.strategy.metric_map import is_financial_industry


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric):
        return default
    return numeric


def detect_non_financial_fundamental_break(financials: pd.DataFrame, strategy_cfg: dict) -> bool:
    if financials.empty:
        return True
    cfg = strategy_cfg["fundamental_break"]["non_financial"]
    ordered = financials.sort_values("date").copy()
    latest = ordered.iloc[-1]
    cfo_series = ordered.get("cfo_ttm")
    if cfo_series is None:
        cfo_series = ordered.get("cfo")
    cfo_non_positive_periods = (pd.to_numeric(cfo_series, errors="coerce").tail(int(cfg["cfo_ttm_non_positive_periods"])) <= 0).sum()
    return bool(
        float(latest["latest_net_profit"]) <= float(cfg["latest_net_profit_max"])
        or float(latest["roe"]) < float(cfg["roe_min"])
        or cfo_non_positive_periods >= int(cfg["cfo_ttm_non_positive_periods"])
    )


def detect_financial_fundamental_break(financials: pd.DataFrame, strategy_cfg: dict) -> bool:
    if financials.empty:
        return True
    cfg = strategy_cfg["fundamental_break"]["financial"]
    latest = financials.sort_values("date").iloc[-1]
    return bool(
        float(latest["latest_net_profit"]) <= float(cfg["latest_net_profit_max"])
        or float(latest["roe"]) < float(cfg["roe_min"])
        or bool(latest.get("is_st", False))
    )


def detect_fundamental_break(financials: pd.DataFrame, industry: str, strategy_cfg: dict, metric_map_cfg: dict) -> bool:
    if is_financial_industry(industry, metric_map_cfg):
        return detect_financial_fundamental_break(financials, strategy_cfg)
    return detect_non_financial_fundamental_break(financials, strategy_cfg)


def force_exit_reasons(row: dict | pd.Series, universe_rules_cfg: dict) -> list[str]:
    record = dict(row)
    reasons: list[str] = []
    configured = set(universe_rules_cfg.get("force_exit_rules", []))
    if "st" in configured and bool(record.get("is_st", False)):
        reasons.append("st")
    if "fundamental_break" in configured and bool(record.get("fundamental_break", False)):
        reasons.append("fundamental_break")
    if "delist_risk" in configured and bool(record.get("delist_risk", False)):
        reasons.append("delist_risk")
    prolonged_days = int(universe_rules_cfg.get("prolonged_data_failure_days", 10))
    if "prolonged_data_failure" in configured and _as_float(record.get("core_data_stale_days"), default=0.0) > prolonged_days:
        reasons.append("prolonged_data_failure")
    suspension_days = int(universe_rules_cfg.get("long_suspension_days", 20))
    if _as_float(record.get("suspension_days"), default=0.0) >= suspension_days:
        reasons.append("long_suspension")
    return reasons


def force_exit(row: dict | pd.Series, universe_rules_cfg: dict) -> bool:
    return bool(force_exit_reasons(row, universe_rules_cfg))
