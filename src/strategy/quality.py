from __future__ import annotations

import math

import pandas as pd


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric):
        return default
    return numeric


def percentile_rank(series: pd.Series, current: float | None = None) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    if current is None:
        current = float(values.iloc[-1])
    return float((values <= float(current)).sum() / len(values) * 100.0)


def evaluate_quality(row: dict | pd.Series, filters_cfg: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    row = dict(row)
    if float(row.get("listed_days", 0)) < float(filters_cfg.get("listed_days_min", 0)):
        reasons.append("listed_days")
    if bool(filters_cfg.get("not_st_required")) and not bool(row.get("not_st", False)):
        reasons.append("not_st")
    if float(row.get("market_cap_billion", 0)) < float(filters_cfg.get("market_cap_billion_min", 0)):
        reasons.append("market_cap")
    if float(row.get("avg_amount_60d_million", 0)) < float(filters_cfg.get("avg_amount_60d_million_min", 0)):
        reasons.append("avg_amount_60d_million")
    if float(row.get("roe", 0)) < float(filters_cfg.get("roe_min", 0)):
        reasons.append("roe")
    if float(row.get("latest_net_profit", 0)) <= float(filters_cfg.get("latest_net_profit_min_exclusive", -10**18)):
        reasons.append("latest_net_profit")
    if "cfo_ttm_min_exclusive" in filters_cfg and float(row.get("cfo_ttm", 0)) <= float(
        filters_cfg["cfo_ttm_min_exclusive"]
    ):
        reasons.append("cfo_ttm")
    if "debt_to_assets_max" in filters_cfg and float(row.get("debt_to_assets", 10**9)) > float(
        filters_cfg["debt_to_assets_max"]
    ):
        reasons.append("debt_to_assets")
    if "dv_ttm_min" in filters_cfg and float(row.get("dv_ttm", 0)) < float(filters_cfg["dv_ttm_min"]):
        reasons.append("dv_ttm")
    if "pb_min_exclusive" in filters_cfg and float(row.get("pb", 0)) <= float(filters_cfg["pb_min_exclusive"]):
        reasons.append("pb")
    if "pe_ttm_max_when_primary_metric" in filters_cfg and row.get("main_metric") == "pe_ttm":
        pe_ttm = _as_float(row.get("pe_ttm"), default=-1.0)
        if pe_ttm <= 0 or pe_ttm > float(filters_cfg["pe_ttm_max_when_primary_metric"]):
            reasons.append("pe_ttm")
    if "pb_q_blended_max_when_primary_metric" in filters_cfg and row.get("main_metric") == "pb":
        if _as_float(row.get("pb"), default=0.0) <= 0 or _as_float(row.get("stock_pb_q_blended"), default=101.0) > float(
            filters_cfg["pb_q_blended_max_when_primary_metric"]
        ):
            reasons.append("pb_q_blended")
    return (len(reasons) == 0, reasons)


def roe_percentile_in_last_12_quarters(latest_row: dict | pd.Series, history_frame: pd.DataFrame, lookback_quarters: int = 12) -> float:
    latest = dict(latest_row)
    if history_frame.empty:
        return 0.0
    ordered = history_frame.sort_values("date").tail(int(lookback_quarters))
    if ordered.empty:
        return 0.0
    latest_roe = latest.get("roe")
    if latest_roe is None or pd.isna(latest_roe):
        if "roe" not in ordered.columns or ordered["roe"].dropna().empty:
            return 0.0
        latest_roe = ordered["roe"].dropna().iloc[-1]
    return percentile_rank(ordered["roe"], current=float(latest_roe))


def is_cycle_peak_trap(latest_row: dict | pd.Series, history_frame: pd.DataFrame, cycle_cfg: dict, trading_days: int) -> bool:
    del trading_days
    latest = dict(latest_row)
    pe_q_blended = latest.get("pe_q_blended")
    if pe_q_blended is None or pd.isna(pe_q_blended):
        pe_q_blended = latest.get("stock_pe_ttm_q_blended", latest.get("pe_ttm_quantile_3y"))
    if pe_q_blended is None or pd.isna(pe_q_blended):
        return False
    roe_pct = latest.get("roe_pct_in_last_12_quarters")
    if roe_pct is None or pd.isna(roe_pct):
        roe_pct = roe_percentile_in_last_12_quarters(latest, history_frame, int(cycle_cfg.get("lookback_quarters", 12)))
    primary_trigger = bool(
        float(pe_q_blended) <= float(cycle_cfg["pe_q_blended_max_primary"])
        and float(roe_pct) >= float(cycle_cfg["roe_pct_in_last_12_quarters_min_primary"])
    )
    secondary_trigger = bool(
        float(pe_q_blended) <= float(cycle_cfg["pe_q_blended_max_secondary"])
        and float(roe_pct) >= float(cycle_cfg["roe_pct_in_last_12_quarters_min_secondary"])
    )
    return primary_trigger or secondary_trigger
