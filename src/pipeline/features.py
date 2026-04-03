from __future__ import annotations

import math

import numpy as np

import pandas as pd

from src.strategy.metric_map import bucket_for_industry_optional, metric_for_industry_optional
from src.strategy.quality import evaluate_quality, is_cycle_peak_trap, percentile_rank
from src.strategy.valuation import add_quantile_columns, latest_quantiles


def _compute_listed_days_from_calendar(
    ordered: pd.DataFrame,
    stock_list: pd.DataFrame | None,
    benchmark_daily: pd.DataFrame | None,
) -> pd.Series:
    if stock_list is None or benchmark_daily is None or stock_list.empty or benchmark_daily.empty:
        return pd.Series(pd.NA, index=ordered.index, dtype="Float64")
    if "listed_date" not in stock_list.columns:
        return pd.Series(pd.NA, index=ordered.index, dtype="Float64")
    listed_dates = stock_list[["code", "listed_date"]].drop_duplicates(subset=["code"]).copy()
    listed_dates["code"] = listed_dates["code"].astype(str)
    listed_dates["listed_date"] = pd.to_datetime(listed_dates["listed_date"], errors="coerce")
    merged = ordered[["code", "date"]].merge(listed_dates, how="left", on="code")
    trade_dates = pd.Index(pd.to_datetime(benchmark_daily["date"], errors="coerce").dropna().sort_values().unique())
    if trade_dates.empty:
        return pd.Series(pd.NA, index=ordered.index, dtype="Float64")
    current_positions = trade_dates.searchsorted(pd.to_datetime(merged["date"], errors="coerce"), side="right")
    listed_positions = trade_dates.searchsorted(merged["listed_date"].fillna(trade_dates[0]), side="left")
    values = current_positions - listed_positions
    values = np.where(merged["listed_date"].isna(), np.nan, np.maximum(values, 0))
    return pd.Series(values, index=ordered.index, dtype="Float64")


def compute_price_features(
    price_frame: pd.DataFrame,
    stock_list: pd.DataFrame | None = None,
    benchmark_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    ordered = price_frame.sort_values(["code", "date"]).copy()
    grouped = ordered.groupby("code", group_keys=False)
    for window in (20, 60, 120, 200, 250):
        ordered[f"ma{window}"] = grouped["close"].transform(lambda series, window=window: series.rolling(window, min_periods=1).mean())
    ordered["prev_close"] = grouped["close"].shift(1)
    ordered["tr"] = (
        pd.concat(
            [
                ordered["high"] - ordered["low"],
                (ordered["high"] - ordered["prev_close"]).abs(),
                (ordered["low"] - ordered["prev_close"]).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
    )
    ordered["atr20"] = grouped["tr"].transform(lambda series: series.rolling(20, min_periods=1).mean())
    ordered["avg_amount_60d_million"] = grouped["amount"].transform(lambda series: series.rolling(60, min_periods=1).mean()) / 1e6
    listed_days_from_calendar = _compute_listed_days_from_calendar(ordered, stock_list=stock_list, benchmark_daily=benchmark_daily)
    ordered["listed_days"] = listed_days_from_calendar.fillna(grouped.cumcount() + 1)
    ordered["ma20_slope_10d"] = grouped["ma20"].transform(lambda series: series.pct_change(10))
    ordered["ma60_slope_20d"] = grouped["ma60"].transform(lambda series: series.pct_change(20))
    ordered["ma120_slope_20d"] = grouped["ma120"].transform(lambda series: series.pct_change(20))
    return ordered.drop(columns=["prev_close", "tr"])


def _quantile_windows(strategy_cfg: dict) -> dict[str, int]:
    return {
        "q_5y": int(strategy_cfg["valuation"]["windows_years"]["q_5y"] * strategy_cfg["valuation"]["trading_days_per_year"]),
        "q_10y": int(strategy_cfg["valuation"]["windows_years"]["q_10y"] * strategy_cfg["valuation"]["trading_days_per_year"]),
    }


def _compute_metric_quantile_panel(
    frame: pd.DataFrame,
    entity_col: str,
    metric_col: str,
    value_col: str,
    date_col: str,
    strategy_cfg: dict,
    prefix: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    selected = frame[[entity_col, metric_col, date_col, value_col]].copy()
    selected[metric_col] = selected[metric_col].astype(str)
    selected["entity_metric_id"] = selected[entity_col].astype(str) + "|" + selected[metric_col]
    selected[value_col] = pd.to_numeric(selected[value_col], errors="coerce")
    selected = selected.dropna(subset=[value_col, date_col])
    weighted = add_quantile_columns(
        selected,
        entity_col="entity_metric_id",
        value_col=value_col,
        date_col=date_col,
        windows=_quantile_windows(strategy_cfg),
        blended_weights=strategy_cfg["valuation"]["blended_weights"],
    )
    weighted["history_observations"] = weighted.groupby("entity_metric_id").cumcount() + 1
    pivot = weighted.pivot_table(
        index=[entity_col, date_col],
        columns=metric_col,
        values=[value_col, "q_5y", "q_10y", "q_blended", "history_observations"],
        aggfunc="last",
    )
    pivot.columns = [
        f"{prefix}_{metric}" if value_name == value_col else f"{prefix}_{metric}_{value_name}"
        for value_name, metric in pivot.columns
    ]
    return pivot.reset_index()


def compute_stock_quantiles(valuation_frame: pd.DataFrame, strategy_cfg: dict) -> pd.DataFrame:
    panel = compute_stock_quantile_panel(valuation_frame, strategy_cfg)
    if panel.empty:
        return panel
    latest_index = panel.sort_values(["code", "date"]).groupby("code")["date"].idxmax()
    return panel.loc[latest_index].reset_index(drop=True)


def compute_stock_quantile_panel(valuation_frame: pd.DataFrame, strategy_cfg: dict) -> pd.DataFrame:
    panel = _compute_metric_quantile_panel(
        valuation_frame.rename(columns={"value": "metric_value"}),
        entity_col="code",
        metric_col="metric",
        value_col="metric_value",
        date_col="date",
        strategy_cfg=strategy_cfg,
        prefix="stock",
    )
    return panel


def compute_industry_quantiles(industry_frame: pd.DataFrame, strategy_cfg: dict, metric: str) -> pd.DataFrame:
    panel = compute_industry_quantile_panel(industry_frame, strategy_cfg, {"fallback_metric": metric})
    if panel.empty:
        return panel
    latest_index = panel.sort_values(["industry_code", "date"]).groupby("industry_code")["date"].idxmax()
    return panel.loc[latest_index].reset_index(drop=True)


def compute_industry_quantile_panel(industry_frame: pd.DataFrame, strategy_cfg: dict, metric_map_cfg: dict) -> pd.DataFrame:
    if industry_frame.empty:
        return industry_frame.copy()
    selected = industry_frame.copy()
    selected["industry_code"] = selected["industry_code"].astype(str).str.replace(".SI", "", regex=False)
    if "metric" not in selected.columns:
        metric_columns = [column for column in ("pb", "pe_ttm") if column in selected.columns]
        stacked_frames: list[pd.DataFrame] = []
        for metric in metric_columns:
            frame = selected[["industry_code", "industry_name", "date", metric]].rename(columns={metric: "metric_value"}).copy()
            frame["metric"] = metric
            stacked_frames.append(frame)
        if not stacked_frames:
            return pd.DataFrame(columns=["industry_code", "date"])
        selected = pd.concat(stacked_frames, ignore_index=True)
    else:
        selected = selected.rename(columns={"value": "metric_value"})
    panel = _compute_metric_quantile_panel(
        selected,
        entity_col="industry_code",
        metric_col="metric",
        value_col="metric_value",
        date_col="date",
        strategy_cfg=strategy_cfg,
        prefix="industry",
    )
    industry_names = industry_frame[["industry_code", "industry_name"]].drop_duplicates()
    return panel.merge(industry_names, how="left", on="industry_code")


def _annual_profit_positive_years(group: pd.DataFrame) -> pd.Series:
    annual = group[group["report_month_day"] == "12-31"][["report_date", "latest_net_profit"]].copy()
    if annual.empty:
        return pd.Series(0, index=group.index, dtype=float)
    counts: list[float] = []
    annual = annual.sort_values("report_date")
    for report_date in pd.to_datetime(group["report_date"]):
        subset = annual[pd.to_datetime(annual["report_date"]) <= report_date].tail(3)
        counts.append(float((pd.to_numeric(subset["latest_net_profit"], errors="coerce") > 0).sum()))
    return pd.Series(counts, index=group.index, dtype=float)


def prepare_financial_effective_frame(financials: pd.DataFrame, strategy_cfg: dict) -> pd.DataFrame:
    if financials.empty:
        return financials.copy()
    fallback_days = int(strategy_cfg["financial_effective_date"]["fallback_days"])
    ordered = financials.sort_values(["code", "report_date", "announcement_date"]).copy()
    ordered["report_date"] = pd.to_datetime(ordered["report_date"])
    ordered["announcement_date"] = pd.to_datetime(ordered["announcement_date"], errors="coerce")
    ordered["effective_date"] = ordered["announcement_date"].fillna(ordered["report_date"] + pd.Timedelta(days=fallback_days))
    ordered["date"] = ordered["effective_date"].dt.strftime("%Y-%m-%d")
    ordered["latest_net_profit"] = pd.to_numeric(ordered["net_profit"], errors="coerce")
    ordered["cfo_ttm"] = (
        ordered.groupby("code")["cfo"].transform(lambda series: pd.to_numeric(series, errors="coerce").rolling(4, min_periods=1).sum())
    )
    ordered["roe"] = pd.to_numeric(ordered["roe"], errors="coerce")
    ordered["debt_to_assets"] = pd.to_numeric(ordered["debt_to_assets"], errors="coerce")
    ordered["dv_ttm"] = pd.to_numeric(ordered.get("dv_ttm"), errors="coerce")
    ordered["report_month_day"] = ordered["report_date"].dt.strftime("%m-%d")
    profit_positive = ordered.groupby("code").apply(_annual_profit_positive_years)
    if isinstance(profit_positive, pd.DataFrame):
        profit_positive = profit_positive.stack()
    ordered["profit_positive_years_3y"] = profit_positive.reset_index(level=0, drop=True).reindex(ordered.index).astype(float)
    ordered["profit_positive_stability_score"] = ordered["profit_positive_years_3y"] / 3.0 * 100.0
    ordered["roe_pct_in_last_12_quarters"] = ordered.groupby("code")["roe"].transform(
        lambda series: series.rolling(12, min_periods=1).apply(
            lambda values: percentile_rank(pd.Series(values), current=float(values.iloc[-1])),
            raw=False,
        )
    )
    return ordered


def _merge_asof_by_key(left: pd.DataFrame, right: pd.DataFrame, by: str) -> pd.DataFrame:
    if right.empty:
        return left.copy()
    left_ordered = left.copy()
    right_ordered = right.copy()
    left_ordered["date"] = pd.to_datetime(left_ordered["date"]).dt.floor("D").astype("datetime64[ns]")
    right_ordered["date"] = pd.to_datetime(right_ordered["date"]).dt.floor("D").astype("datetime64[ns]")
    left_ordered = left_ordered.sort_values(["date", by]).copy()
    right_ordered = right_ordered.sort_values(["date", by]).copy()
    merged = pd.merge_asof(left_ordered, right_ordered, on="date", by=by, direction="backward")
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged


def _cross_sectional_percentiles(frame: pd.DataFrame, column: str, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    pct = numeric.rank(method="average", pct=True)
    values = pct * 100.0
    return values if higher_is_better else 100.0 - values


def _coalesce_metric_quantiles(frame: pd.DataFrame, metric: str) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return (
        pd.to_numeric(frame.get(f"stock_{metric}_q_5y"), errors="coerce"),
        pd.to_numeric(frame.get(f"stock_{metric}_q_10y"), errors="coerce"),
        pd.to_numeric(frame.get(f"stock_{metric}_q_blended"), errors="coerce"),
        pd.to_numeric(frame.get(f"stock_{metric}_history_observations"), errors="coerce"),
    )


def _industry_level_rank(level: object) -> int | None:
    if level is None or pd.isna(level):
        return None
    value = str(level).strip().lower()
    mapping = {
        "first": 1,
        "一级行业": 1,
        "一级": 1,
        "second": 2,
        "二级行业": 2,
        "二级": 2,
        "third": 3,
        "三级行业": 3,
        "三级": 3,
    }
    return mapping.get(value)


def _normalize_industry_members(industry_members: pd.DataFrame) -> pd.DataFrame:
    if industry_members.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "name",
                "industry_l1_code",
                "industry_l1",
                "industry_l2_code",
                "industry_l2",
                "industry_l3_code",
                "industry_l3",
                "industry_code",
                "industry",
            ]
        )
    members = industry_members.rename(columns={"code": "symbol", "industry_name": "industry"}).copy()
    members["symbol"] = members["symbol"].astype(str)
    members["industry_code"] = members["industry_code"].astype(str).str.replace(".SI", "", regex=False)
    if "name" not in members.columns:
        members["name"] = members["symbol"]
    members["_row_order"] = range(len(members))
    if "industry_level" in members.columns:
        members["_level_rank"] = members["industry_level"].map(_industry_level_rank)
    else:
        members["_level_rank"] = pd.NA

    rows: list[dict] = []
    for symbol, group in members.groupby("symbol", sort=False):
        ordered = group.drop_duplicates(subset=["industry_code", "industry"]).copy()
        ordered["_level_rank"] = ordered["_level_rank"].fillna(pd.Series(range(1, len(ordered) + 1), index=ordered.index))
        ordered = ordered.sort_values(["_level_rank", "_row_order"], ascending=[True, True]).reset_index(drop=True)
        entry = {
            "symbol": symbol,
            "name": str(ordered["name"].dropna().iloc[0]) if ordered["name"].notna().any() else symbol,
        }
        for idx, suffix in enumerate(("l1", "l2", "l3"), start=1):
            row = ordered[ordered["_level_rank"] == idx].head(1)
            if row.empty and len(ordered) >= idx:
                row = ordered.iloc[[idx - 1]]
            if row.empty:
                continue
            entry[f"industry_{suffix}_code"] = str(row.iloc[0]["industry_code"])
            entry[f"industry_{suffix}"] = row.iloc[0]["industry"]
        entry["industry_code"] = entry.get("industry_l1_code") or entry.get("industry_l2_code") or entry.get("industry_l3_code")
        entry["industry"] = entry.get("industry_l1") or entry.get("industry_l2") or entry.get("industry_l3")
        rows.append(entry)
    return pd.DataFrame(rows)


def build_daily_feature_panel(
    price_features: pd.DataFrame,
    financials_effective: pd.DataFrame,
    stock_quantile_panel: pd.DataFrame,
    industry_quantile_panel: pd.DataFrame,
    industry_members: pd.DataFrame,
    st_flags: pd.DataFrame,
    market_caps: pd.DataFrame,
    strategy_cfg: dict,
    metric_map_cfg: dict,
) -> pd.DataFrame:
    panel = price_features.rename(columns={"code": "symbol"}).copy()
    members = _normalize_industry_members(industry_members)
    panel = panel.merge(members, how="left", on="symbol")

    financial_pti = financials_effective.rename(columns={"code": "symbol"})[
        [
            "symbol",
            "date",
            "report_date",
            "announcement_date",
            "effective_date",
            "roe",
            "latest_net_profit",
            "cfo_ttm",
            "debt_to_assets",
            "is_st",
            "dv_ttm",
            "profit_positive_years_3y",
            "profit_positive_stability_score",
            "roe_pct_in_last_12_quarters",
        ]
    ]
    panel = _merge_asof_by_key(panel, financial_pti, by="symbol")

    stock_panel = stock_quantile_panel.rename(columns={"code": "symbol"})
    panel = _merge_asof_by_key(panel, stock_panel, by="symbol")

    industry_panel = industry_quantile_panel.drop(columns=["industry_name"], errors="ignore")
    panel = _merge_asof_by_key(panel, industry_panel, by="industry_code")

    panel = _merge_asof_by_key(panel, st_flags.rename(columns={"code": "symbol"})[["symbol", "date", "is_st"]], by="symbol")
    panel = _merge_asof_by_key(
        panel,
        market_caps.rename(columns={"code": "symbol"})[["symbol", "date", "market_cap_billion"]],
        by="symbol",
    )

    panel["name"] = panel["name"].fillna(panel["symbol"])
    if "is_st_x" in panel.columns or "is_st_y" in panel.columns:
        left_st = panel["is_st_x"] if "is_st_x" in panel.columns else pd.Series(False, index=panel.index)
        right_st = panel["is_st_y"] if "is_st_y" in panel.columns else pd.Series(False, index=panel.index)
        panel["is_st"] = left_st.fillna(False) | right_st.fillna(False)
        panel = panel.drop(columns=[column for column in ("is_st_x", "is_st_y") if column in panel.columns])
    elif "is_st" not in panel.columns:
        panel["is_st"] = False
    panel["not_st"] = ~panel["is_st"].fillna(False)
    panel["is_a_share"] = panel["symbol"].astype(str).str.endswith((".sh", ".sz"))
    panel["pb"] = pd.to_numeric(panel.get("stock_pb"), errors="coerce")
    panel["pe_ttm"] = pd.to_numeric(panel.get("stock_pe_ttm"), errors="coerce")
    panel["stock_pb_q_5y"], panel["stock_pb_q_10y"], panel["stock_pb_q_blended"], panel["stock_pb_history_observations"] = _coalesce_metric_quantiles(panel, "pb")
    panel["stock_pe_ttm_q_5y"], panel["stock_pe_ttm_q_10y"], panel["stock_pe_ttm_q_blended"], panel["stock_pe_ttm_history_observations"] = _coalesce_metric_quantiles(panel, "pe_ttm")
    panel["industry_pb_q_5y"] = pd.to_numeric(panel.get("industry_pb_q_5y"), errors="coerce")
    panel["industry_pb_q_10y"] = pd.to_numeric(panel.get("industry_pb_q_10y"), errors="coerce")
    panel["industry_pb_q_blended"] = pd.to_numeric(panel.get("industry_pb_q_blended"), errors="coerce")
    panel["industry_pe_ttm_q_5y"] = pd.to_numeric(panel.get("industry_pe_ttm_q_5y"), errors="coerce")
    panel["industry_pe_ttm_q_10y"] = pd.to_numeric(panel.get("industry_pe_ttm_q_10y"), errors="coerce")
    panel["industry_pe_ttm_q_blended"] = pd.to_numeric(panel.get("industry_pe_ttm_q_blended"), errors="coerce")
    panel["primary_metric"] = panel["industry"].map(lambda industry: metric_for_industry_optional(industry, metric_map_cfg))
    panel["main_metric"] = panel["primary_metric"]
    panel["stock_q_5y"] = panel.apply(
        lambda row: row.get(f"stock_{row['main_metric']}_q_5y") if pd.notna(row.get("main_metric")) else pd.NA,
        axis=1,
    )
    panel["stock_q_10y"] = panel.apply(
        lambda row: row.get(f"stock_{row['main_metric']}_q_10y") if pd.notna(row.get("main_metric")) else pd.NA,
        axis=1,
    )
    panel["stock_q_blended"] = panel.apply(
        lambda row: row.get(f"stock_{row['main_metric']}_q_blended") if pd.notna(row.get("main_metric")) else pd.NA,
        axis=1,
    )
    panel["industry_q_5y"] = panel.apply(
        lambda row: row.get(f"industry_{row['main_metric']}_q_5y") if pd.notna(row.get("main_metric")) else pd.NA,
        axis=1,
    )
    panel["industry_q_10y"] = panel.apply(
        lambda row: row.get(f"industry_{row['main_metric']}_q_10y") if pd.notna(row.get("main_metric")) else pd.NA,
        axis=1,
    )
    panel["industry_q_blended"] = panel.apply(
        lambda row: row.get(f"industry_{row['main_metric']}_q_blended") if pd.notna(row.get("main_metric")) else pd.NA,
        axis=1,
    )
    panel["main_metric_history_observations"] = panel.apply(
        lambda row: row.get(f"stock_{row['main_metric']}_history_observations") if pd.notna(row.get("main_metric")) else pd.NA,
        axis=1,
    )
    panel["bucket"] = panel["industry"].map(lambda industry: bucket_for_industry_optional(industry, metric_map_cfg))

    panel["market_cap_percentile"] = panel.groupby("date", group_keys=False).apply(
        lambda frame: _cross_sectional_percentiles(frame, "market_cap_billion", higher_is_better=True)
    ).reset_index(level=0, drop=True)
    panel["avg_amount_60d_percentile"] = panel.groupby("date", group_keys=False).apply(
        lambda frame: _cross_sectional_percentiles(frame, "avg_amount_60d_million", higher_is_better=True)
    ).reset_index(level=0, drop=True)
    panel["roe_percentile"] = panel.groupby("date", group_keys=False).apply(
        lambda frame: _cross_sectional_percentiles(frame, "roe", higher_is_better=True)
    ).reset_index(level=0, drop=True)
    panel["debt_to_assets_percentile"] = panel.groupby("date", group_keys=False).apply(
        lambda frame: _cross_sectional_percentiles(frame, "debt_to_assets", higher_is_better=True)
    ).reset_index(level=0, drop=True)
    panel["dv_ttm_percentile"] = panel.groupby("date", group_keys=False).apply(
        lambda frame: _cross_sectional_percentiles(frame, "dv_ttm", higher_is_better=True)
    ).reset_index(level=0, drop=True)
    panel["cfo_score"] = panel["cfo_ttm"].map(lambda value: 100.0 if pd.notna(value) and float(value) > 0 else 0.0)
    panel["industry_market_cap_rank"] = (
        panel.groupby(["date", "industry"])["market_cap_billion"].rank(method="first", ascending=False)
    )
    panel["industry_market_cap_percentile"] = (
        panel.groupby(["date", "industry"])["market_cap_billion"].rank(method="average", pct=True, ascending=False)
    )
    panel["core_data_stale_days"] = (
        pd.to_datetime(panel["date"], errors="coerce") - pd.to_datetime(panel["effective_date"], errors="coerce")
    ).dt.days
    panel["core_fields_complete"] = panel[
        [
            "industry",
            "market_cap_billion",
            "avg_amount_60d_million",
            "roe",
            "latest_net_profit",
            "cfo_ttm",
            "debt_to_assets",
            "stock_pb_q_blended",
            "industry_pb_q_blended",
            "stock_pe_ttm_q_blended",
            "industry_pe_ttm_q_blended",
        ]
    ].notna().all(axis=1)
    quality_results = panel.apply(
        lambda row: evaluate_quality(row, strategy_cfg["buckets"][row["bucket"]]["filters"]) if pd.notna(row["bucket"]) else (False, ["bucket"]),
        axis=1,
    )
    panel["quality_pass"] = quality_results.map(lambda item: item[0])
    panel["quality_fail_reasons"] = quality_results.map(lambda item: item[1])
    cyclical_industries = set(strategy_cfg["buckets"]["cyclical_rotation"]["allowed_industries"])
    cyclical_cfg = strategy_cfg["buckets"]["cyclical_rotation"]["cycle_trap_filter"]
    panel["pe_q_blended"] = panel["stock_pe_ttm_q_blended"]
    panel["cycle_peak_trap"] = panel.apply(
        lambda row: is_cycle_peak_trap(row, pd.DataFrame(), cyclical_cfg, int(strategy_cfg["valuation"]["trading_days_per_year"]))
        if row.get("industry") in cyclical_industries
        else False,
        axis=1,
    )
    panel["thesis_still_valid"] = panel["quality_pass"] & ~panel["cycle_peak_trap"]
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_feature_snapshot(
    price_features: pd.DataFrame,
    latest_financials: pd.DataFrame,
    st_flags: pd.DataFrame,
    market_caps: pd.DataFrame,
    industry_members: pd.DataFrame,
    industry_quantiles: pd.DataFrame,
    stock_quantiles: pd.DataFrame,
    strategy_cfg: dict,
    metric_map_cfg: dict,
) -> pd.DataFrame:
    latest_prices = price_features.sort_values(["code", "date"]).groupby("code", as_index=False).tail(1).copy()
    latest_financials = latest_financials.sort_values(["code", "date"]).groupby("code", as_index=False).tail(1).copy()
    st_flags = st_flags.sort_values(["code", "date"]).groupby("code", as_index=False).tail(1).copy()
    market_caps = market_caps.sort_values(["code", "date"]).groupby("code", as_index=False).tail(1).copy()
    stock_quantiles = stock_quantiles.sort_values(["code", "date"]).groupby("code", as_index=False).tail(1).copy()
    industry_quantiles = industry_quantiles.sort_values(["industry_code", "date"]).groupby("industry_code", as_index=False).tail(1).copy()
    return build_daily_feature_panel(
        price_features=latest_prices,
        financials_effective=latest_financials,
        stock_quantile_panel=stock_quantiles,
        industry_quantile_panel=industry_quantiles,
        industry_members=industry_members,
        st_flags=st_flags,
        market_caps=market_caps,
        strategy_cfg=strategy_cfg,
        metric_map_cfg=metric_map_cfg,
    )
