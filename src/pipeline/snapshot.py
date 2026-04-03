from __future__ import annotations

import pandas as pd

from src.strategy.fundamentals import detect_fundamental_break, force_exit, force_exit_reasons
from src.strategy.regime import determine_market_regime
from src.strategy.signals import SignalEngine
from src.strategy.universe import effective_universe_frame


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(numeric):
        return default
    return numeric


def _normalize_account_state(account_cfg: dict, positions_frame: pd.DataFrame) -> dict:
    account = account_cfg.get("account", {}) if isinstance(account_cfg, dict) else {}
    current_cash = account.get("current_cash")
    latest_total_equity = account.get("latest_total_equity")
    reserved_cash = account.get("reserved_cash", 0)
    orders_degraded = current_cash is None or latest_total_equity is None or _safe_float(latest_total_equity) <= 0
    current_invested_weight = _safe_float(positions_frame.get("current_weight", pd.Series(dtype=float)).sum(), default=0.0)
    state = {
        "mode": account_cfg.get("mode"),
        "base_currency": account_cfg.get("base_currency", "CNY"),
        "orders_degraded": orders_degraded,
        "current_cash": _safe_float(current_cash),
        "reserved_cash": _safe_float(reserved_cash),
        "latest_total_equity": _safe_float(latest_total_equity),
        "current_invested_value": 0.0,
    }
    if not orders_degraded:
        state["current_invested_value"] = round(current_invested_weight * state["latest_total_equity"], 2)
    return state


def _build_scope(feature_snapshot: pd.DataFrame, universe_cfg: dict, positions_frame: pd.DataFrame) -> pd.DataFrame:
    universe_frame = effective_universe_frame(universe_cfg).rename(
        columns={
            "final_score": "universe_final_score",
            "bucket": "universe_bucket",
            "main_metric": "universe_main_metric",
            "industry_l1": "universe_industry",
        }
    )
    universe_symbols = set(universe_frame.get("symbol", pd.Series(dtype=str)).astype(str))
    holding_symbols = set(positions_frame.get("symbol", pd.Series(dtype=str)).astype(str))
    scope_symbols = sorted(universe_symbols | holding_symbols)
    features_idx = feature_snapshot.set_index("symbol") if not feature_snapshot.empty else pd.DataFrame(index=pd.Index([], name="symbol"))
    universe_idx = universe_frame.set_index("symbol") if not universe_frame.empty else pd.DataFrame(index=pd.Index([], name="symbol"))
    positions_idx = positions_frame.set_index("symbol") if not positions_frame.empty else pd.DataFrame(index=pd.Index([], name="symbol"))
    rows: list[dict] = []
    for symbol in scope_symbols:
        row: dict = {"symbol": symbol, "missing_from_features": symbol not in features_idx.index}
        if symbol in features_idx.index:
            row.update(features_idx.loc[symbol].to_dict())
        if symbol in universe_idx.index:
            row.update(universe_idx.loc[symbol].to_dict())
            row["in_effective_universe"] = True
        else:
            row["in_effective_universe"] = False
        if symbol in positions_idx.index:
            row.update(positions_idx.loc[symbol].to_dict())
        row["bucket"] = row.get("universe_bucket", row.get("bucket"))
        row["main_metric"] = row.get("universe_main_metric", row.get("main_metric"))
        row["industry"] = row.get("industry", row.get("universe_industry"))
        row["name"] = row.get("name", symbol)
        rows.append(row)
    return pd.DataFrame(rows)


def _holding_state(row: pd.Series, freeze_removed_holdings: bool) -> str:
    current_tranches = int(row.get("current_position_tranches", 0))
    if current_tranches <= 0:
        return "NONE"
    if bool(row.get("force_exit")):
        return "FORCE_EXIT"
    if bool(row.get("in_effective_universe")):
        return "ACTIVE"
    return "FROZEN" if freeze_removed_holdings else "FORCE_EXIT"


def prepare_daily_snapshot(
    as_of_date: str,
    feature_snapshot: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    positions_frame: pd.DataFrame,
    financial_history: pd.DataFrame,
    strategy_cfg: dict,
    metric_map_cfg: dict,
    universe_cfg: dict,
    universe_rules_cfg: dict,
    account_cfg: dict,
    data_errors: list[str] | None = None,
    data_notes: list[str] | None = None,
) -> dict:
    benchmark_today = benchmark_frame[benchmark_frame["date"] <= as_of_date].copy()
    regime = determine_market_regime(benchmark_today, strategy_cfg)
    enriched = _build_scope(feature_snapshot, universe_cfg, positions_frame)
    enriched["current_position_tranches"] = (
        pd.to_numeric(enriched["current_position_tranches"], errors="coerce").fillna(0).astype(int)
        if "current_position_tranches" in enriched.columns
        else 0
    )
    enriched["current_weight"] = (
        pd.to_numeric(enriched["current_weight"], errors="coerce").fillna(0.0)
        if "current_weight" in enriched.columns
        else 0.0
    )
    enriched["fundamental_break"] = enriched.apply(
        lambda row: detect_fundamental_break(
            financial_history[financial_history["code"] == row["symbol"]],
            row["industry"],
            strategy_cfg,
            metric_map_cfg,
        )
        if pd.notna(row.get("industry"))
        else True,
        axis=1,
    )
    stale_threshold = int(universe_rules_cfg.get("prolonged_data_failure_days", 10))
    missing_from_features = enriched["missing_from_features"] if "missing_from_features" in enriched.columns else pd.Series(False, index=enriched.index)
    core_fields_complete = (
        enriched["core_fields_complete"].fillna(False)
        if "core_fields_complete" in enriched.columns
        else pd.Series(False, index=enriched.index)
    )
    core_data_stale_days = (
        pd.to_numeric(enriched["core_data_stale_days"], errors="coerce").fillna(0)
        if "core_data_stale_days" in enriched.columns
        else pd.Series(0, index=enriched.index)
    )
    enriched["data_stale"] = missing_from_features | ~core_fields_complete | (core_data_stale_days > stale_threshold)
    enriched["force_exit"] = enriched.apply(lambda row: force_exit(row, universe_rules_cfg), axis=1)
    enriched["force_exit_reasons"] = enriched.apply(lambda row: force_exit_reasons(row, universe_rules_cfg), axis=1)
    enriched["holding_state"] = enriched.apply(
        lambda row: _holding_state(row, bool(universe_rules_cfg.get("freeze_removed_holdings", True))),
        axis=1,
    )
    enriched["risk_flags"] = enriched["force_exit_reasons"].apply(list)
    enriched["reason_codes"] = enriched["force_exit_reasons"].apply(lambda values: [item.upper() for item in values])
    enriched["data_status"] = enriched["data_stale"].map(lambda flag: "stale" if flag else "ok")

    safe_mode = bool(data_errors) or bool(universe_rules_cfg.get("data_stale_blocks_new_entries", True) and enriched["data_stale"].any())
    account_state = _normalize_account_state(account_cfg, positions_frame)
    if account_state["orders_degraded"]:
        safe_note = "仅有方向性建议，未完成金额约束"
    else:
        safe_note = None
    engine = SignalEngine(strategy_cfg, universe_rules_cfg, account_cfg)
    decisions = engine.generate(enriched, positions_frame, regime, safe_mode=safe_mode, account_state=account_state)
    buy_candidates = [
        item
        for item in decisions
        if item["action_enum"] in {"BUY_1", "BUY_2", "BUY_3", "BLOCKED"} and item["current_position_tranches"] == 0
    ]
    frozen_holdings = [item for item in decisions if item["holding_state"] == "FROZEN" and item["current_position_tranches"] > 0]
    force_exit_list = [item for item in decisions if item["holding_state"] == "FORCE_EXIT"]
    universe_changes = universe_cfg.get("changes", []) if bool(universe_cfg.get("rebalance_day")) else []
    notes = list(data_notes or [])
    if safe_mode:
        notes.append("仅做风险控制，不开新仓。")
    if safe_note:
        notes.append(safe_note)
    return {
        "as_of_date": as_of_date,
        "data_status": {
            "benchmark_rows": int(len(benchmark_today)),
            "feature_rows": int(len(feature_snapshot)),
            "decision_scope_rows": int(len(enriched)),
            "degraded": bool(data_errors),
            "safe_mode": safe_mode,
            "orders_degraded": account_state["orders_degraded"],
            "notes": notes,
        },
        "portfolio_state": {
            "market_regime": regime["regime"],
            "max_total_position": regime["max_total_position"],
            "current_positions": positions_frame.to_dict(orient="records"),
            "account_state": account_state,
        },
        "summary_action": engine.summarize_actions(decisions, safe_mode=safe_mode),
        "errors": data_errors or [],
        "decisions": decisions,
        "orders": decisions,
        "top_candidates_to_buy": sorted(
            buy_candidates,
            key=lambda item: (-item["priority_score"], item["symbol"]),
        )[:10],
        "frozen_holdings": frozen_holdings,
        "force_exit_list": force_exit_list,
        "universe_change_summary": universe_changes,
        "notes": notes,
    }
