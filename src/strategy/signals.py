from __future__ import annotations

from collections import Counter
from math import ceil

import pandas as pd

from src.strategy.grid import can_add_grid_tranche, can_reduce_grid_tranche


ACTIONS = {"BUY_1", "BUY_2", "BUY_3", "HOLD", "HOLD_FROZEN", "REDUCE", "SELL_ALL", "EMPTY", "BLOCKED", "DATA_ERROR"}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(numeric):
        return default
    return numeric


def _safe_int(value: object, default: int = 0) -> int:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return default
    return numeric if numeric >= 0 else default


def _close_to_ma(row: dict, ma_field: str) -> float:
    return float(row["close"]) / float(row[ma_field]) if _safe_float(row.get(ma_field)) else float("inf")


def _risk_off_allows_open(row: dict, regime: str, strategy_cfg: dict) -> bool:
    if regime != "risk_off":
        return True
    risk_off_cfg = strategy_cfg["market_regime"]["risk_off_opening_rules"]
    if row["bucket"] not in set(risk_off_cfg["allow_new_buckets"]):
        return False
    return _safe_float(row.get("stock_q_blended"), default=100.0) <= float(risk_off_cfg["defensive_dividend_max_stock_q_blended"])


def _entry_rule(row: dict, bucket_cfg: dict, current_tranches: int, max_tranches: int) -> tuple[str | None, list[str]]:
    next_tranche = current_tranches + 1
    if next_tranche > max_tranches:
        return None, []
    level_name = f"BUY_{next_tranche}"
    rule = bucket_cfg["buy_levels"].get(level_name)
    if not rule:
        return None, []
    reason_codes: list[str] = []
    if _safe_float(row.get("stock_q_blended"), default=100.0) <= float(rule["stock_q_blended_max"]):
        reason_codes.append("LOW_VALUATION")
    if _safe_float(row.get("industry_q_blended"), default=100.0) <= float(rule["industry_q_blended_max"]):
        reason_codes.append("INDUSTRY_CHEAP")
    if row["bucket"] == "defensive_dividend":
        if _close_to_ma(row, "ma120") <= float(rule["close_to_ma120_max"]):
            reason_codes.append("MA120_DISCOUNT")
        passed = bool(
            _safe_float(row.get("stock_q_blended"), default=100.0) <= float(rule["stock_q_blended_max"])
            and _safe_float(row.get("industry_q_blended"), default=100.0) <= float(rule["industry_q_blended_max"])
            and _close_to_ma(row, "ma120") <= float(rule["close_to_ma120_max"])
            and bool(row.get("quality_pass", False))
        )
    else:
        close_gt_ma20 = not rule.get("close_gt_ma20") or _safe_float(row.get("close")) > _safe_float(row.get("ma20"))
        close_gt_ma60 = not rule.get("close_gt_ma60") or _safe_float(row.get("close")) > _safe_float(row.get("ma60"))
        ma20_slope_ok = _safe_float(row.get("ma20_slope_10d"), default=-10**9) >= float(rule.get("ma20_slope_10d_min", -10**9))
        ma60_slope_ok = _safe_float(row.get("ma60_slope_20d"), default=-10**9) >= float(rule.get("ma60_slope_20d_min", -10**9))
        if close_gt_ma20 or close_gt_ma60 or ma20_slope_ok or ma60_slope_ok:
            reason_codes.append("TREND_CONFIRMED")
        passed = bool(
            _safe_float(row.get("stock_q_blended"), default=100.0) <= float(rule["stock_q_blended_max"])
            and _safe_float(row.get("industry_q_blended"), default=100.0) <= float(rule["industry_q_blended_max"])
            and close_gt_ma20
            and close_gt_ma60
            and ma20_slope_ok
            and ma60_slope_ok
            and bool(row.get("quality_pass", False))
            and not bool(row.get("cycle_peak_trap", False))
        )
    return (level_name if passed else None), reason_codes


def compute_entry_signal_score(row: dict, bucket_cfg: dict, level_name: str) -> float:
    rule = bucket_cfg["buy_levels"][level_name]
    stock_threshold = float(rule["stock_q_blended_max"])
    industry_threshold = float(rule["industry_q_blended_max"])
    stock_depth = max(stock_threshold - _safe_float(row.get("stock_q_blended"), default=stock_threshold), 0.0) / max(stock_threshold, 1.0) * 100.0
    industry_depth = max(industry_threshold - _safe_float(row.get("industry_q_blended"), default=industry_threshold), 0.0) / max(industry_threshold, 1.0) * 100.0
    valuation_score = stock_depth * 0.65 + industry_depth * 0.35
    if row["bucket"] == "defensive_dividend":
        ma_trigger = float(rule["close_to_ma120_max"])
        trend_score = max(ma_trigger - _close_to_ma(row, "ma120"), 0.0) / max(ma_trigger, 1e-6) * 100.0
    else:
        close_over_ma20 = max((_safe_float(row.get("close")) / max(_safe_float(row.get("ma20"), default=1.0), 1e-6)) - 1.0, 0.0) * 100.0
        close_over_ma60 = max((_safe_float(row.get("close")) / max(_safe_float(row.get("ma60"), default=1.0), 1e-6)) - 1.0, 0.0) * 100.0
        slope_bonus = max(_safe_float(row.get("ma20_slope_10d")), 0.0) * 2000.0 + max(_safe_float(row.get("ma60_slope_20d")), 0.0) * 1000.0
        trend_score = min(100.0, close_over_ma20 + close_over_ma60 + slope_bonus)
    return round(min(100.0, valuation_score * 0.6 + trend_score * 0.4), 4)


def _priority_score(row: dict, entry_signal_score: float) -> float:
    universe_score = _safe_float(row.get("universe_final_score", row.get("final_score")), default=0.0)
    return round(universe_score * 0.60 + entry_signal_score * 0.40, 4)


def _action_for_target(current_tranches: int, target_tranches: int, holding_state: str) -> str:
    if holding_state == "FORCE_EXIT":
        return "SELL_ALL"
    if target_tranches == 0 and current_tranches > 0:
        return "SELL_ALL"
    if target_tranches < current_tranches:
        return "REDUCE"
    if target_tranches > current_tranches:
        return f"BUY_{target_tranches}"
    if holding_state == "FROZEN" and current_tranches > 0:
        return "HOLD_FROZEN"
    if current_tranches > 0:
        return "HOLD"
    return "EMPTY"


class SignalEngine:
    def __init__(self, strategy_cfg: dict, universe_rules_cfg: dict | None = None, account_cfg: dict | None = None) -> None:
        self.strategy_cfg = strategy_cfg
        self.universe_rules_cfg = universe_rules_cfg or {}
        self.account_cfg = account_cfg or {}
        execution_cfg = self.account_cfg.get("execution", {}) if isinstance(self.account_cfg, dict) else {}
        position_sizing = self.account_cfg.get("position_sizing", {}) if isinstance(self.account_cfg, dict) else {}
        self.default_tranches = int(position_sizing.get("max_tranches_per_stock", strategy_cfg["execution"]["default_tranches"]))
        self.tranche_weight_map = {
            int(key): float(value)
            for key, value in (position_sizing.get("tranche_weights", {}) or {}).items()
        }
        if 0 not in self.tranche_weight_map:
            self.tranche_weight_map[0] = 0.0
        if not any(key > 0 for key in self.tranche_weight_map):
            target_universe = int(
                self.universe_rules_cfg.get("target_universe_size", strategy_cfg["execution"].get("equal_weight_target_universe_size", 16))
            )
            unit_weight = 1.0 / float(target_universe) / float(self.default_tranches)
            self.tranche_weight_map = {index: round(unit_weight * index, 6) for index in range(0, self.default_tranches + 1)}
        self.max_single_stock_weight = float(position_sizing.get("max_single_stock_weight", max(self.tranche_weight_map.values(), default=0.0)))
        self.round_lot = max(1, int(execution_cfg.get("round_lot", 100)))
        self.commission_rate = float(execution_cfg.get("commission_rate", strategy_cfg["execution"].get("fee_rate", 0.0)))
        self.stamp_duty_rate_sell = float(
            execution_cfg.get("stamp_duty_rate_sell", strategy_cfg["execution"].get("stamp_tax_rate", 0.0))
        )
        self.min_trade_value = float(position_sizing.get("min_trade_value", 0.0))

    def _weight_for_tranches(self, tranches: int) -> float:
        tranches = max(0, int(tranches))
        if tranches in self.tranche_weight_map:
            return float(self.tranche_weight_map[tranches])
        return float(self.tranche_weight_map.get(max(self.tranche_weight_map), 0.0))

    def _base_decision(self, record: dict, market_regime: dict, safe_mode: bool) -> dict:
        bucket_cfg = self.strategy_cfg["buckets"].get(record["bucket"], {})
        current_tranches = int(record.get("current_position_tranches", 0))
        current_weight = _safe_float(record.get("current_weight"), default=self._weight_for_tranches(current_tranches))
        holding_state = record.get("holding_state", "NONE")
        record["current_position_tranches"] = current_tranches
        record["current_weight"] = round(current_weight, 6)
        record["current_shares"] = _safe_int(record.get("current_shares"))
        record["avg_cost"] = round(_safe_float(record.get("avg_cost")), 4)
        record["risk_flags"] = list(record.get("risk_flags", []))
        record["reason_codes"] = list(record.get("reason_codes", []))
        record["blocked_reason"] = record.get("blocked_reason")
        record["data_status"] = record.get("data_status", "ok")

        if bool(record.get("missing_from_features")):
            record["action_enum"] = "DATA_ERROR"
            record["action_reason"] = "决策范围内缺少当日特征，保留人工核查。"
            record["desired_target_tranches"] = current_tranches
            record["desired_target_weight"] = round(current_weight, 6)
            record["target_position_tranches"] = current_tranches
            record["target_weight"] = round(current_weight, 6)
            record["target_position_change"] = 0.0
            record["priority_score"] = 0.0
            record["entry_signal_score"] = 0.0
            return record

        if bool(record.get("cycle_peak_trap")):
            record["reason_codes"].append("CYCLE_TRAP")
            record["risk_flags"].append("cycle_trap")
        if bool(record.get("fundamental_break")):
            record["reason_codes"].append("FUNDAMENTAL_BREAK")
            record["risk_flags"].append("fundamental_break")
        if bool(record.get("data_stale")):
            record["reason_codes"].append("DATA_STALE_BLOCK")
            record["risk_flags"].append("data_stale")

        if holding_state == "FORCE_EXIT":
            record["desired_target_tranches"] = 0
            record["desired_target_weight"] = 0.0
            record["target_position_tranches"] = 0
            record["target_weight"] = 0.0
            record["target_position_change"] = round(-current_weight, 6)
            record["action_enum"] = "SELL_ALL"
            record["action_reason"] = "命中强制退出规则。"
            record["priority_score"] = 100.0
            record["entry_signal_score"] = 0.0
            return record

        if current_tranches > 0:
            position = {
                "current_position": current_weight,
                "remaining_tranches": max(0, self.default_tranches - current_tranches),
                "extra_tranches": int(record.get("extra_tranches", 0)),
                "last_fill_price": _safe_float(record.get("last_fill_price"), default=_safe_float(record.get("close"))),
            }
            if bool(record.get("fundamental_break")):
                target_tranches = 0
                action_reason = "基本面破坏，清仓退出。"
            elif record["bucket"] == "defensive_dividend" and (
                _safe_float(record.get("stock_q_blended")) >= float(bucket_cfg["sell_all"]["stock_q_blended_min"])
                or _safe_float(record.get("industry_q_blended")) >= float(bucket_cfg["sell_all"]["industry_q_blended_min"])
            ):
                target_tranches = 0
                action_reason = "估值过热，触发清仓。"
            elif record["bucket"] == "cyclical_rotation" and (
                _safe_float(record.get("stock_q_blended")) >= float(bucket_cfg["sell_all"]["stock_q_blended_min"])
                or _safe_float(record.get("industry_q_blended")) >= float(bucket_cfg["sell_all"]["industry_q_blended_min"])
                or (_safe_float(record.get("close")) < _safe_float(record.get("ma60")) and _safe_float(record.get("ma20_slope_10d")) < 0)
            ):
                target_tranches = 0
                action_reason = "周期趋势转弱，触发清仓。"
            elif can_reduce_grid_tranche(record, position, self.strategy_cfg["execution"]["grid_execution"]):
                target_tranches = max(0, current_tranches - 1)
                action_reason = "触发网格减仓。"
            elif record["bucket"] == "defensive_dividend" and (
                _safe_float(record.get("stock_q_blended")) >= float(bucket_cfg["reduce"]["stock_q_blended_min"])
                and _close_to_ma(record, "ma120") >= float(bucket_cfg["reduce"]["close_to_ma120_min"])
            ):
                target_tranches = max(0, current_tranches - 1)
                action_reason = "估值回升至减仓区间。"
            elif record["bucket"] == "cyclical_rotation" and (
                _safe_float(record.get("stock_q_blended")) >= float(bucket_cfg["reduce"]["stock_q_blended_min"])
                and _safe_float(record.get("close")) >= _safe_float(record.get("ma20"))
            ):
                target_tranches = max(0, current_tranches - 1)
                action_reason = "周期估值回升，减仓锁定收益。"
            elif holding_state == "FROZEN":
                target_tranches = current_tranches
                action_reason = "已冻结持仓，仅观察或风险控制。"
                record["reason_codes"].append("FROZEN_NOT_BUYABLE")
            elif safe_mode:
                target_tranches = current_tranches
                action_reason = "safe_mode 启用，仅做风险控制，不开新仓。"
            else:
                next_level, reason_codes = _entry_rule(record, bucket_cfg, current_tranches, self.default_tranches)
                record["reason_codes"].extend(reason_codes)
                if next_level and can_add_grid_tranche(record, position, self.strategy_cfg["execution"]["grid_execution"]):
                    target_tranches = min(self.default_tranches, current_tranches + 1)
                    action_reason = "满足下一笔买点并通过网格补仓条件。"
                else:
                    target_tranches = current_tranches
                    action_reason = "持仓继续观察。"
        else:
            if not bool(record.get("in_effective_universe", False)):
                target_tranches = 0
                action_reason = "不在 effective universe 中。"
            elif safe_mode:
                target_tranches = 0
                action_reason = "safe_mode 启用，仅做风险控制，不开新仓。"
                record["blocked_reason"] = "DATA_STALE_BLOCK"
                record["reason_codes"].append("DATA_STALE_BLOCK")
            elif bool(record.get("cycle_peak_trap")):
                target_tranches = 0
                action_reason = "命中周期高点陷阱过滤器。"
                record["blocked_reason"] = "CYCLE_TRAP"
            elif not _risk_off_allows_open(record, market_regime["regime"], self.strategy_cfg):
                target_tranches = 0
                action_reason = f"当前市场 {market_regime['regime']}，禁止该 bucket 新开仓。"
                record["blocked_reason"] = "REGIME_OPEN_BLOCK"
            else:
                next_level, reason_codes = _entry_rule(record, bucket_cfg, 0, self.default_tranches)
                record["reason_codes"].extend(reason_codes)
                if next_level:
                    target_tranches = 1
                    action_reason = "满足第一笔买点。"
                else:
                    target_tranches = 0
                    action_reason = "未满足开仓条件。"

        entry_signal_score = 0.0
        if target_tranches > current_tranches:
            level_name = f"BUY_{target_tranches}"
            entry_signal_score = compute_entry_signal_score(record, bucket_cfg, level_name)
        record["entry_signal_score"] = round(entry_signal_score, 4)
        record["priority_score"] = _priority_score(record, entry_signal_score) if target_tranches > current_tranches else round(
            _safe_float(record.get("universe_final_score", record.get("final_score")), default=0.0),
            4,
        )
        record["desired_target_tranches"] = target_tranches
        record["desired_target_weight"] = round(self._weight_for_tranches(target_tranches), 6)
        record["action_reason"] = action_reason
        return record

    def _merge_position_fields(self, decisions: list[dict], positions: pd.DataFrame) -> None:
        if positions.empty or not {"symbol"} <= set(positions.columns):
            return
        lookup = positions.set_index("symbol").to_dict(orient="index")
        for decision in decisions:
            position = lookup.get(decision["symbol"])
            if not position:
                continue
            for field in ("current_shares", "avg_cost", "current_position_tranches", "current_weight", "extra_tranches", "last_fill_price"):
                if field not in decision or pd.isna(decision.get(field)):
                    decision[field] = position.get(field)

    def _reset_to_current(self, decision: dict, action_reason: str | None = None) -> dict:
        decision["target_position_tranches"] = int(decision["current_position_tranches"])
        decision["target_weight"] = round(_safe_float(decision["current_weight"]), 6)
        decision["target_position_change"] = 0.0
        decision["target_shares"] = _safe_int(decision.get("current_shares"))
        decision["delta_shares"] = 0
        decision["rounded_lots"] = 0
        decision["estimated_turnover"] = 0.0
        decision["estimated_commission"] = 0.0
        decision["estimated_stamp_duty"] = 0.0
        decision["estimated_total_cash_impact"] = 0.0
        decision["target_price_reference"] = round(_safe_float(decision.get("close")), 4) if _safe_float(decision.get("close")) > 0 else None
        decision["target_order_value"] = 0.0
        if action_reason:
            decision["action_reason"] = action_reason
        return decision

    def _rounded_target_shares(self, target_value: float, price: float) -> int:
        if target_value <= 0 or price <= 0:
            return 0
        raw_shares = int(target_value // price)
        return raw_shares // self.round_lot * self.round_lot

    def _execution_plan(self, decision: dict, action_enum: str, latest_total_equity: float, orders_degraded: bool) -> tuple[dict, float]:
        decision["action_enum"] = action_enum
        price = _safe_float(decision.get("close"))
        current_shares = _safe_int(decision.get("current_shares"))
        decision["target_price_reference"] = round(price, 4) if price > 0 else None
        if orders_degraded or latest_total_equity <= 0 or price <= 0:
            decision["target_shares"] = None
            decision["delta_shares"] = None
            decision["rounded_lots"] = None
            decision["estimated_turnover"] = None
            decision["estimated_commission"] = None
            decision["estimated_stamp_duty"] = None
            decision["estimated_total_cash_impact"] = None
            decision["target_order_value"] = None
            if action_enum in {"HOLD", "HOLD_FROZEN", "EMPTY", "BLOCKED", "DATA_ERROR"}:
                decision["target_position_tranches"] = int(decision.get("target_position_tranches", decision["current_position_tranches"]))
                decision["target_weight"] = round(_safe_float(decision.get("target_weight", decision["current_weight"])), 6)
                decision["target_position_change"] = round(decision["target_weight"] - _safe_float(decision["current_weight"]), 6)
            return decision, 0.0

        if current_shares <= 0 and int(decision["current_position_tranches"]) > 0 and action_enum in {"BUY_2", "BUY_3", "REDUCE", "SELL_ALL", "HOLD", "HOLD_FROZEN"}:
            decision["blocked_reason"] = "MISSING_SHARE_COUNT"
            decision["reason_codes"].append("MISSING_SHARE_COUNT")
            decision["action_enum"] = "BLOCKED" if action_enum != "DATA_ERROR" else "DATA_ERROR"
            self._reset_to_current(decision, "缺少当前持仓股数，无法输出可执行订单。")
            return decision, 0.0

        desired_target_weight = round(_safe_float(decision.get("desired_target_weight", decision.get("target_weight"))), 6)
        desired_target_tranches = int(decision.get("desired_target_tranches", decision["current_position_tranches"]))
        target_shares = current_shares
        if action_enum == "SELL_ALL":
            target_shares = 0
        elif action_enum == "REDUCE":
            target_shares = min(current_shares, self._rounded_target_shares(desired_target_weight * latest_total_equity, price))
        elif action_enum in {"BUY_1", "BUY_2", "BUY_3"}:
            target_shares = max(current_shares, self._rounded_target_shares(desired_target_weight * latest_total_equity, price))
        elif action_enum in {"HOLD", "HOLD_FROZEN", "EMPTY", "BLOCKED", "DATA_ERROR"}:
            target_shares = current_shares

        delta_shares = target_shares - current_shares
        estimated_turnover = round(abs(delta_shares) * price, 2)
        estimated_commission = round(estimated_turnover * self.commission_rate, 2)
        estimated_stamp_duty = round(estimated_turnover * self.stamp_duty_rate_sell, 2) if delta_shares < 0 else 0.0
        estimated_total_cash_impact = round(
            -(estimated_turnover + estimated_commission) if delta_shares > 0 else estimated_turnover - estimated_commission - estimated_stamp_duty,
            2,
        )
        rounded_lots = ceil(abs(delta_shares) / self.round_lot) if delta_shares else 0

        if action_enum in {"BUY_1", "BUY_2", "BUY_3", "REDUCE"} and delta_shares == 0:
            decision["blocked_reason"] = "ROUND_LOT_BLOCK"
            decision["reason_codes"].append("ROUND_LOT_BLOCK")
            decision["action_enum"] = "BLOCKED"
            self._reset_to_current(decision, "目标变动低于整手约束，无法下达可执行订单。")
            return decision, 0.0

        if action_enum in {"BUY_1", "BUY_2", "BUY_3", "REDUCE"} and estimated_turnover > 0 and estimated_turnover < self.min_trade_value:
            decision["blocked_reason"] = "MIN_TRADE_VALUE"
            decision["reason_codes"].append("MIN_TRADE_VALUE")
            decision["action_enum"] = "BLOCKED"
            self._reset_to_current(decision, "目标成交额低于最小交易额，订单被阻断。")
            return decision, 0.0

        actual_target_weight = round((target_shares * price) / latest_total_equity, 6)
        decision["target_position_tranches"] = desired_target_tranches if action_enum in {"BUY_1", "BUY_2", "BUY_3", "REDUCE"} else int(
            decision.get("target_position_tranches", decision["current_position_tranches"])
        )
        decision["target_weight"] = actual_target_weight
        decision["target_position_change"] = round(actual_target_weight - _safe_float(decision["current_weight"]), 6)
        decision["target_shares"] = target_shares
        decision["delta_shares"] = delta_shares
        decision["rounded_lots"] = rounded_lots
        decision["estimated_turnover"] = estimated_turnover
        decision["estimated_commission"] = estimated_commission
        decision["estimated_stamp_duty"] = estimated_stamp_duty
        decision["estimated_total_cash_impact"] = estimated_total_cash_impact
        decision["target_order_value"] = estimated_turnover
        return decision, max(0.0, -estimated_total_cash_impact)

    def generate(
        self,
        snapshot: pd.DataFrame,
        positions: pd.DataFrame,
        market_regime: dict,
        safe_mode: bool = False,
        account_state: dict | None = None,
    ) -> list[dict]:
        if snapshot.empty:
            return []
        account_state = account_state or {}
        orders_degraded = bool(account_state.get("orders_degraded", False))
        latest_total_equity = float(account_state.get("latest_total_equity", 0.0))
        current_cash = float(account_state.get("current_cash", 0.0))
        reserved_cash = float(account_state.get("reserved_cash", 0.0))
        current_invested_value = float(account_state.get("current_invested_value", 0.0))

        decisions = [self._base_decision(record.copy(), market_regime, safe_mode) for record in snapshot.to_dict(orient="records")]
        self._merge_position_fields(decisions, positions)
        fixed_weight = 0.0
        released_cash = 0.0
        buy_requests: list[dict] = []
        for decision in decisions:
            current_tranches = int(decision["current_position_tranches"])
            target_tranches = int(decision["desired_target_tranches"])
            if target_tranches > current_tranches:
                buy_requests.append(decision)
                continue

            decision["target_position_tranches"] = target_tranches
            decision["target_weight"] = round(self._weight_for_tranches(target_tranches), 6)
            decision["target_position_change"] = round(decision["target_weight"] - decision["current_weight"], 6)
            action_enum = (
                "BLOCKED"
                if current_tranches == 0 and decision.get("blocked_reason")
                else _action_for_target(current_tranches, target_tranches, decision.get("holding_state", "NONE"))
            )
            decision, _ = self._execution_plan(decision, action_enum, latest_total_equity, orders_degraded)
            fixed_weight += decision["target_weight"]
            if not orders_degraded and decision.get("estimated_total_cash_impact") is not None:
                released_cash += max(0.0, float(decision["estimated_total_cash_impact"]))

        capacity = max(0.0, float(market_regime["max_total_position"]) - fixed_weight)
        initial_buying_power = None
        available_buying_power = None
        if not orders_degraded and latest_total_equity > 0:
            gross_room_value = max(
                0.0,
                float(market_regime["max_total_position"]) * latest_total_equity - max(0.0, current_invested_value - released_cash),
            )
            cash_room_value = max(0.0, current_cash - reserved_cash + released_cash)
            initial_buying_power = round(min(cash_room_value, gross_room_value), 2)
            available_buying_power = initial_buying_power

        ordered_requests = sorted(
            buy_requests,
            key=lambda item: (
                0 if int(item["current_position_tranches"]) > 0 else 1,
                -float(item["priority_score"]),
                item["symbol"],
            ),
        )
        for decision in ordered_requests:
            current_tranches = int(decision["current_position_tranches"])
            target_tranches = int(decision["desired_target_tranches"])
            decision["target_position_tranches"] = target_tranches
            decision["target_weight"] = round(self._weight_for_tranches(target_tranches), 6)
            decision["target_position_change"] = round(decision["target_weight"] - decision["current_weight"], 6)
            decision, required_cash = self._execution_plan(
                decision,
                _action_for_target(current_tranches, target_tranches, decision.get("holding_state", "NONE")),
                latest_total_equity,
                orders_degraded,
            )
            requested_weight = max(0.0, decision["target_weight"] - _safe_float(decision["current_weight"]))
            cash_blocked = available_buying_power is not None and required_cash > available_buying_power + 1e-9
            if (
                decision["action_enum"] in {"BUY_1", "BUY_2", "BUY_3"}
                and requested_weight <= capacity + 1e-9
                and not decision.get("blocked_reason")
                and not cash_blocked
            ):
                capacity = round(max(0.0, capacity - requested_weight), 6)
                if available_buying_power is not None:
                    available_buying_power = round(max(0.0, available_buying_power - required_cash), 2)
                continue

            if decision["action_enum"] in {"BUY_1", "BUY_2", "BUY_3"}:
                if cash_blocked:
                    decision["blocked_reason"] = "INSUFFICIENT_CASH"
                    decision["reason_codes"].append("INSUFFICIENT_CASH")
                elif not decision.get("blocked_reason"):
                    decision["blocked_reason"] = "REGIME_CAP_BLOCK"
                    decision["reason_codes"].append("REGIME_CAP_BLOCK")
                decision["action_enum"] = "BLOCKED"
                self._reset_to_current(decision, "买入请求因仓位上限、资金约束或执行约束被阻断。")

        remaining_buying_power = initial_buying_power if initial_buying_power is not None else None
        if available_buying_power is not None:
            remaining_buying_power = available_buying_power
        for decision in decisions:
            decision["action_enum"] = decision.get("action_enum", "EMPTY")
            if decision["action_enum"] not in ACTIONS:
                decision["action_enum"] = "DATA_ERROR"
            decision["regime"] = market_regime["regime"]
            decision["reason_codes"] = sorted(set(decision.get("reason_codes", [])))
            decision["risk_flags"] = sorted(set(decision.get("risk_flags", [])))
            decision["current_position_tranches"] = int(decision["current_position_tranches"])
            decision["target_position_tranches"] = int(decision.get("target_position_tranches", decision["current_position_tranches"]))
            decision["current_shares"] = _safe_int(decision.get("current_shares"))
            decision["target_weight"] = round(_safe_float(decision.get("target_weight")), 6)
            decision["current_weight"] = round(_safe_float(decision.get("current_weight")), 6)
            decision["target_position_change"] = round(_safe_float(decision.get("target_position_change")), 6)
            if remaining_buying_power is None:
                decision["target_order_value"] = None
                decision["target_shares"] = decision.get("target_shares")
                decision["delta_shares"] = decision.get("delta_shares")
                decision["rounded_lots"] = decision.get("rounded_lots")
                decision["estimated_turnover"] = decision.get("estimated_turnover")
                decision["estimated_commission"] = decision.get("estimated_commission")
                decision["estimated_stamp_duty"] = decision.get("estimated_stamp_duty")
                decision["estimated_total_cash_impact"] = decision.get("estimated_total_cash_impact")
                decision["orders_degraded"] = True
                decision["latest_total_equity"] = None
                decision["current_cash"] = None
                decision["available_buying_power"] = None
            else:
                decision["orders_degraded"] = False
                decision["latest_total_equity"] = latest_total_equity
                decision["current_cash"] = current_cash
                decision["available_buying_power"] = remaining_buying_power
        return sorted(decisions, key=lambda item: (item.get("holding_state") == "NONE", item["symbol"]))

    @staticmethod
    def summarize_actions(decisions: list[dict], safe_mode: bool = False) -> str:
        if safe_mode:
            return "safe_mode"
        counter = Counter(item["action_enum"] for item in decisions)
        if not counter:
            return "无决策"
        return " / ".join(f"{action}:{count}" for action, count in sorted(counter.items()))
