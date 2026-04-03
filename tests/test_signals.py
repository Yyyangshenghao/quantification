from __future__ import annotations

import copy

import pandas as pd

from src.strategy.signals import SignalEngine, compute_entry_signal_score


def base_snapshot_row(bucket: str) -> dict:
    return {
        "symbol": "600000.sh",
        "name": "测试股",
        "industry": "银行" if bucket == "defensive_dividend" else "煤炭",
        "bucket": bucket,
        "close": 10.0,
        "ma20": 9.5,
        "ma60": 9.0,
        "ma120": 11.0,
        "ma200": 8.5,
        "ma250": 10.0,
        "atr20": 0.5,
        "ma20_slope_10d": 0.02,
        "ma60_slope_20d": 0.01,
        "ma120_slope_20d": 0.0,
        "stock_q_blended": 10 if bucket == "defensive_dividend" else 8,
        "industry_q_blended": 10,
        "quality_pass": True,
        "thesis_still_valid": True,
        "cycle_peak_trap": False,
        "fundamental_break": False,
        "in_effective_universe": True,
        "holding_state": "NONE",
        "current_position_tranches": 0,
        "current_weight": 0.0,
        "current_shares": 0,
        "final_score": 82,
        "universe_final_score": 82,
        "data_stale": False,
    }


def _engine_configs(configs: dict) -> tuple[dict, dict, dict]:
    strategy_cfg = copy.deepcopy(configs["strategy"])
    account_cfg = copy.deepcopy(configs["account"])
    account_cfg["account"]["current_cash"] = 40000
    account_cfg["account"]["latest_total_equity"] = 200000
    return strategy_cfg, copy.deepcopy(configs["universe_rules"]), account_cfg


def test_signal_engine_blocks_new_cyclical_positions_in_risk_off(configs: dict) -> None:
    strategy_cfg, universe_rules_cfg, account_cfg = _engine_configs(configs)
    engine = SignalEngine(strategy_cfg, universe_rules_cfg, account_cfg)
    snapshot = pd.DataFrame([base_snapshot_row("cyclical_rotation")])
    positions = pd.DataFrame(columns=["symbol", "current_position_tranches", "current_weight", "extra_tranches", "last_fill_price"])
    decisions = engine.generate(
        snapshot,
        positions,
        {"regime": "risk_off", "max_total_position": 0.4},
        account_state={"current_cash": 40000, "reserved_cash": 0, "latest_total_equity": 200000, "current_invested_value": 0},
    )
    assert decisions[0]["action_enum"] == "BLOCKED"


def test_frozen_holding_cannot_buy_additional_tranche(configs: dict) -> None:
    strategy_cfg, universe_rules_cfg, account_cfg = _engine_configs(configs)
    engine = SignalEngine(strategy_cfg, universe_rules_cfg, account_cfg)
    row = base_snapshot_row("defensive_dividend")
    row["holding_state"] = "FROZEN"
    row["current_position_tranches"] = 1
    row["current_weight"] = 0.03
    row["current_shares"] = 600
    snapshot = pd.DataFrame([row])
    decisions = engine.generate(
        snapshot,
        pd.DataFrame(
            [
                {
                    "symbol": "600000.sh",
                    "current_position_tranches": 1,
                    "current_weight": 0.03,
                    "current_shares": 600,
                    "avg_cost": 10.0,
                    "extra_tranches": 0,
                    "last_fill_price": 10.0,
                }
            ]
        ),
        {"regime": "risk_on", "max_total_position": 1.0},
        account_state={"current_cash": 40000, "reserved_cash": 0, "latest_total_equity": 200000, "current_invested_value": 6000},
    )
    assert decisions[0]["action_enum"] == "HOLD_FROZEN"


def test_allocator_prefers_existing_holding_before_new_name(configs: dict) -> None:
    strategy_cfg, universe_rules_cfg, account_cfg = _engine_configs(configs)
    account_cfg["account"]["current_cash"] = 7000
    engine = SignalEngine(strategy_cfg, universe_rules_cfg, account_cfg)
    held = base_snapshot_row("defensive_dividend")
    held["symbol"] = "600001.sh"
    held["current_position_tranches"] = 1
    held["current_weight"] = 0.03
    held["current_shares"] = 600
    held["holding_state"] = "ACTIVE"
    held["last_fill_price"] = 11.0
    held["close"] = 9.5
    held["final_score"] = 80
    fresh = base_snapshot_row("defensive_dividend")
    fresh["symbol"] = "600002.sh"
    fresh["final_score"] = 70
    snapshot = pd.DataFrame([held, fresh])
    positions = pd.DataFrame(
        [
            {
                "symbol": "600001.sh",
                "current_position_tranches": 1,
                "current_weight": 0.03,
                "current_shares": 600,
                "avg_cost": 11.0,
                "extra_tranches": 0,
                "last_fill_price": 11.0,
            }
        ]
    )
    decisions = engine.generate(
        snapshot,
        positions,
        {"regime": "risk_on", "max_total_position": 1.0},
        account_state={"current_cash": 7000, "reserved_cash": 0, "latest_total_equity": 200000, "current_invested_value": 6000},
    )
    by_symbol = {item["symbol"]: item for item in decisions}
    assert by_symbol["600001.sh"]["action_enum"] == "BUY_2"
    assert by_symbol["600002.sh"]["action_enum"] == "BLOCKED"
    assert by_symbol["600002.sh"]["blocked_reason"] == "INSUFFICIENT_CASH"


def test_safe_mode_blocks_new_buy(configs: dict) -> None:
    strategy_cfg, universe_rules_cfg, account_cfg = _engine_configs(configs)
    engine = SignalEngine(strategy_cfg, universe_rules_cfg, account_cfg)
    snapshot = pd.DataFrame([base_snapshot_row("defensive_dividend")])
    decisions = engine.generate(
        snapshot,
        pd.DataFrame(columns=["symbol", "current_position_tranches", "current_weight", "extra_tranches", "last_fill_price"]),
        {"regime": "risk_on", "max_total_position": 1.0},
        safe_mode=True,
        account_state={"current_cash": 40000, "reserved_cash": 0, "latest_total_equity": 200000, "current_invested_value": 0},
    )
    assert decisions[0]["action_enum"] == "BLOCKED"
    assert decisions[0]["blocked_reason"] == "DATA_STALE_BLOCK"


def test_entry_signal_score_is_higher_for_deeper_valuation_and_trend(configs: dict) -> None:
    row_lo = base_snapshot_row("defensive_dividend")
    row_hi = base_snapshot_row("defensive_dividend")
    row_lo["stock_q_blended"] = 18
    row_hi["stock_q_blended"] = 8
    row_hi["close"] = 9.6
    bucket_cfg = configs["strategy"]["buckets"]["defensive_dividend"]
    assert compute_entry_signal_score(row_hi, bucket_cfg, "BUY_1") > compute_entry_signal_score(row_lo, bucket_cfg, "BUY_1")


def test_round_lot_and_min_trade_value_can_block_buy(configs: dict) -> None:
    strategy_cfg = copy.deepcopy(configs["strategy"])
    universe_rules_cfg = copy.deepcopy(configs["universe_rules"])
    account_cfg = copy.deepcopy(configs["account"])
    account_cfg["account"]["current_cash"] = 20000
    account_cfg["account"]["latest_total_equity"] = 100000
    engine = SignalEngine(strategy_cfg, universe_rules_cfg, account_cfg)
    snapshot = pd.DataFrame([base_snapshot_row("defensive_dividend")])
    decisions = engine.generate(
        snapshot,
        pd.DataFrame(columns=["symbol", "current_position_tranches", "current_weight", "current_shares", "extra_tranches", "last_fill_price"]),
        {"regime": "risk_on", "max_total_position": 1.0},
        account_state={"current_cash": 20000, "reserved_cash": 0, "latest_total_equity": 100000, "current_invested_value": 0},
    )
    assert decisions[0]["action_enum"] == "BLOCKED"
    assert decisions[0]["blocked_reason"] == "MIN_TRADE_VALUE"
