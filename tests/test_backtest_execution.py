from __future__ import annotations

import copy
import json

import pandas as pd

from src.strategy.backtest_engine import BacktestEngine


def test_backtest_executes_at_next_day_open(configs: dict) -> None:
    strategy_cfg = copy.deepcopy(configs["strategy"])
    universe_rules_cfg = copy.deepcopy(configs["universe_rules"])
    account_cfg = copy.deepcopy(configs["account"])
    account_cfg["account"]["initial_capital"] = 100_000
    account_cfg["account"]["current_cash"] = 100_000
    account_cfg["account"]["latest_total_equity"] = 100_000
    account_cfg["execution"]["commission_rate"] = 0.0
    account_cfg["execution"]["stamp_duty_rate_sell"] = 0.0
    account_cfg["execution"]["slippage_bps"] = 0
    account_cfg["execution"]["round_lot"] = 1
    account_cfg["position_sizing"]["min_trade_value"] = 0
    strategy_cfg["execution"]["fee_rate"] = 0.0
    strategy_cfg["execution"]["stamp_tax_rate"] = 0.0
    strategy_cfg["execution"]["slippage_rate"] = 0.0
    engine = BacktestEngine(strategy_cfg, universe_rules_cfg=universe_rules_cfg, account_cfg=account_cfg)
    features = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "600000.sh",
                "name": "测试银行",
                "industry": "银行",
                "bucket": "defensive_dividend",
                "open": 10.0,
                "close": 10.0,
                "ma20": 10.0,
                "ma60": 9.5,
                "ma120": 11.0,
                "ma200": 9.0,
                "ma250": 10.0,
                "atr20": 0.5,
                "ma20_slope_10d": 0.01,
                "ma60_slope_20d": 0.01,
                "ma120_slope_20d": 0.0,
                "stock_q_blended": 10.0,
                "industry_q_blended": 10.0,
                "quality_pass": True,
                "cycle_peak_trap": False,
                "fundamental_break": False,
                "in_effective_universe": True,
                "holding_state": "NONE",
                "current_position_tranches": 0,
                "current_weight": 0.0,
                "final_score": 82.0,
                "universe_final_score": 82.0,
                "data_stale": False,
            },
            {
                "date": "2024-01-03",
                "symbol": "600000.sh",
                "name": "测试银行",
                "industry": "银行",
                "bucket": "defensive_dividend",
                "open": 11.0,
                "close": 11.5,
                "ma20": 10.2,
                "ma60": 9.6,
                "ma120": 11.1,
                "ma200": 9.1,
                "ma250": 10.0,
                "atr20": 0.5,
                "ma20_slope_10d": 0.01,
                "ma60_slope_20d": 0.01,
                "ma120_slope_20d": 0.0,
                "stock_q_blended": 12.0,
                "industry_q_blended": 12.0,
                "quality_pass": True,
                "cycle_peak_trap": False,
                "fundamental_break": False,
                "in_effective_universe": True,
                "holding_state": "NONE",
                "current_position_tranches": 0,
                "current_weight": 0.0,
                "final_score": 82.0,
                "universe_final_score": 82.0,
                "data_stale": False,
            },
        ]
    )
    benchmark = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "close": [100, 102]})
    result = engine.run(features=features, benchmark=benchmark, bucket="combined")
    trade = result.trade_list.iloc[0]
    assert trade["action"] == "BUY_1"
    assert trade["fill_date"] == "2024-01-03"
    assert trade["fill_price"] == 11.0


def test_backtest_falls_back_when_historical_effective_universe_is_missing(configs: dict) -> None:
    strategy_cfg = copy.deepcopy(configs["strategy"])
    universe_rules_cfg = copy.deepcopy(configs["universe_rules"])
    account_cfg = copy.deepcopy(configs["account"])
    account_cfg["account"]["initial_capital"] = 100_000
    account_cfg["account"]["current_cash"] = 100_000
    account_cfg["account"]["latest_total_equity"] = 100_000
    account_cfg["execution"]["commission_rate"] = 0.0
    account_cfg["execution"]["stamp_duty_rate_sell"] = 0.0
    account_cfg["execution"]["slippage_bps"] = 0
    account_cfg["execution"]["round_lot"] = 1
    account_cfg["position_sizing"]["min_trade_value"] = 0
    strategy_cfg["execution"]["fee_rate"] = 0.0
    strategy_cfg["execution"]["stamp_tax_rate"] = 0.0
    strategy_cfg["execution"]["slippage_rate"] = 0.0
    engine = BacktestEngine(strategy_cfg, universe_rules_cfg=universe_rules_cfg, account_cfg=account_cfg)
    features = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "600000.sh",
                "name": "测试银行",
                "industry": "银行",
                "bucket": "defensive_dividend",
                "open": 10.0,
                "close": 10.0,
                "ma20": 10.0,
                "ma60": 9.5,
                "ma120": 11.0,
                "ma200": 9.0,
                "ma250": 10.0,
                "atr20": 0.5,
                "ma20_slope_10d": 0.01,
                "ma60_slope_20d": 0.01,
                "ma120_slope_20d": 0.0,
                "stock_q_blended": 10.0,
                "industry_q_blended": 10.0,
                "final_score": 80.0,
            },
            {
                "date": "2024-01-03",
                "symbol": "600000.sh",
                "name": "测试银行",
                "industry": "银行",
                "bucket": "defensive_dividend",
                "open": 11.0,
                "close": 11.0,
                "ma20": 10.2,
                "ma60": 9.6,
                "ma120": 11.1,
                "ma200": 9.1,
                "ma250": 10.0,
                "atr20": 0.5,
                "ma20_slope_10d": 0.01,
                "ma60_slope_20d": 0.01,
                "ma120_slope_20d": 0.0,
                "stock_q_blended": 12.0,
                "industry_q_blended": 12.0,
                "final_score": 80.0,
            },
        ]
    )
    benchmark = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "close": [100, 102]})
    result = engine.run(features=features, benchmark=benchmark, bucket="combined")
    assert len(result.trade_list) == 1
    assert result.trade_list.iloc[0]["action"] == "BUY_1"
    assert result.approximate_backtest is True


def test_backtest_carries_position_state_between_days(configs: dict) -> None:
    strategy_cfg = copy.deepcopy(configs["strategy"])
    universe_rules_cfg = copy.deepcopy(configs["universe_rules"])
    account_cfg = copy.deepcopy(configs["account"])
    account_cfg["account"]["initial_capital"] = 100_000
    account_cfg["account"]["current_cash"] = 100_000
    account_cfg["account"]["latest_total_equity"] = 100_000
    account_cfg["execution"]["commission_rate"] = 0.0
    account_cfg["execution"]["stamp_duty_rate_sell"] = 0.0
    account_cfg["execution"]["slippage_bps"] = 0
    account_cfg["execution"]["round_lot"] = 1
    account_cfg["position_sizing"]["min_trade_value"] = 0
    strategy_cfg["execution"]["fee_rate"] = 0.0
    strategy_cfg["execution"]["stamp_tax_rate"] = 0.0
    strategy_cfg["execution"]["slippage_rate"] = 0.0
    engine = BacktestEngine(strategy_cfg, universe_rules_cfg=universe_rules_cfg, account_cfg=account_cfg)
    features = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "600309.sh",
                "name": "测试化工",
                "industry": "基础化工",
                "bucket": "cyclical_rotation",
                "open": 100.0,
                "close": 100.0,
                "ma20": 95.0,
                "ma60": 94.0,
                "ma120": 90.0,
                "ma200": 88.0,
                "ma250": 96.0,
                "atr20": 2.0,
                "ma20_slope_10d": 0.01,
                "ma60_slope_20d": 0.01,
                "ma120_slope_20d": 0.01,
                "stock_q_blended": 10.0,
                "industry_q_blended": 10.0,
                "quality_pass": True,
                "thesis_still_valid": True,
                "cycle_peak_trap": False,
                "fundamental_break": False,
                "in_effective_universe": True,
                "holding_state": "NONE",
                "current_position_tranches": 0,
                "current_weight": 0.0,
                "final_score": 85.0,
                "universe_final_score": 85.0,
                "data_stale": False,
            },
            {
                "date": "2024-01-03",
                "symbol": "600309.sh",
                "name": "测试化工",
                "industry": "基础化工",
                "bucket": "cyclical_rotation",
                "open": 98.0,
                "close": 94.0,
                "ma20": 95.0,
                "ma60": 93.0,
                "ma120": 90.0,
                "ma200": 88.0,
                "ma250": 96.0,
                "atr20": 2.0,
                "ma20_slope_10d": 0.01,
                "ma60_slope_20d": 0.01,
                "ma120_slope_20d": 0.01,
                "stock_q_blended": 8.0,
                "industry_q_blended": 8.0,
                "quality_pass": True,
                "thesis_still_valid": True,
                "cycle_peak_trap": False,
                "fundamental_break": False,
                "in_effective_universe": True,
                "holding_state": "NONE",
                "current_position_tranches": 0,
                "current_weight": 0.0,
                "final_score": 85.0,
                "universe_final_score": 85.0,
                "data_stale": False,
            },
            {
                "date": "2024-01-04",
                "symbol": "600309.sh",
                "name": "测试化工",
                "industry": "基础化工",
                "bucket": "cyclical_rotation",
                "open": 92.0,
                "close": 92.0,
                "ma20": 94.0,
                "ma60": 92.0,
                "ma120": 90.0,
                "ma200": 88.0,
                "ma250": 95.0,
                "atr20": 2.0,
                "ma20_slope_10d": 0.01,
                "ma60_slope_20d": 0.01,
                "ma120_slope_20d": 0.01,
                "stock_q_blended": 5.0,
                "industry_q_blended": 5.0,
                "quality_pass": True,
                "thesis_still_valid": True,
                "cycle_peak_trap": False,
                "fundamental_break": False,
                "in_effective_universe": True,
                "holding_state": "NONE",
                "current_position_tranches": 0,
                "current_weight": 0.0,
                "final_score": 85.0,
                "universe_final_score": 85.0,
                "data_stale": False,
            },
        ]
    )
    benchmark = pd.DataFrame({"date": ["2024-01-02", "2024-01-03", "2024-01-04"], "close": [100, 101, 102]})
    result = engine.run(features=features, benchmark=benchmark, bucket="cyclical_rotation")
    assert result.trade_list["action"].tolist() == ["BUY_1", "BUY_2"]


def test_backtest_replays_historical_effective_universe(configs: dict, tmp_path) -> None:
    strategy_cfg = copy.deepcopy(configs["strategy"])
    universe_rules_cfg = copy.deepcopy(configs["universe_rules"])
    account_cfg = copy.deepcopy(configs["account"])
    account_cfg["account"]["initial_capital"] = 100_000
    account_cfg["account"]["current_cash"] = 100_000
    account_cfg["account"]["latest_total_equity"] = 100_000
    account_cfg["execution"]["commission_rate"] = 0.0
    account_cfg["execution"]["stamp_duty_rate_sell"] = 0.0
    account_cfg["execution"]["slippage_bps"] = 0
    account_cfg["execution"]["round_lot"] = 1
    account_cfg["position_sizing"]["min_trade_value"] = 0
    history_dir = tmp_path / "universe_history"
    history_dir.mkdir()
    payload = {
        "effective_from": "2024-01-02",
        "effective_to": "2024-01-31",
        "stocks": [{"symbol": "600000.sh"}],
    }
    (history_dir / "2024-01-02.json").write_text(json.dumps(payload), encoding="utf-8")
    engine = BacktestEngine(
        strategy_cfg,
        universe_rules_cfg=universe_rules_cfg,
        account_cfg=account_cfg,
        historical_universe_dir=history_dir,
    )
    features = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "symbol": "600000.sh",
                "name": "测试银行",
                "industry": "银行",
                "bucket": "defensive_dividend",
                "open": 10.0,
                "close": 10.0,
                "ma20": 10.0,
                "ma60": 9.5,
                "ma120": 11.0,
                "ma200": 9.0,
                "ma250": 10.0,
                "atr20": 0.5,
                "ma20_slope_10d": 0.01,
                "ma60_slope_20d": 0.01,
                "ma120_slope_20d": 0.0,
                "stock_q_blended": 10.0,
                "industry_q_blended": 10.0,
                "quality_pass": True,
                "cycle_peak_trap": False,
                "fundamental_break": False,
                "final_score": 82.0,
            },
            {
                "date": "2024-01-02",
                "symbol": "601398.sh",
                "name": "测试二号",
                "industry": "银行",
                "bucket": "defensive_dividend",
                "open": 10.0,
                "close": 10.0,
                "ma20": 10.0,
                "ma60": 9.5,
                "ma120": 11.0,
                "ma200": 9.0,
                "ma250": 10.0,
                "atr20": 0.5,
                "ma20_slope_10d": 0.01,
                "ma60_slope_20d": 0.01,
                "ma120_slope_20d": 0.0,
                "stock_q_blended": 10.0,
                "industry_q_blended": 10.0,
                "quality_pass": True,
                "cycle_peak_trap": False,
                "fundamental_break": False,
                "final_score": 82.0,
            },
            {
                "date": "2024-01-03",
                "symbol": "600000.sh",
                "name": "测试银行",
                "industry": "银行",
                "bucket": "defensive_dividend",
                "open": 11.0,
                "close": 11.0,
                "ma20": 10.2,
                "ma60": 9.6,
                "ma120": 11.1,
                "ma200": 9.1,
                "ma250": 10.0,
                "atr20": 0.5,
                "ma20_slope_10d": 0.01,
                "ma60_slope_20d": 0.01,
                "ma120_slope_20d": 0.0,
                "stock_q_blended": 12.0,
                "industry_q_blended": 12.0,
                "quality_pass": True,
                "cycle_peak_trap": False,
                "fundamental_break": False,
                "final_score": 82.0,
            },
            {
                "date": "2024-01-03",
                "symbol": "601398.sh",
                "name": "测试二号",
                "industry": "银行",
                "bucket": "defensive_dividend",
                "open": 11.0,
                "close": 11.0,
                "ma20": 10.2,
                "ma60": 9.6,
                "ma120": 11.1,
                "ma200": 9.1,
                "ma250": 10.0,
                "atr20": 0.5,
                "ma20_slope_10d": 0.01,
                "ma60_slope_20d": 0.01,
                "ma120_slope_20d": 0.0,
                "stock_q_blended": 12.0,
                "industry_q_blended": 12.0,
                "quality_pass": True,
                "cycle_peak_trap": False,
                "fundamental_break": False,
                "final_score": 82.0,
            },
        ]
    )
    benchmark = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "close": [100, 102]})
    result = engine.run(features=features, benchmark=benchmark, bucket="combined")
    assert result.approximate_backtest is False
    assert result.trade_list["symbol"].tolist() == ["600000.sh"]
