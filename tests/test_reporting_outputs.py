from __future__ import annotations

import json

from src.reporting.render import write_daily_report
from src.utils.config import resolve_path


def test_orders_latest_output_fields_complete(tmp_path) -> None:
    report = {
        "as_of_date": "2026-04-03",
        "data_status": {"decision_scope_rows": 1, "degraded": False, "safe_mode": False, "orders_degraded": False, "notes": []},
        "portfolio_state": {"market_regime": "risk_on", "max_total_position": 1.0, "account_state": {"current_cash": 10000, "latest_total_equity": 20000}},
        "summary_action": "BUY_1:1",
        "errors": [],
        "notes": [],
        "decisions": [],
        "frozen_holdings": [],
        "force_exit_list": [],
        "universe_change_summary": [],
        "orders": [
            {
                "symbol": "600000.sh",
                "name": "测试股",
                "holding_state": "NONE",
                "current_position_tranches": 0,
                "target_position_tranches": 1,
                "current_weight": 0.0,
                "target_weight": 0.03,
                "target_position_change": 0.03,
                "current_shares": 0,
                "target_shares": 600,
                "delta_shares": 600,
                "rounded_lots": 6,
                "action_enum": "BUY_1",
                "priority_score": 80.0,
                "action_reason": "满足第一笔买点。",
                "blocked_reason": None,
                "data_status": "ok",
                "latest_total_equity": 20000,
                "current_cash": 10000,
                "available_buying_power": 10000,
                "target_order_value": 600,
                "estimated_turnover": 600,
                "estimated_commission": 0.18,
                "estimated_stamp_duty": 0.0,
                "estimated_total_cash_impact": -600.18,
                "target_price_reference": 1.0,
            }
        ],
        "top_candidates_to_buy": [],
    }
    json_path = tmp_path / "daily.json"
    md_path = tmp_path / "daily.md"
    orders_json_path = tmp_path / "orders.json"
    orders_csv_path = tmp_path / "orders.csv"
    write_daily_report(report, json_path, md_path, orders_json_path, orders_csv_path)
    orders = json.loads(orders_json_path.read_text(encoding="utf-8"))
    expected = {
        "date",
        "symbol",
        "name",
        "holding_state",
        "current_position_tranches",
        "target_position_tranches",
        "current_weight",
        "target_weight",
        "target_position_change",
        "current_shares",
        "target_shares",
        "delta_shares",
        "rounded_lots",
        "action_enum",
        "priority_score",
        "action_reason",
        "blocked_reason",
        "data_status",
        "latest_total_equity",
        "current_cash",
        "available_buying_power",
        "target_order_value",
        "estimated_turnover",
        "estimated_commission",
        "estimated_stamp_duty",
        "estimated_total_cash_impact",
        "target_price_reference",
    }
    assert expected.issubset(set(orders[0].keys()))


def test_daily_decision_prompt_only_explains_actions() -> None:
    prompt = resolve_path("prompts/daily_decision.md").read_text(encoding="utf-8")
    assert "不得修改、覆盖、重算任何 `action_enum`" in prompt
    assert "不得替代规则引擎决定买卖" in prompt
