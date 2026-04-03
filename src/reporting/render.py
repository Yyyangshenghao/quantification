from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.cli import write_json
from src.utils.config import resolve_path


def _format_action_line(item: dict) -> str:
    blocked = f" | blocked={item['blocked_reason']}" if item.get("blocked_reason") else ""
    shares = ""
    if item.get("delta_shares") not in (None, 0):
        shares = f" | shares {item.get('delta_shares'):+d}"
    return (
        f"- {item['symbol']} {item['name']}: {item['action_enum']} | "
        f"{item['current_position_tranches']} -> {item['target_position_tranches']} tranche | "
        f"w {item['current_weight']:.2%} -> {item['target_weight']:.2%}{shares}{blocked} | "
        f"{item['action_reason']}"
    )


def render_markdown(report: dict) -> str:
    regime = report["portfolio_state"]["market_regime"]
    account_state = report["portfolio_state"].get("account_state", {})
    lines = [
        "# 今日总览",
        "",
        f"- 日期: {report['as_of_date']}",
        f"- 汇总动作: {report['summary_action']}",
        f"- 决策范围: {report['data_status']['decision_scope_rows']} 只",
        "",
        "# 数据状态与是否进入 safe_mode",
        "",
        f"- degraded: {'是' if report['data_status']['degraded'] else '否'}",
        f"- safe_mode: {'是' if report['data_status']['safe_mode'] else '否'}",
        f"- orders_degraded: {'是' if report['data_status']['orders_degraded'] else '否'}",
        "",
        "# 市场 regime 与总仓位上限",
        "",
        f"- regime: {regime}",
        f"- max_total_position: {report['portfolio_state']['max_total_position']:.0%}",
        f"- current_cash: {account_state.get('current_cash')}",
        f"- latest_total_equity: {account_state.get('latest_total_equity')}",
        "",
        "# 当前持仓动作",
        "",
    ]
    held = [item for item in report["decisions"] if item["current_position_tranches"] > 0]
    if held:
        for item in held:
            lines.append(_format_action_line(item))
    else:
        lines.append("- 当前无手工持仓记录")

    lines.extend(["", "# FROZEN 持仓列表", ""])
    if report["frozen_holdings"]:
        for item in report["frozen_holdings"]:
            lines.append(_format_action_line(item))
    else:
        lines.append("- 无")

    lines.extend(["", "# 候选买入列表", ""])
    if report["top_candidates_to_buy"]:
        for item in report["top_candidates_to_buy"]:
            lines.append(_format_action_line(item))
    else:
        lines.append("- 无")

    lines.extend(["", "# 强制退出列表", ""])
    if report["force_exit_list"]:
        for item in report["force_exit_list"]:
            lines.append(_format_action_line(item))
    else:
        lines.append("- 无")

    lines.extend(["", "# 异常与降级", ""])
    if report["errors"]:
        for item in report["errors"]:
            lines.append(f"- {item}")
    else:
        lines.append("- 无新增错误")
    for note in report["notes"]:
        lines.append(f"- {note}")

    lines.extend(["", "# universe 变更摘要", ""])
    if report["universe_change_summary"]:
        for item in report["universe_change_summary"]:
            lines.append(f"- {item['change_tag']}: {item['symbol']} {item.get('name', '')}".rstrip())
    else:
        lines.append("- 非再平衡日或无变更")
    return "\n".join(lines).strip() + "\n"


def _orders_frame(report: dict) -> pd.DataFrame:
    fields = [
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
    ]
    rows = []
    for item in report.get("orders", []):
        row = {"date": report["as_of_date"]}
        for field in fields:
            row[field] = item.get(field)
        rows.append(row)
    return pd.DataFrame(rows)


def write_daily_report(
    report: dict,
    json_path: str | Path,
    markdown_path: str | Path,
    orders_json_path: str | Path = "reports/daily/orders_latest.json",
    orders_csv_path: str | Path = "reports/daily/orders_latest.csv",
) -> tuple[Path, Path, Path, Path]:
    json_file = write_json(json_path, report)
    markdown_file = resolve_path(markdown_path)
    markdown_file.parent.mkdir(parents=True, exist_ok=True)
    markdown_file.write_text(render_markdown(report), encoding="utf-8")

    orders = _orders_frame(report)
    orders_json_file = resolve_path(orders_json_path)
    orders_csv_file = resolve_path(orders_csv_path)
    orders_json_file.parent.mkdir(parents=True, exist_ok=True)
    orders.to_json(orders_json_file, orient="records", force_ascii=False, indent=2)
    orders.to_csv(orders_csv_file, index=False)
    return json_file, markdown_file, orders_json_file, orders_csv_file
