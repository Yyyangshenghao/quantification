#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy.universe import (
    build_candidate_pool,
    build_effective_universe,
    effective_to_date,
    effective_universe_frame,
    is_rebalance_day,
    next_trading_day,
    serialize_universe_payload,
    universe_report_payload,
    write_universe_outputs,
)
from src.utils.config import load_project_configs, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the effective universe from rule-based candidate screening.")
    parser.add_argument("--as-of-date", default="", help="As-of date; defaults to latest available feature date.")
    parser.add_argument("--features-file", default="data/features/daily_features.parquet", help="Daily feature parquet.")
    parser.add_argument("--apply", action="store_true", help="Write the generated effective universe back to config/universe.yml.")
    parser.add_argument("--frequency", choices=["monthly", "quarterly"], default="", help="Override rebalance frequency for this run.")
    parser.add_argument("--preview-only", action="store_true", help="Preview the generated universe without applying it.")
    return parser.parse_args()


def _positions_frame(positions_cfg: dict) -> pd.DataFrame:
    positions = positions_cfg.get("positions", [])
    if not positions:
        return pd.DataFrame(columns=["symbol"])
    return pd.DataFrame(positions)


def main() -> int:
    args = parse_args()
    configs = load_project_configs()
    metric_map_cfg = configs["metric_map"]
    universe_cfg = configs["universe"]
    universe_rules_cfg = configs["universe_rules"].copy()
    positions_cfg = configs["positions"]
    features = pd.read_parquet(resolve_path(args.features_file))
    as_of_date = args.as_of_date or str(features["date"].max())
    frequency = args.frequency or universe_rules_cfg.get("rebalance_frequency", "monthly")
    universe_rules_cfg["rebalance_frequency"] = frequency
    latest = features[features["date"] == as_of_date].copy()
    current_holdings = _positions_frame(positions_cfg)
    rebalance_day = is_rebalance_day(features, as_of_date, frequency)
    has_existing_universe = not effective_universe_frame(universe_cfg).empty

    if not rebalance_day and has_existing_universe:
        payload = universe_cfg.copy()
        payload["as_of_date"] = as_of_date
        payload["rebalance_day"] = False
        report_payload = {
            "as_of_date": as_of_date,
            "rebalance_day": False,
            "effective_from": payload.get("effective_from"),
            "effective_to": payload.get("effective_to"),
            "methodology_version": payload.get("methodology_version"),
            "candidate_pool_size": 0,
            "effective_universe_size": len(payload.get("stocks", [])),
            "change_counts": {},
            "changes": [],
            "stocks": payload.get("stocks", []),
            "note": "non_rebalance_day_reuse_existing_universe",
        }
    else:
        candidate_pool = build_candidate_pool(latest, universe_rules_cfg, metric_map_cfg, as_of_date)
        selected, changes = build_effective_universe(candidate_pool, universe_cfg, current_holdings, universe_rules_cfg)
        payload = serialize_universe_payload(
            selected=selected,
            changes=changes,
            as_of_date=as_of_date,
            effective_from=next_trading_day(features, as_of_date),
            effective_to=effective_to_date(as_of_date, frequency),
            frequency=frequency,
            methodology_version=universe_rules_cfg["methodology_version"],
            rebalance_day=True,
        )
        report_payload = universe_report_payload(candidate_pool, selected, changes, payload)

    reports_dir = resolve_path("reports/universe")
    reports_dir.mkdir(parents=True, exist_ok=True)
    resolve_path("reports/universe/latest.json").write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# Effective Universe Preview ({report_payload['as_of_date']})",
        "",
        f"- Rebalance day: {'yes' if report_payload['rebalance_day'] else 'no'}",
        f"- Effective universe size: {report_payload['effective_universe_size']}",
        "",
    ]
    if report_payload.get("changes"):
        lines.extend(["## Diff", ""])
        for change in report_payload["changes"]:
            lines.append(f"- {change['change_tag']}: {change['symbol']} {change.get('name', '')}".rstrip())
        lines.append("")
    lines.extend(["## Stocks", ""])
    for stock in report_payload.get("stocks", []):
        lines.append(
            f"- {stock['symbol']} {stock['name']} | {stock['bucket']} | {stock['industry_l1']} | score={stock['final_score']:.2f}"
        )
    resolve_path("reports/universe/latest.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    if args.apply and not args.preview_only and report_payload["rebalance_day"]:
        write_universe_outputs(payload, report_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
