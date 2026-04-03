#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.strategy.backtest_engine import BacktestEngine
from src.utils.cli import write_json
from src.utils.config import load_project_configs, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run point-in-time backtests using next-day-open execution.")
    parser.add_argument("--bucket", default="combined", choices=["defensive_dividend", "cyclical_rotation", "combined"], help="Backtest bucket.")
    parser.add_argument("--start-date", required=True, help="Backtest start date.")
    parser.add_argument("--end-date", required=True, help="Backtest end date.")
    parser.add_argument("--features-file", default="data/features/daily_features.parquet", help="Daily feature parquet.")
    parser.add_argument("--benchmark-file", default="data/raw/benchmark_daily.parquet", help="Benchmark parquet.")
    parser.add_argument("--output-prefix", default="", help="Backtest output prefix.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = load_project_configs()
    features = pd.read_parquet(resolve_path(args.features_file))
    benchmark = pd.read_parquet(resolve_path(args.benchmark_file))
    features = features[(features["date"] >= args.start_date) & (features["date"] <= args.end_date)].copy()
    benchmark = benchmark[(benchmark["date"] >= args.start_date) & (benchmark["date"] <= args.end_date)].copy()

    engine = BacktestEngine(
        configs["strategy"],
        universe_rules_cfg=configs["universe_rules"],
        account_cfg=configs["account"],
    )
    result = engine.run(features=features, benchmark=benchmark, bucket=args.bucket)

    prefix = args.output_prefix or f"reports/backtests/{args.bucket}_{args.start_date}_{args.end_date}"
    json_path = f"{prefix}.json"
    md_path = f"{prefix}.md"
    payload = {
        "bucket": args.bucket,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "metrics": result.metrics,
        "trade_list": result.trade_list.to_dict(orient="records"),
        "stock_attribution": result.stock_attribution.to_dict(orient="records"),
        "industry_attribution": result.industry_attribution.to_dict(orient="records"),
    }
    write_json(json_path, payload)
    lines = [
        f"# Backtest {args.bucket}",
        "",
        f"- Start: {args.start_date}",
        f"- End: {args.end_date}",
        f"- CAGR: {result.metrics['cagr']:.2%}",
        f"- Max Drawdown: {result.metrics['max_drawdown']:.2%}",
        f"- Win Rate: {result.metrics['win_rate']:.2%}",
        f"- Sharpe: {result.metrics['sharpe']:.2f}",
        "",
        "## Trades",
        "",
    ]
    if result.trade_list.empty:
        lines.append("- No trades")
    else:
        for row in result.trade_list.to_dict(orient="records"):
            lines.append(
                f"- {row['signal_date']} -> {row['fill_date']} | {row['symbol']} | {row['action']} | fill={row['fill_price']:.4f} | pnl={row['realized_pnl']:.2f}"
            )
    md_file = resolve_path(md_path)
    md_file.parent.mkdir(parents=True, exist_ok=True)
    md_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
