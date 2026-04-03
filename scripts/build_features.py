#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.features import (
    build_daily_feature_panel,
    compute_industry_quantile_panel,
    compute_price_features,
    compute_stock_quantile_panel,
    prepare_financial_effective_frame,
)
from src.utils.config import load_project_configs, resolve_path
from src.utils.exceptions import DataSourceError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build point-in-time daily features from cached raw data.")
    parser.add_argument("--output-file", default="data/features/daily_features.parquet", help="Daily feature parquet output path.")
    parser.add_argument("--latest-output-file", default="data/features/latest_feature_snapshot.parquet", help="Latest feature snapshot output path.")
    return parser.parse_args()


def read_required(path_like: str) -> pd.DataFrame:
    path = resolve_path(path_like)
    if not path.exists():
        raise DataSourceError(f"Required raw dataset missing: {path}")
    return pd.read_parquet(path)


def main() -> int:
    args = parse_args()
    configs = load_project_configs()
    strategy_cfg = configs["strategy"]
    metric_map_cfg = configs["metric_map"]

    prices = read_required("data/raw/price_daily.parquet")
    stock_valuation = read_required("data/raw/stock_valuation.parquet")
    industry_daily = read_required("data/raw/industry_daily.parquet")
    industry_members = read_required("data/raw/industry_members.parquet")
    financials = read_required("data/raw/financials.parquet")
    st_flags = read_required("data/raw/st_flags.parquet")
    market_caps = read_required("data/raw/market_caps.parquet")

    price_features = compute_price_features(prices)
    financials_effective = prepare_financial_effective_frame(financials, strategy_cfg)
    stock_quantile_panel = compute_stock_quantile_panel(stock_valuation, strategy_cfg)
    industry_quantile_panel = compute_industry_quantile_panel(industry_daily, strategy_cfg, metric_map_cfg)
    daily_features = build_daily_feature_panel(
        price_features,
        financials_effective,
        stock_quantile_panel,
        industry_quantile_panel,
        industry_members,
        st_flags,
        market_caps,
        strategy_cfg,
        metric_map_cfg,
    )
    output_file = resolve_path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    daily_features.to_parquet(output_file, index=False)

    latest = daily_features.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1).copy()
    latest_output = resolve_path(args.latest_output_file)
    latest_output.parent.mkdir(parents=True, exist_ok=True)
    latest.to_parquet(latest_output, index=False)

    resolve_path("data/curated/financials_effective.parquet").parent.mkdir(parents=True, exist_ok=True)
    financials_effective.to_parquet(resolve_path("data/curated/financials_effective.parquet"), index=False)
    stock_quantile_panel.to_parquet(resolve_path("data/curated/stock_quantiles.parquet"), index=False)
    industry_quantile_panel.to_parquet(resolve_path("data/curated/industry_quantiles.parquet"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
