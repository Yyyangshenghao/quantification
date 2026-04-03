#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.snapshot import prepare_daily_snapshot
from src.utils.cli import write_json
from src.utils.config import load_project_configs, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a daily snapshot JSON from local feature files.")
    parser.add_argument("--as-of-date", default="", help="As-of date in YYYY-MM-DD format; defaults to latest feature date.")
    parser.add_argument("--features-file", default="data/features/daily_features", help="Daily feature parquet file or partitioned directory.")
    parser.add_argument("--benchmark-file", default="data/raw/benchmark_daily.parquet", help="Benchmark price parquet.")
    parser.add_argument("--financials-file", default="data/curated/financials_effective.parquet", help="Point-in-time financial parquet.")
    parser.add_argument("--output-json", default="", help="Snapshot JSON path; defaults to date-based path.")
    return parser.parse_args()


def _infer_tranches(current_weight: float, tranche_weights: dict[int, float]) -> int:
    if current_weight <= 0:
        return 0
    ordered = sorted(tranche_weights.items(), key=lambda item: (abs(item[1] - current_weight), item[0]))
    return int(ordered[0][0]) if ordered else 0


def load_positions_frame(positions_cfg: dict, account_cfg: dict) -> pd.DataFrame:
    positions = positions_cfg.get("positions", [])
    if not positions:
        return pd.DataFrame(
            columns=[
                "symbol",
                "name",
                "current_shares",
                "avg_cost",
                "current_position_tranches",
                "current_weight",
                "extra_tranches",
                "last_fill_price",
            ]
        )
    frame = pd.DataFrame(positions)
    tranche_weights = {
        int(key): float(value)
        for key, value in (account_cfg.get("position_sizing", {}).get("tranche_weights", {}) or {}).items()
    }
    if "current_weight" not in frame.columns:
        frame["current_weight"] = pd.to_numeric(frame.get("current_position"), errors="coerce").fillna(0.0)
    else:
        frame["current_weight"] = pd.to_numeric(frame["current_weight"], errors="coerce").fillna(0.0)
    if "current_position_tranches" not in frame.columns:
        frame["current_position_tranches"] = frame["current_weight"].map(lambda value: _infer_tranches(float(value), tranche_weights))
    else:
        frame["current_position_tranches"] = pd.to_numeric(frame["current_position_tranches"], errors="coerce").fillna(0).astype(int)
    if "extra_tranches" not in frame.columns:
        frame["extra_tranches"] = 0
    if "last_fill_price" not in frame.columns:
        frame["last_fill_price"] = 0.0
    if "current_shares" not in frame.columns:
        frame["current_shares"] = 0
    else:
        frame["current_shares"] = pd.to_numeric(frame["current_shares"], errors="coerce").fillna(0).astype(int)
    if "avg_cost" not in frame.columns:
        frame["avg_cost"] = 0.0
    else:
        frame["avg_cost"] = pd.to_numeric(frame["avg_cost"], errors="coerce").fillna(0.0)
    if "name" not in frame.columns:
        frame["name"] = frame["symbol"]
    return frame[
        [
            "symbol",
            "name",
            "current_shares",
            "avg_cost",
            "current_position_tranches",
            "current_weight",
            "extra_tranches",
            "last_fill_price",
        ]
    ]


def main() -> int:
    args = parse_args()
    configs = load_project_configs()
    strategy_cfg = configs["strategy"]
    metric_map_cfg = configs["metric_map"]
    positions_cfg = configs["positions"]
    universe_cfg = configs["universe"]
    universe_rules_cfg = configs["universe_rules"]
    account_cfg = configs["account"]

    features = pd.read_parquet(resolve_path(args.features_file))
    benchmark = pd.read_parquet(resolve_path(args.benchmark_file))
    financials = pd.read_parquet(resolve_path(args.financials_file))
    as_of_date = args.as_of_date or str(features["date"].max())
    snapshot_features = features[features["date"] == as_of_date].copy()
    positions_frame = load_positions_frame(positions_cfg, account_cfg)
    report = prepare_daily_snapshot(
        as_of_date=as_of_date,
        feature_snapshot=snapshot_features,
        benchmark_frame=benchmark,
        positions_frame=positions_frame,
        financial_history=financials,
        strategy_cfg=strategy_cfg,
        metric_map_cfg=metric_map_cfg,
        universe_cfg=universe_cfg,
        universe_rules_cfg=universe_rules_cfg,
        account_cfg=account_cfg,
        data_errors=[],
        data_notes=[
            "开源模式下行业历史归属可能使用当前申万成分近似映射。",
            "若财报公告日缺失，按报告期 + 30 天生效。",
            "历史市值在开源模式下可能使用最近快照近似。",
        ],
    )
    output_json = args.output_json or f"data/snapshots/{as_of_date}.json"
    write_json(output_json, report)
    write_json("data/snapshots/latest.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
