#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_project_configs, resolve_path
from src.utils.ledger import rebuild_positions_from_ledger, write_positions_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild config/positions.yml from the local trade ledger.")
    parser.add_argument("--ledger-csv", default="data/ledger/trades.csv", help="Trade ledger CSV.")
    parser.add_argument("--price-file", default="data/raw/price_daily.parquet", help="Latest price parquet for weight calculation.")
    parser.add_argument("--output-yaml", default="config/positions.yml", help="Rebuilt positions YAML.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = load_project_configs()
    account_cfg = configs["account"]
    strategy_cfg = configs["strategy"]
    ledger_file = resolve_path(args.ledger_csv)
    trades = pd.read_csv(ledger_file) if ledger_file.exists() else pd.DataFrame()
    prices = pd.read_parquet(resolve_path(args.price_file)) if resolve_path(args.price_file).exists() else pd.DataFrame(columns=["code", "date", "close"])
    latest_total_equity = float(account_cfg.get("account", {}).get("latest_total_equity", 0.0) or 0.0)
    tranche_weights = {
        int(key): float(value)
        for key, value in (account_cfg.get("position_sizing", {}).get("tranche_weights", {}) or {}).items()
    }
    if 0 not in tranche_weights:
        tranche_weights[0] = 0.0
    positions = rebuild_positions_from_ledger(
        trades,
        latest_prices=prices,
        latest_total_equity=latest_total_equity,
        tranche_weights=tranche_weights,
    )
    write_positions_yaml(
        args.output_yaml,
        positions=positions,
        default_tranches=int(strategy_cfg["execution"]["default_tranches"]),
        latest_total_equity=latest_total_equity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
