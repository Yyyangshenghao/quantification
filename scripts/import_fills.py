#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import resolve_path
from src.utils.ledger import append_fills_to_ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import manual fills CSV into the local trade ledger.")
    parser.add_argument("--fills-csv", required=True, help="Source fills CSV.")
    parser.add_argument("--ledger-csv", default="data/ledger/trades.csv", help="Target ledger CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fills = pd.read_csv(resolve_path(args.fills_csv))
    append_fills_to_ledger(fills, args.ledger_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
