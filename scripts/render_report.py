#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.render import write_daily_report
from src.utils.config import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Markdown/JSON daily reports and order files from a snapshot JSON.")
    parser.add_argument("--snapshot-json", default="data/snapshots/latest.json", help="Snapshot JSON file.")
    parser.add_argument("--output-json", default="reports/daily/latest.json", help="Rendered JSON path.")
    parser.add_argument("--output-md", default="reports/daily/latest.md", help="Rendered Markdown path.")
    parser.add_argument("--orders-json", default="reports/daily/orders_latest.json", help="Orders JSON path.")
    parser.add_argument("--orders-csv", default="reports/daily/orders_latest.csv", help="Orders CSV path.")
    parser.add_argument("--codex-note-file", default="", help="Optional Codex CLI markdown file to append into notes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(resolve_path(args.snapshot_json).read_text(encoding="utf-8"))
    if args.codex_note_file:
        note_path = resolve_path(args.codex_note_file)
        if note_path.exists():
            report.setdefault("notes", []).append(note_path.read_text(encoding="utf-8").strip())
    write_daily_report(report, args.output_json, args.output_md, args.orders_json, args.orders_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
