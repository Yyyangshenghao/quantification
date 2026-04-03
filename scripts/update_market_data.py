#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adapters.akshare_adapter import AkshareAdapter
from src.adapters.factory import create_default_adapter
from src.utils.config import load_project_configs, resolve_path
from src.utils.exceptions import DataSourceError
from src.utils.ops import write_data_quality_report, write_provider_health_report
from src.utils.storage import clear_directories, configured_cache_directories


RAW_FILES = {
    "stock_list": "data/raw/stock_list.parquet",
    "price_daily": "data/raw/price_daily.parquet",
    "benchmark_daily": "data/raw/benchmark_daily.parquet",
    "stock_valuation": "data/raw/stock_valuation.parquet",
    "industry_daily": "data/raw/industry_daily.parquet",
    "industry_members": "data/raw/industry_members.parquet",
    "financials": "data/raw/financials.parquet",
    "st_flags": "data/raw/st_flags.parquet",
    "market_caps": "data/raw/market_caps.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and cache A-share research data locally.")
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD format.")
    parser.add_argument("--price-start-date", default="", help="Optional override for price history fetch start date.")
    parser.add_argument("--benchmark-start-date", default="", help="Optional override for benchmark fetch start date.")
    parser.add_argument("--industry-start-date", default="", help="Optional override for industry history fetch start date.")
    parser.add_argument(
        "--financial-start-date",
        default="",
        help="Optional override for financial statement fetch start date. Defaults to --start-date.",
    )
    parser.add_argument("--skip-valuations", action="store_true", help="Skip stock valuation history refresh.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark refresh.")
    parser.add_argument("--symbols", default="", help="Comma-separated stock symbols.")
    parser.add_argument("--adjust", default="qfq", help="Price adjustment: none/qfq/hfq.")
    parser.add_argument("--benchmark", default="000300", help="Benchmark index symbol.")
    parser.add_argument("--all-stocks", action="store_true", help="Fetch all stocks from the active source.")
    parser.add_argument("--max-symbols", type=int, default=0, help="Optional cap for requested symbols.")
    return parser.parse_args()


def append_dataset(path_like: str, frame: pd.DataFrame, subset: list[str]) -> Path:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = pd.read_parquet(path)
        frame = pd.concat([current, frame], ignore_index=True)
    frame = frame.drop_duplicates(subset=subset).sort_values(subset).reset_index(drop=True)
    frame.to_parquet(path, index=False)
    return path


def discover_industries(data_cfg: dict) -> pd.DataFrame:
    ak = AkshareAdapter(data_cfg)
    frames: list[pd.DataFrame] = []
    for fn_name, level in (
        ("sw_index_first_info", "first"),
        ("sw_index_second_info", "second"),
        ("sw_index_third_info", "third"),
    ):
        try:
            frame = ak._invoke(fn_name)  # noqa: SLF001 - internal helper is acceptable in this repo script
        except Exception:
            continue
        if frame.empty:
            continue
        frame = frame.rename(columns={"行业代码": "industry_code", "行业名称": "industry_name", "指数代码": "industry_code", "指数名称": "industry_name"})
        if {"industry_code", "industry_name"} <= set(frame.columns):
            frame["level"] = level
            frames.append(frame[["industry_code", "industry_name", "level"]].drop_duplicates())
    if not frames:
        raise DataSourceError("Unable to discover SW industry catalog from AkShare.")
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["industry_code"])


def main() -> int:
    args = parse_args()
    configs = load_project_configs()
    data_cfg = configs["data_sources"]
    if data_cfg.get("cache", {}).get("clear_on_run", False):
        clear_directories(configured_cache_directories(data_cfg))
    adapter = create_default_adapter(data_cfg)
    provider_entries: list[dict[str, object]] = []

    stock_list = adapter.get_stock_list(args.end_date)
    append_dataset(RAW_FILES["stock_list"], stock_list, ["code"])

    if args.symbols:
        symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    else:
        symbols = stock_list["code"].tolist()
    if args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]
    if not symbols:
        raise DataSourceError("No symbols selected for update_market_data.py")
    price_start_date = args.price_start_date or args.start_date
    benchmark_start_date = args.benchmark_start_date or args.start_date
    industry_start_date = args.industry_start_date or args.start_date
    financial_start_date = args.financial_start_date or args.start_date

    if not args.skip_benchmark:
        try:
            benchmark = adapter.get_index_daily(args.benchmark, benchmark_start_date, args.end_date)
        except DataSourceError as exc:
            print(
                f"[warn] benchmark download failed for {args.benchmark}: {exc}. "
                "Continuing with stock data refresh; backtests should provide a local benchmark fallback.",
                file=sys.stderr,
            )
            provider_entries.append(
                {"method": "get_index_daily", "adapter": "composite", "success": False, "error": str(exc), "attempt_index": 0}
            )
        else:
            append_dataset(RAW_FILES["benchmark_daily"], benchmark, ["code", "date"])

    prices = adapter.get_price_daily(symbols, price_start_date, args.end_date, args.adjust)
    append_dataset(RAW_FILES["price_daily"], prices, ["code", "date"])

    st_flags = adapter.get_st_flags(symbols, args.end_date)
    append_dataset(RAW_FILES["st_flags"], st_flags, ["code", "date"])

    market_caps = adapter.get_market_caps(symbols, args.end_date)
    append_dataset(RAW_FILES["market_caps"], market_caps, ["code", "date"])

    financial_frames: list[pd.DataFrame] = []
    for symbol in symbols:
        try:
            frame = adapter.get_financials(symbol, start_date=financial_start_date, end_date=args.end_date)
        except DataSourceError as exc:
            print(f"[warn] financial download failed for {symbol}: {exc}", file=sys.stderr)
            continue
        frame["code"] = symbol
        financial_frames.append(frame)
    if financial_frames:
        financials = pd.concat(financial_frames, ignore_index=True)
        append_dataset(RAW_FILES["financials"], financials, ["code", "report_date"])
    else:
        print("[warn] financial download failed for all selected symbols.", file=sys.stderr)

    industry_catalog = discover_industries(data_cfg)
    provider_entries.append(
        {"method": "discover_industries", "adapter": "akshare", "success": True, "error": None, "attempt_index": 1, "rows": int(len(industry_catalog))}
    )
    member_frames: list[pd.DataFrame] = []
    for industry_code in industry_catalog["industry_code"].astype(str):
        try:
            frame = adapter.get_industry_members(industry_code, as_of_date=args.end_date)
        except Exception:
            continue
        if frame.empty:
            continue
        matched = industry_catalog.loc[industry_catalog["industry_code"] == industry_code, ["industry_name", "level"]].iloc[0]
        industry_name = matched["industry_name"]
        frame["industry_name"] = industry_name
        frame["industry_level"] = matched["level"]
        member_frames.append(frame)
    if not member_frames:
        raise DataSourceError("Industry member download failed for all SW industries.")
    industry_members = pd.concat(member_frames, ignore_index=True)
    append_dataset(RAW_FILES["industry_members"], industry_members, ["industry_code", "code"])

    industry_daily_frames: list[pd.DataFrame] = []
    for level_label, level_value in (("first", "一级行业"), ("second", "二级行业"), ("third", "三级行业")):
        frame = adapter.get_industry_daily(industry_start_date, args.end_date, level=level_value)
        frame["industry_level"] = level_label
        industry_daily_frames.append(frame)
    industry_daily = pd.concat(industry_daily_frames, ignore_index=True)
    append_dataset(RAW_FILES["industry_daily"], industry_daily, ["industry_code", "date"])

    if not args.skip_valuations:
        valuation_frames: list[pd.DataFrame] = []
        for symbol in symbols:
            for metric in ("pb", "pe_ttm"):
                try:
                    frame = adapter.get_stock_valuation_history(symbol, metric=metric, period=args.end_date)
                except Exception:
                    print(f"[warn] valuation download failed for {symbol} {metric}", file=sys.stderr)
                    continue
                valuation_frames.append(frame)
        if not valuation_frames:
            raise DataSourceError("Stock valuation download failed for all selected symbols.")
        valuations = pd.concat(valuation_frames, ignore_index=True)
        append_dataset(RAW_FILES["stock_valuation"], valuations, ["code", "date", "metric"])
    provider_entries.extend(adapter.call_history)
    write_provider_health_report(
        "provider_health/latest.json",
        as_of_date=args.end_date,
        entries=provider_entries,
        source="update_market_data",
    )
    write_data_quality_report("data_quality/latest.json", list(RAW_FILES.values()), args.end_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
