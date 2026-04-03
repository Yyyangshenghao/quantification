from __future__ import annotations

import pandas as pd

from src.pipeline.features import _merge_asof_by_key, _normalize_industry_members, compute_price_features, prepare_financial_effective_frame


def test_prepare_financial_effective_frame_computes_roe_percentile_without_index_error(configs: dict) -> None:
    financials = pd.DataFrame(
        {
            "code": ["600000.sh", "600000.sh", "600000.sh"],
            "report_date": ["2024-03-31", "2024-06-30", "2024-09-30"],
            "announcement_date": ["2024-04-20", "2024-07-20", "2024-10-20"],
            "roe": [8.0, 10.0, 12.0],
            "net_profit": [10.0, 12.0, 15.0],
            "cfo": [5.0, 6.0, 7.0],
            "debt_to_assets": [50.0, 49.0, 48.0],
            "is_st": [False, False, False],
        }
    )
    result = prepare_financial_effective_frame(financials, configs["strategy"])
    assert "roe_pct_in_last_12_quarters" in result.columns
    assert float(result["roe_pct_in_last_12_quarters"].iloc[-1]) > 0


def test_merge_asof_by_key_handles_multiple_symbols_sorted_by_date_then_key() -> None:
    left = pd.DataFrame(
        {
            "symbol": ["b", "a", "b", "a"],
            "date": ["2024-01-02", "2024-01-01", "2024-01-03", "2024-01-03"],
            "value": [2, 1, 3, 4],
        }
    )
    right = pd.DataFrame(
        {
            "symbol": ["a", "b"],
            "date": ["2024-01-01", "2024-01-02"],
            "flag": [10, 20],
        }
    )
    merged = _merge_asof_by_key(left, right, by="symbol")
    assert list(merged["flag"]) == [10, 20, 10, 20]


def test_normalize_industry_members_prefers_level_hierarchy_and_l1_industry() -> None:
    members = pd.DataFrame(
        [
            {"code": "600000.sh", "name": "浦发银行", "industry_code": "801780.SI", "industry_name": "银行", "industry_level": "first"},
            {"code": "600000.sh", "name": "浦发银行", "industry_code": "801783.SI", "industry_name": "股份制银行Ⅱ", "industry_level": "second"},
            {"code": "600000.sh", "name": "浦发银行", "industry_code": "857831.SI", "industry_name": "股份制银行Ⅲ", "industry_level": "third"},
        ]
    )
    normalized = _normalize_industry_members(members)
    row = normalized.iloc[0].to_dict()
    assert row["industry"] == "银行"
    assert row["industry_code"] == "801780"
    assert row["industry_l1"] == "银行"
    assert row["industry_l2"] == "股份制银行Ⅱ"
    assert row["industry_l3"] == "股份制银行Ⅲ"


def test_compute_price_features_uses_benchmark_calendar_for_listed_days() -> None:
    prices = pd.DataFrame(
        {
            "code": ["600000.sh", "600000.sh"],
            "date": ["2026-04-02", "2026-04-03"],
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "volume": [1000, 1100],
            "amount": [1_000_000, 1_100_000],
        }
    )
    stock_list = pd.DataFrame([{"code": "600000.sh", "listed_date": "2026-04-02"}])
    benchmark = pd.DataFrame(
        {
            "code": ["000300.sh", "000300.sh", "000300.sh"],
            "date": ["2026-04-01", "2026-04-02", "2026-04-03"],
            "close": [1.0, 1.0, 1.0],
        }
    )

    features = compute_price_features(prices, stock_list=stock_list, benchmark_daily=benchmark)

    assert list(features["listed_days"].astype(int)) == [1, 2]
