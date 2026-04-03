from __future__ import annotations

import pandas as pd

from src.pipeline.features import _merge_asof_by_key, prepare_financial_effective_frame


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
