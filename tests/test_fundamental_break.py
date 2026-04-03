from __future__ import annotations

import pandas as pd

from src.strategy.fundamentals import (
    detect_financial_fundamental_break,
    detect_non_financial_fundamental_break,
    force_exit,
    force_exit_reasons,
)


def test_non_financial_fundamental_break_works(configs: dict) -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-03-31", "2024-06-30"],
            "latest_net_profit": [10, 10],
            "roe": [8, 4],
            "cfo_ttm": [-1, -2],
        }
    )
    assert detect_non_financial_fundamental_break(frame, configs["strategy"]) is True


def test_financial_fundamental_break_works(configs: dict) -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-06-30"],
            "latest_net_profit": [10],
            "roe": [10],
            "is_st": [True],
        }
    )
    assert detect_financial_fundamental_break(frame, configs["strategy"]) is True


def test_force_exit_reasons_include_st_and_data_failure(configs: dict) -> None:
    row = {"is_st": True, "fundamental_break": False, "delist_risk": False, "core_data_stale_days": 30}
    reasons = force_exit_reasons(row, configs["universe_rules"])
    assert "st" in reasons
    assert "prolonged_data_failure" in reasons
    assert force_exit(row, configs["universe_rules"]) is True
