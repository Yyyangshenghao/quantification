from __future__ import annotations

import pandas as pd

from src.utils.ledger import rebuild_positions_from_ledger


def test_rebuild_positions_from_ledger_calculates_shares_weight_and_tranches() -> None:
    trades = pd.DataFrame(
        [
            {
                "trade_date": "2026-03-15",
                "symbol": "600036.sh",
                "name": "招商银行",
                "side": "BUY",
                "shares": 200,
                "price": 29.0,
                "commission": 2.0,
                "stamp_duty": 0.0,
                "other_fees": 0.0,
                "source": "manual_import",
                "notes": "",
            }
        ]
    )
    latest_prices = pd.DataFrame([{"code": "600036.sh", "date": "2026-04-30", "close": 30.0}])
    positions = rebuild_positions_from_ledger(
        trades,
        latest_prices=latest_prices,
        latest_total_equity=200000.0,
        tranche_weights={0: 0.0, 1: 0.03, 2: 0.06, 3: 0.09},
    )
    assert positions == [
        {
            "symbol": "600036.sh",
            "name": "招商银行",
            "current_shares": 200,
            "avg_cost": 29.01,
            "current_weight": 0.03,
            "current_position_tranches": 1,
            "extra_tranches": 0,
            "last_fill_price": 29.0,
        }
    ]
