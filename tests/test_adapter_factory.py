from __future__ import annotations

import pandas as pd

from src.adapters.base import DataAdapter
from src.adapters.factory import CompositeAdapter
from src.utils.exceptions import DataSourceError


class StubAdapter(DataAdapter):
    def __init__(self, name: str, call_log: list[str], behaviors: dict[str, object]) -> None:
        self.name = name
        self.call_log = call_log
        self.behaviors = behaviors

    def _run(self, method_name: str) -> pd.DataFrame:
        self.call_log.append(f"{self.name}.{method_name}")
        behavior = self.behaviors.get(method_name)
        if isinstance(behavior, Exception):
            raise behavior
        if isinstance(behavior, pd.DataFrame):
            return behavior
        raise DataSourceError(f"{self.name}.{method_name} not stubbed")

    def get_stock_list(self, as_of_date: str) -> pd.DataFrame:
        return self._run("get_stock_list")

    def get_price_daily(
        self, symbols: list[str], start_date: str, end_date: str, adjust: str
    ) -> pd.DataFrame:
        return self._run("get_price_daily")

    def get_index_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._run("get_index_daily")

    def get_stock_valuation_history(self, symbol: str, metric: str, period: str) -> pd.DataFrame:
        return self._run("get_stock_valuation_history")

    def get_industry_daily(self, start_date: str, end_date: str, level: str) -> pd.DataFrame:
        return self._run("get_industry_daily")

    def get_industry_members(self, industry_code: str, as_of_date: str | None = None) -> pd.DataFrame:
        return self._run("get_industry_members")

    def get_financials(
        self, symbol: str, start_date: str | None = None, end_date: str | None = None
    ) -> pd.DataFrame:
        return self._run("get_financials")

    def get_st_flags(self, symbols: list[str], as_of_date: str) -> pd.DataFrame:
        return self._run("get_st_flags")

    def get_market_caps(self, symbols: list[str], as_of_date: str) -> pd.DataFrame:
        return self._run("get_market_caps")


def test_composite_adapter_honors_method_specific_source_order() -> None:
    calls: list[str] = []
    result = pd.DataFrame([{"code": "600519.sh"}])
    adapter_a = StubAdapter("akshare", calls, {"get_price_daily": pd.DataFrame([{"code": "000001.sz"}])})
    adapter_b = StubAdapter("baostock", calls, {"get_price_daily": result})
    adapter_c = StubAdapter("efinance", calls, {"get_price_daily": pd.DataFrame([{"code": "300750.sz"}])})

    composite = CompositeAdapter(
        [adapter_a, adapter_b, adapter_c],
        source_order_map={"get_price_daily": ["baostock", "akshare"]},
    )

    frame = composite.get_price_daily(["600519.sh"], "2024-12-01", "2024-12-31", "qfq")

    assert frame.equals(result)
    assert calls == ["baostock.get_price_daily"]


def test_composite_adapter_appends_unconfigured_adapters_as_fallback() -> None:
    calls: list[str] = []
    adapter_a = StubAdapter("akshare", calls, {"get_market_caps": DataSourceError("akshare failed")})
    adapter_b = StubAdapter("baostock", calls, {"get_market_caps": DataSourceError("baostock failed")})
    adapter_c = StubAdapter("efinance", calls, {"get_market_caps": pd.DataFrame([{"code": "600519.sh"}])})

    composite = CompositeAdapter(
        [adapter_a, adapter_b, adapter_c],
        source_order_map={"get_market_caps": ["baostock"]},
    )

    frame = composite.get_market_caps(["600519.sh"], "2024-12-31")

    assert frame.to_dict(orient="records") == [{"code": "600519.sh"}]
    assert calls == [
        "baostock.get_market_caps",
        "akshare.get_market_caps",
        "efinance.get_market_caps",
    ]
