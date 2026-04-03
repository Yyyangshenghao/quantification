from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.adapters.akshare_adapter import AkshareAdapter
from src.adapters.baostock_adapter import BaoStockAdapter
from src.adapters.base import DataAdapter
from src.adapters.efinance_adapter import EfinanceAdapter
from src.adapters.jqdata_adapter import JQDataAdapter
from src.utils.exceptions import DataSourceError


@dataclass
class AdapterCallResult:
    adapter_name: str
    frame: pd.DataFrame | None = None
    error: str | None = None


class CompositeAdapter(DataAdapter):
    name = "composite"

    METHOD_ORDER_KEYS = {
        "get_stock_list": "stock_list_source_order",
        "get_price_daily": "price_source_order",
        "get_index_daily": "price_source_order",
        "get_stock_valuation_history": "valuation_source_order",
        "get_industry_daily": "industry_source_order",
        "get_industry_members": "industry_source_order",
        "get_financials": "financial_source_order",
        "get_st_flags": "st_flag_source_order",
        "get_market_caps": "market_cap_source_order",
    }

    def __init__(self, adapters: list[DataAdapter], source_order_map: dict[str, list[str]] | None = None) -> None:
        self.adapters = adapters
        self.adapters_by_name = {adapter.name: adapter for adapter in adapters}
        self.source_order_map = source_order_map or {}
        self.call_history: list[dict[str, object]] = []

    def _ordered_adapters(self, method_name: str) -> list[DataAdapter]:
        configured_names = self.source_order_map.get(method_name, [])
        ordered: list[DataAdapter] = []
        seen: set[str] = set()
        for name in configured_names:
            adapter = self.adapters_by_name.get(name)
            if adapter is None or name in seen:
                continue
            ordered.append(adapter)
            seen.add(name)
        for adapter in self.adapters:
            if adapter.name in seen:
                continue
            ordered.append(adapter)
            seen.add(adapter.name)
        return ordered

    def _dispatch(self, method_name: str, *args, **kwargs) -> pd.DataFrame:
        errors: list[str] = []
        for attempt_index, adapter in enumerate(self._ordered_adapters(method_name), start=1):
            if not adapter.is_available():
                self.call_history.append(
                    {
                        "method": method_name,
                        "adapter": adapter.name,
                        "success": False,
                        "error": "adapter_unavailable",
                        "attempt_index": attempt_index,
                    }
                )
                continue
            method = getattr(adapter, method_name)
            try:
                frame = method(*args, **kwargs)
                if not isinstance(frame, pd.DataFrame):
                    raise DataSourceError(f"{adapter.name}.{method_name} must return DataFrame.")
                if frame.empty:
                    raise DataSourceError(f"{adapter.name}.{method_name} returned empty DataFrame.")
                self.call_history.append(
                    {
                        "method": method_name,
                        "adapter": adapter.name,
                        "success": True,
                        "error": None,
                        "attempt_index": attempt_index,
                        "rows": int(len(frame)),
                    }
                )
                return frame
            except Exception as exc:
                errors.append(f"{adapter.name}: {exc}")
                self.call_history.append(
                    {
                        "method": method_name,
                        "adapter": adapter.name,
                        "success": False,
                        "error": str(exc),
                        "attempt_index": attempt_index,
                    }
                )
        raise DataSourceError(f"All adapters failed for {method_name}: {' | '.join(errors)}")

    def get_stock_list(self, as_of_date: str) -> pd.DataFrame:
        return self._dispatch("get_stock_list", as_of_date)

    def get_price_daily(
        self, symbols: list[str], start_date: str, end_date: str, adjust: str
    ) -> pd.DataFrame:
        return self._dispatch("get_price_daily", symbols, start_date, end_date, adjust)

    def get_index_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._dispatch("get_index_daily", symbol, start_date, end_date)

    def get_stock_valuation_history(self, symbol: str, metric: str, period: str) -> pd.DataFrame:
        return self._dispatch("get_stock_valuation_history", symbol, metric, period)

    def get_industry_daily(self, start_date: str, end_date: str, level: str) -> pd.DataFrame:
        return self._dispatch("get_industry_daily", start_date, end_date, level)

    def get_industry_members(self, industry_code: str, as_of_date: str | None = None) -> pd.DataFrame:
        return self._dispatch("get_industry_members", industry_code, as_of_date)

    def get_financials(
        self, symbol: str, start_date: str | None = None, end_date: str | None = None
    ) -> pd.DataFrame:
        return self._dispatch("get_financials", symbol, start_date, end_date)

    def get_st_flags(self, symbols: list[str], as_of_date: str) -> pd.DataFrame:
        return self._dispatch("get_st_flags", symbols, as_of_date)

    def get_market_caps(self, symbols: list[str], as_of_date: str) -> pd.DataFrame:
        return self._dispatch("get_market_caps", symbols, as_of_date)


def create_default_adapter(config: dict) -> CompositeAdapter:
    adapters: list[DataAdapter] = [
        JQDataAdapter(config),
        AkshareAdapter(config),
        BaoStockAdapter(config),
        EfinanceAdapter(config),
    ]
    defaults_cfg = config.get("defaults", {})
    source_order_map = {
        method_name: list(defaults_cfg.get(config_key, []))
        for method_name, config_key in CompositeAdapter.METHOD_ORDER_KEYS.items()
        if defaults_cfg.get(config_key)
    }
    return CompositeAdapter(adapters, source_order_map=source_order_map)
