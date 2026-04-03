from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataAdapter(ABC):
    name = "base"

    def is_available(self) -> bool:
        return True

    @abstractmethod
    def get_stock_list(self, as_of_date: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_price_daily(
        self, symbols: list[str], start_date: str, end_date: str, adjust: str
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_index_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_stock_valuation_history(self, symbol: str, metric: str, period: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_industry_daily(self, start_date: str, end_date: str, level: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_industry_members(self, industry_code: str, as_of_date: str | None = None) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_financials(
        self, symbol: str, start_date: str | None = None, end_date: str | None = None
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_st_flags(self, symbols: list[str], as_of_date: str) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_market_caps(self, symbols: list[str], as_of_date: str) -> pd.DataFrame:
        raise NotImplementedError
