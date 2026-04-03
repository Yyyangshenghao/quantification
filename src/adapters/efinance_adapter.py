from __future__ import annotations

import pandas as pd

from src.adapters.base import DataAdapter
from src.adapters.common import normalize_frame, normalize_symbol
from src.utils.exceptions import DataSourceError


class EfinanceAdapter(DataAdapter):
    name = "efinance"

    def __init__(self, config: dict) -> None:
        self.config = config

    def _ef(self):
        try:
            import efinance as ef
        except ImportError as exc:
            raise DataSourceError("efinance is not installed.") from exc
        return ef

    def _get_base_info(self, symbols: list[str]) -> pd.DataFrame:
        requested_codes = [normalize_symbol(item).split(".")[0] for item in symbols]
        frame = self._ef().stock.get_base_info(requested_codes)
        if isinstance(frame, pd.Series):
            frame = frame.to_frame().T
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise DataSourceError("efinance base info returned no rows.")
        return frame

    def get_stock_list(self, as_of_date: str) -> pd.DataFrame:
        raise DataSourceError("efinance is only used as a price fallback in this project.")

    def get_price_daily(
        self, symbols: list[str], start_date: str, end_date: str, adjust: str
    ) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for symbol in symbols:
            code = symbol.split(".")[0]
            frame = self._ef().stock.get_quote_history(
                code,
                beg=start_date.replace("-", ""),
                end=end_date.replace("-", ""),
                klt=101,
                fqt={"none": 0, "qfq": 1, "hfq": 2}.get(adjust, 1),
            )
            if frame.empty:
                continue
            frame = normalize_frame(frame, {"code": normalize_symbol(code)})
            rows.append(frame[["code", "date", "open", "high", "low", "close", "volume", "amount"]])
        if not rows:
            raise DataSourceError("efinance price fallback returned no rows.")
        return pd.concat(rows, ignore_index=True)

    def get_index_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self.get_price_daily([symbol], start_date, end_date, adjust="none")

    def get_stock_valuation_history(self, symbol: str, metric: str, period: str) -> pd.DataFrame:
        raise DataSourceError("efinance does not provide valuation history here.")

    def get_industry_daily(self, start_date: str, end_date: str, level: str) -> pd.DataFrame:
        raise DataSourceError("efinance does not provide industry daily data here.")

    def get_industry_members(self, industry_code: str, as_of_date: str | None = None) -> pd.DataFrame:
        raise DataSourceError("efinance does not provide industry member data here.")

    def get_financials(
        self, symbol: str, start_date: str | None = None, end_date: str | None = None
    ) -> pd.DataFrame:
        raise DataSourceError("efinance does not provide financials here.")

    def get_st_flags(self, symbols: list[str], as_of_date: str) -> pd.DataFrame:
        frame = self._get_base_info(symbols).rename(columns={"股票代码": "code", "股票名称": "name"})
        if "code" not in frame.columns or "name" not in frame.columns:
            raise DataSourceError("efinance base info missing stock code/name columns.")
        frame["code"] = frame["code"].astype(str).map(normalize_symbol)
        frame["is_st"] = frame["name"].astype(str).str.contains("ST", case=False, na=False)
        frame["date"] = as_of_date
        filtered = frame[frame["code"].isin([normalize_symbol(item) for item in symbols])].copy()
        return filtered[["code", "date", "is_st"]]

    def get_market_caps(self, symbols: list[str], as_of_date: str) -> pd.DataFrame:
        frame = self._get_base_info(symbols).rename(columns={"股票代码": "code", "总市值": "market_cap"})
        if "code" not in frame.columns or "market_cap" not in frame.columns:
            raise DataSourceError("efinance base info missing market cap columns.")
        frame["code"] = frame["code"].astype(str).map(normalize_symbol)
        frame["market_cap_billion"] = pd.to_numeric(frame["market_cap"], errors="coerce") / 1e8
        frame["date"] = as_of_date
        filtered = frame[frame["code"].isin([normalize_symbol(item) for item in symbols])].copy()
        filtered = filtered.dropna(subset=["market_cap_billion"])
        if filtered.empty:
            raise DataSourceError("efinance market caps empty after normalization.")
        return filtered[["code", "date", "market_cap_billion"]]
