from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.adapters.base import DataAdapter
from src.adapters.common import is_a_share_equity_symbol, normalize_symbol, to_baostock_symbol
from src.utils.exceptions import DataSourceError


@dataclass
class _BaoStockSession:
    active: bool = False


class BaoStockAdapter(DataAdapter):
    name = "baostock"

    def __init__(self, config: dict) -> None:
        self.config = config
        self.session = _BaoStockSession()

    def _bs(self):
        try:
            import baostock as bs
        except ImportError as exc:
            raise DataSourceError("baostock is not installed.") from exc
        if not self.session.active:
            login_result = bs.login()
            if login_result.error_code != "0":
                raise DataSourceError(f"BaoStock login failed: {login_result.error_msg}")
            self.session.active = True
        return bs

    @staticmethod
    def _result_to_frame(result) -> pd.DataFrame:
        rows = []
        while result.error_code == "0" and result.next():
            rows.append(result.get_row_data())
        return pd.DataFrame(rows, columns=result.fields)

    def get_stock_list(self, as_of_date: str) -> pd.DataFrame:
        bs = self._bs()
        frame = self._result_to_frame(bs.query_all_stock(day=as_of_date))
        if frame.empty:
            raise DataSourceError("BaoStock stock list empty.")
        frame = frame.rename(columns={"code_name": "name"})
        frame["code"] = frame["code"].map(normalize_symbol)
        frame = frame[frame["code"].map(is_a_share_equity_symbol)].copy()
        if frame.empty:
            raise DataSourceError("BaoStock stock list contains no A-share equities after filtering.")
        frame["as_of_date"] = as_of_date
        return frame[["code", "name", "as_of_date"]]

    def get_price_daily(
        self, symbols: list[str], start_date: str, end_date: str, adjust: str
    ) -> pd.DataFrame:
        bs = self._bs()
        rows: list[pd.DataFrame] = []
        fields = "date,code,open,high,low,close,volume,amount,turn"
        adjust_flag = {"none": "3", "qfq": "2", "hfq": "1"}.get(adjust, "2")
        for symbol in symbols:
            result = bs.query_history_k_data_plus(
                to_baostock_symbol(symbol),
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag=adjust_flag,
            )
            frame = self._result_to_frame(result)
            if frame.empty:
                continue
            rows.append(frame)
        if not rows:
            raise DataSourceError("BaoStock price_daily returned no rows.")
        frame = pd.concat(rows, ignore_index=True)
        numeric_columns = ["open", "high", "low", "close", "volume", "amount", "turn"]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["code"] = frame["code"].map(normalize_symbol)
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        frame = frame.dropna(subset=["date", "code", "close"]).drop_duplicates(subset=["code", "date"])
        return frame[["code", "date", "open", "high", "low", "close", "volume", "amount", "turn"]]

    def get_index_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self.get_price_daily([symbol], start_date, end_date, adjust="none")

    def get_stock_valuation_history(self, symbol: str, metric: str, period: str) -> pd.DataFrame:
        bs = self._bs()
        metric_map = {
            "pb": "pbMRQ",
            "pe_ttm": "peTTM",
            "ps_ttm": "psTTM",
            "pcf_ncf_ttm": "pcfNcfTTM",
        }
        source_column = metric_map.get(metric)
        if source_column is None:
            raise DataSourceError(f"BaoStock does not support metric {metric}.")
        result = bs.query_history_k_data_plus(
            to_baostock_symbol(symbol),
            f"date,code,{source_column}",
            start_date="2016-01-01",
            end_date=period,
            frequency="d",
            adjustflag="3",
        )
        frame = self._result_to_frame(result)
        if frame.empty:
            raise DataSourceError(f"BaoStock valuation history empty for {symbol} {metric}.")
        frame["code"] = frame["code"].map(normalize_symbol)
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        frame[source_column] = pd.to_numeric(frame[source_column], errors="coerce")
        frame["metric"] = metric
        frame = frame.rename(columns={source_column: "value"}).dropna(subset=["value"])
        if frame.empty:
            raise DataSourceError(f"BaoStock valuation history has no numeric values for {symbol} {metric}.")
        return frame[["code", "date", "metric", "value"]]

    def get_industry_daily(self, start_date: str, end_date: str, level: str) -> pd.DataFrame:
        raise DataSourceError("BaoStock does not provide SW industry daily data.")

    def get_industry_members(self, industry_code: str, as_of_date: str | None = None) -> pd.DataFrame:
        raise DataSourceError("BaoStock does not provide industry member data.")

    def get_financials(
        self, symbol: str, start_date: str | None = None, end_date: str | None = None
    ) -> pd.DataFrame:
        bs = self._bs()
        start_year = pd.Timestamp(start_date or "2016-01-01").year
        end_year = pd.Timestamp(end_date or pd.Timestamp.today()).year
        records: list[pd.DataFrame] = []
        bs_code = to_baostock_symbol(symbol)
        for year in range(start_year, end_year + 1):
            for quarter in (1, 2, 3, 4):
                profit = self._result_to_frame(bs.query_profit_data(code=bs_code, year=year, quarter=quarter))
                cash_flow = self._result_to_frame(bs.query_cash_flow_data(code=bs_code, year=year, quarter=quarter))
                dupont = self._result_to_frame(bs.query_dupont_data(code=bs_code, year=year, quarter=quarter))
                balance = self._result_to_frame(bs.query_balance_data(code=bs_code, year=year, quarter=quarter))
                merged = profit.copy()
                for extra in (cash_flow, dupont, balance):
                    if not extra.empty:
                        merged = merged.merge(extra, how="outer", on=["code", "pubDate", "statDate"])
                if not merged.empty:
                    records.append(merged)
        if not records:
            raise DataSourceError(f"BaoStock financials empty for {symbol}.")
        frame = pd.concat(records, ignore_index=True)
        frame = frame.rename(
            columns={
                "pubDate": "announcement_date",
                "statDate": "report_date",
                "roeAvg": "roe",
                "dupontROE": "roe_dupont",
                "netProfit": "net_profit",
            }
        )
        if "roe" not in frame.columns and "roe_dupont" in frame.columns:
            frame["roe"] = pd.to_numeric(frame["roe_dupont"], errors="coerce")
        if "cfo" not in frame.columns and {"CFOToNP", "net_profit"} <= set(frame.columns):
            frame["cfo"] = pd.to_numeric(frame["CFOToNP"], errors="coerce") * pd.to_numeric(frame["net_profit"], errors="coerce")
        if "debt_to_assets" not in frame.columns and {"liabilityToAsset", "debtToAssetRatio"} & set(frame.columns):
            for candidate in ("liabilityToAsset", "debtToAssetRatio"):
                if candidate in frame.columns:
                    frame["debt_to_assets"] = pd.to_numeric(frame[candidate], errors="coerce")
                    break
        numeric_columns = ["roe", "net_profit", "cfo", "debt_to_assets"]
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            else:
                frame[column] = pd.NA
        frame["code"] = frame["code"].map(normalize_symbol)
        frame["date"] = pd.to_datetime(frame["report_date"]).dt.strftime("%Y-%m-%d")
        frame["is_st"] = False
        frame = frame.drop_duplicates(subset=["code", "report_date", "announcement_date"]).sort_values("report_date")
        return frame[
            [
                "code",
                "date",
                "report_date",
                "announcement_date",
                "roe",
                "net_profit",
                "cfo",
                "debt_to_assets",
                "is_st",
            ]
        ]

    def get_st_flags(self, symbols: list[str], as_of_date: str) -> pd.DataFrame:
        listing = self.get_stock_list(as_of_date)
        listing["is_st"] = listing["name"].astype(str).str.contains("ST", case=False, na=False)
        listing["date"] = as_of_date
        filtered = listing[listing["code"].isin([normalize_symbol(item) for item in symbols])].copy()
        return filtered[["code", "date", "is_st"]]

    def get_market_caps(self, symbols: list[str], as_of_date: str) -> pd.DataFrame:
        raise DataSourceError("BaoStock market cap retrieval is not implemented; use AkShare or JQData.")
