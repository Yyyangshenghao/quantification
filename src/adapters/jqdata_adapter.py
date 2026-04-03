from __future__ import annotations

import os

import pandas as pd

from src.adapters.base import DataAdapter
from src.adapters.common import normalize_frame, normalize_symbol, to_jq_symbol
from src.utils.exceptions import DataSourceError


class JQDataAdapter(DataAdapter):
    name = "jqdata"

    def __init__(self, config: dict) -> None:
        jq_cfg = config.get("jqdata", {})
        username_env = jq_cfg.get("username_env", "JQDATA_USERNAME")
        password_env = jq_cfg.get("password_env", "JQDATA_PASSWORD")
        self.username = os.getenv(username_env)
        self.password = os.getenv(password_env)
        self._authed = False

    def is_available(self) -> bool:
        return bool(self.username and self.password)

    def _auth(self):
        if not self.is_available():
            raise DataSourceError("JQData credentials missing; adapter skipped.")
        if self._authed:
            import jqdatasdk as jq

            return jq
        try:
            import jqdatasdk as jq
        except ImportError as exc:
            raise DataSourceError("jqdatasdk is not installed.") from exc
        jq.auth(self.username, self.password)
        self._authed = True
        return jq

    def get_stock_list(self, as_of_date: str) -> pd.DataFrame:
        jq = self._auth()
        frame = jq.get_all_securities(types=["stock"], date=as_of_date).reset_index()
        frame = frame.rename(columns={"index": "code", "display_name": "name", "start_date": "listed_date"})
        frame["code"] = frame["code"].map(normalize_symbol)
        frame["as_of_date"] = as_of_date
        return frame[["code", "name", "listed_date", "as_of_date"]]

    def get_price_daily(
        self, symbols: list[str], start_date: str, end_date: str, adjust: str
    ) -> pd.DataFrame:
        jq = self._auth()
        frame = jq.get_price(
            security=[to_jq_symbol(item) for item in symbols],
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            fields=["open", "high", "low", "close", "volume", "money"],
            fq=adjust,
            panel=False,
        )
        frame = frame.rename(columns={"time": "date", "code": "code", "money": "amount"})
        frame["code"] = frame["code"].map(normalize_symbol)
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        return frame[["code", "date", "open", "high", "low", "close", "volume", "amount"]]

    def get_index_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self.get_price_daily([symbol], start_date, end_date, adjust="none")

    def get_stock_valuation_history(self, symbol: str, metric: str, period: str) -> pd.DataFrame:
        jq = self._auth()
        from jqdatasdk import query, valuation

        query_field = getattr(valuation, metric)
        frame = jq.get_fundamentals_continuously(
            query(query_field, valuation.code, valuation.day).filter(valuation.code == to_jq_symbol(symbol)),
            end_date=period,
            count=2500,
        )
        if frame.empty:
            raise DataSourceError(f"JQData valuation history empty for {symbol} {metric}.")
        melted = frame.melt(id_vars=["day"], var_name="code", value_name="value")
        melted = melted.rename(columns={"day": "date"})
        melted["metric"] = metric
        melted["code"] = melted["code"].map(normalize_symbol)
        melted["date"] = pd.to_datetime(melted["date"]).dt.strftime("%Y-%m-%d")
        return melted[["code", "date", "metric", "value"]]

    def get_industry_daily(self, start_date: str, end_date: str, level: str) -> pd.DataFrame:
        raise DataSourceError("JQData industry daily endpoint is not implemented in this project.")

    def get_industry_members(self, industry_code: str, as_of_date: str | None = None) -> pd.DataFrame:
        jq = self._auth()
        members = jq.get_industry_stocks(industry_code, date=as_of_date)
        return pd.DataFrame(
            {
                "industry_code": industry_code,
                "code": [normalize_symbol(item) for item in members],
                "as_of_date": as_of_date,
            }
        )

    def get_financials(
        self, symbol: str, start_date: str | None = None, end_date: str | None = None
    ) -> pd.DataFrame:
        jq = self._auth()
        from jqdatasdk import balance, cash_flow, indicator, income, query, valuation

        q = (
            query(
                valuation.code,
                income.statDate,
                income.pubDate,
                indicator.roe,
                income.net_profit,
                cash_flow.net_operate_cash_flow,
                balance.total_assets,
                balance.total_liability,
            )
            .filter(valuation.code == to_jq_symbol(symbol))
            .limit(40)
        )
        frame = jq.get_fundamentals(q, date=end_date)
        if frame.empty:
            raise DataSourceError(f"JQData financials empty for {symbol}.")
        frame = frame.rename(
            columns={
                "statDate": "report_date",
                "pubDate": "announcement_date",
                "net_operate_cash_flow": "cfo",
            }
        )
        frame["code"] = frame["code"].map(normalize_symbol)
        frame["date"] = pd.to_datetime(frame["report_date"]).dt.strftime("%Y-%m-%d")
        frame["net_profit"] = pd.to_numeric(frame["net_profit"], errors="coerce")
        frame["cfo"] = pd.to_numeric(frame["cfo"], errors="coerce")
        frame["debt_to_assets"] = (
            pd.to_numeric(frame["total_liability"], errors="coerce")
            / pd.to_numeric(frame["total_assets"], errors="coerce")
            * 100
        )
        return frame[
            ["code", "date", "report_date", "announcement_date", "roe", "net_profit", "cfo", "debt_to_assets"]
        ]

    def get_st_flags(self, symbols: list[str], as_of_date: str) -> pd.DataFrame:
        jq = self._auth()
        extras = jq.get_extras(
            "is_st",
            [to_jq_symbol(item) for item in symbols],
            start_date=as_of_date,
            end_date=as_of_date,
            df=True,
        )
        records = []
        for code, value in extras.iloc[0].items():
            records.append({"code": normalize_symbol(code), "date": as_of_date, "is_st": bool(value)})
        return pd.DataFrame(records)

    def get_market_caps(self, symbols: list[str], as_of_date: str) -> pd.DataFrame:
        jq = self._auth()
        from jqdatasdk import query, valuation

        q = query(valuation.code, valuation.market_cap).filter(valuation.code.in_([to_jq_symbol(item) for item in symbols]))
        frame = jq.get_fundamentals(q, date=as_of_date)
        if frame.empty:
            raise DataSourceError("JQData market caps empty.")
        frame = normalize_frame(frame.rename(columns={"market_cap": "market_cap_billion"}))
        frame["code"] = frame["code"].map(normalize_symbol)
        frame["date"] = as_of_date
        return frame[["code", "date", "market_cap_billion"]]
