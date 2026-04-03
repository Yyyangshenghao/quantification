from __future__ import annotations

import inspect

import pandas as pd

from src.adapters.base import DataAdapter
from src.adapters.common import cache_file, normalize_frame, normalize_symbol, retry_call
from src.utils.config import resolve_path
from src.utils.exceptions import DataSourceError


class AkshareAdapter(DataAdapter):
    name = "akshare"

    def __init__(self, config: dict) -> None:
        self.config = config
        self.retry_cfg = config.get("retry", {})
        raw_dir = config.get("storage", {}).get("raw_dir", "data/raw")
        self.cache_root = resolve_path(raw_dir) / self.name

    def _ak(self):
        try:
            import akshare as ak
        except ImportError as exc:
            raise DataSourceError("akshare is not installed.") from exc
        return ak

    def _call_with_cache(self, namespace: str, loader, *cache_parts: object) -> pd.DataFrame:
        path = cache_file(self.cache_root, namespace, *cache_parts)
        if path.exists():
            return pd.read_parquet(path)
        frame = retry_call(
            loader,
            attempts=int(self.retry_cfg.get("attempts", 3)),
            backoff_seconds=float(self.retry_cfg.get("backoff_seconds", 1.5)),
        )
        frame.to_parquet(path, index=False)
        return frame

    def _invoke(self, fn_name: str, **kwargs):
        fn = getattr(self._ak(), fn_name)
        signature = inspect.signature(fn)
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return fn(**filtered_kwargs)

    def get_stock_list(self, as_of_date: str) -> pd.DataFrame:
        frame = self._call_with_cache(
            "stock_list",
            lambda: self._invoke("stock_info_a_code_name"),
            as_of_date,
        )
        renamed = frame.rename(columns={"code": "code", "name": "name"})
        renamed["code"] = renamed["code"].map(normalize_symbol)
        renamed["as_of_date"] = as_of_date
        return renamed[["code", "name", "as_of_date"]]

    def get_price_daily(
        self, symbols: list[str], start_date: str, end_date: str, adjust: str
    ) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for symbol in symbols:
            code = symbol.split(".")[0]
            frame = self._call_with_cache(
                "price_daily",
                lambda code=code: self._invoke(
                    "stock_zh_a_hist",
                    symbol=code,
                    period="daily",
                    start_date=start_date.replace("-", ""),
                    end_date=end_date.replace("-", ""),
                    adjust=adjust,
                ),
                code,
                start_date,
                end_date,
                adjust,
            )
            frame = normalize_frame(frame, {"code": normalize_symbol(code)})
            rows.append(frame[["code", "date", "open", "high", "low", "close", "volume", "amount"]])
        if not rows:
            raise DataSourceError("AkShare price_daily received no symbols.")
        return pd.concat(rows, ignore_index=True)

    def get_index_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        frame = self._call_with_cache(
            "index_daily",
            lambda: self._invoke(
                "index_zh_a_hist",
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            ),
            symbol,
            start_date,
            end_date,
        )
        frame = normalize_frame(frame, {"code": normalize_symbol(symbol)})
        return frame[["code", "date", "open", "high", "low", "close", "volume", "amount"]]

    def get_stock_valuation_history(self, symbol: str, metric: str, period: str) -> pd.DataFrame:
        code = symbol.split(".")[0]
        indicator_map = {
            "pb": "市净率",
            "pe_ttm": "市盈率(TTM)",
        }
        indicator = indicator_map.get(metric)
        if indicator is None:
            raise DataSourceError(f"AkShare does not support valuation metric {metric}.")
        frame = self._call_with_cache(
            "valuation_history",
            lambda: self._invoke(
                "stock_zh_valuation_baidu",
                symbol=code,
                indicator=indicator,
                period="全部",
            ),
            code,
            metric,
            "all",
        )
        frame = normalize_frame(frame, {"code": normalize_symbol(code), "metric": metric})
        cutoff = pd.to_datetime(period, errors="coerce")
        if pd.notna(cutoff) and "date" in frame.columns:
            frame = frame[pd.to_datetime(frame["date"], errors="coerce") <= cutoff].copy()
        value_column = "value"
        if value_column not in frame.columns:
            candidate_columns = [column for column in frame.columns if column not in {"code", "date", "metric"}]
            if not candidate_columns:
                raise DataSourceError(f"Cannot detect valuation value column for {symbol} {metric}.")
            frame = frame.rename(columns={candidate_columns[-1]: value_column})
        frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
        frame = frame.dropna(subset=["date", value_column])
        if frame.empty:
            raise DataSourceError(f"AkShare valuation history empty for {symbol} {metric}.")
        return frame[["code", "date", "metric", "value"]]

    def get_industry_daily(self, start_date: str, end_date: str, level: str) -> pd.DataFrame:
        frame = self._call_with_cache(
            "industry_daily",
            lambda: self._invoke(
                "index_analysis_daily_sw",
                symbol=level,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            ),
            start_date,
            end_date,
            level,
        )
        frame = normalize_frame(frame)
        frame = frame.rename(
            columns={
                "指数代码": "industry_code",
                "指数名称": "industry_name",
                "发布日期": "date",
                "市盈率": "pe_ttm",
                "市净率": "pb",
            }
        )
        required = ["industry_code", "industry_name", "date"]
        missing = [item for item in required if item not in frame.columns]
        if missing:
            raise DataSourceError(f"Industry daily missing columns: {missing}")
        return frame

    def get_industry_members(self, industry_code: str, as_of_date: str | None = None) -> pd.DataFrame:
        def _load_primary():
            return self._invoke("sw_index_third_cons", symbol=industry_code)

        def _load_fallback():
            return self._invoke("index_component_sw", symbol=industry_code)

        try:
            frame = self._call_with_cache("industry_members", _load_primary, industry_code, as_of_date)
            if frame.empty:
                raise DataSourceError(f"AkShare primary industry_members empty for {industry_code}.")
        except DataSourceError:
            frame = self._call_with_cache("industry_members_fallback", _load_fallback, industry_code, as_of_date)
        frame = frame.rename(
            columns={
                "证券代码": "code",
                "成分券代码": "code",
                "股票代码": "code",
                "证券简称": "name",
                "股票简称": "name",
                "证券名称": "name",
            }
        )
        required = {"code", "name"}
        missing = required - set(frame.columns)
        if missing:
            raise DataSourceError(f"Industry members missing columns: {sorted(missing)}")
        frame["code"] = frame["code"].astype(str).map(normalize_symbol)
        frame["industry_code"] = industry_code
        frame["as_of_date"] = as_of_date
        return frame[["industry_code", "code", "name", "as_of_date"]]

    def get_financials(
        self, symbol: str, start_date: str | None = None, end_date: str | None = None
    ) -> pd.DataFrame:
        code = symbol.split(".")[0]
        frame = self._call_with_cache(
            "financial_abstract",
            lambda: self._invoke("stock_financial_abstract", symbol=code),
            code,
        )
        if frame.empty:
            raise DataSourceError(f"AkShare financial abstract empty for {symbol}.")

        metric_aliases = {
            "roe": ["净资产收益率(ROE)", "净资产收益率_平均", "摊薄净资产收益率"],
            "net_profit": ["归母净利润", "净利润"],
            "cfo": ["经营现金流量净额"],
            "debt_to_assets": ["资产负债率"],
        }
        value_columns = [column for column in frame.columns if str(column).isdigit()]
        if not value_columns:
            raise DataSourceError(f"AkShare financial abstract has no report-date columns for {symbol}.")

        normalized = frame.rename(columns={"指标": "metric_name"})[["metric_name", *value_columns]].copy()
        normalized["metric_name"] = normalized["metric_name"].astype(str)
        rows: list[pd.DataFrame] = []
        for target_field, aliases in metric_aliases.items():
            subset = normalized[normalized["metric_name"].isin(aliases)].copy()
            if subset.empty:
                continue
            selected = subset.iloc[0]
            melted = selected[value_columns].rename_axis("report_date").reset_index(name=target_field)
            rows.append(melted)
        if not rows:
            raise DataSourceError(f"AkShare financial abstract missing required metrics for {symbol}.")

        merged = rows[0]
        for extra in rows[1:]:
            merged = merged.merge(extra, how="outer", on="report_date")
        merged["report_date"] = pd.to_datetime(merged["report_date"], format="%Y%m%d", errors="coerce")
        merged = merged.dropna(subset=["report_date"]).copy()
        merged["announcement_date"] = pd.NaT
        merged["code"] = normalize_symbol(code)
        merged["date"] = merged["report_date"].dt.strftime("%Y-%m-%d")
        merged["report_date"] = merged["report_date"].dt.strftime("%Y-%m-%d")
        for column in ("roe", "net_profit", "cfo", "debt_to_assets"):
            if column not in merged.columns:
                merged[column] = pd.NA
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
        if start_date:
            merged = merged[merged["report_date"] >= start_date].copy()
        if end_date:
            merged = merged[merged["report_date"] <= end_date].copy()
        if merged.empty:
            raise DataSourceError(f"AkShare financial abstract filtered empty for {symbol}.")
        return merged[
            ["code", "date", "report_date", "announcement_date", "roe", "net_profit", "cfo", "debt_to_assets"]
        ].sort_values("report_date")

    def get_st_flags(self, symbols: list[str], as_of_date: str) -> pd.DataFrame:
        listing = self.get_stock_list(as_of_date)
        listing["is_st"] = listing["name"].astype(str).str.contains("ST", case=False, na=False)
        filtered = listing[listing["code"].isin([normalize_symbol(item) for item in symbols])].copy()
        filtered["date"] = as_of_date
        return filtered[["code", "date", "is_st"]]

    def get_market_caps(self, symbols: list[str], as_of_date: str) -> pd.DataFrame:
        raise DataSourceError("AkShare market cap retrieval is delegated to EfinanceAdapter for stability.")
