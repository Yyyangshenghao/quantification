from __future__ import annotations

import pandas as pd
import pytest

from src.adapters.akshare_adapter import AkshareAdapter
from src.adapters.baostock_adapter import BaoStockAdapter
from src.adapters.common import (
    is_a_share_equity_symbol,
    normalize_symbol,
    to_baostock_symbol,
    to_jq_symbol,
)
from src.adapters.efinance_adapter import EfinanceAdapter


def test_symbol_normalization_handles_vendor_formats() -> None:
    assert normalize_symbol("600519.SH") == "600519.sh"
    assert normalize_symbol("sh.600519") == "600519.sh"
    assert normalize_symbol("600519.XSHG") == "600519.sh"
    assert normalize_symbol("000001.XSHE") == "000001.sz"
    assert normalize_symbol("830799.BJSE") == "830799.bj"
    assert is_a_share_equity_symbol("600519.XSHG") is True
    assert is_a_share_equity_symbol("sh.000001") is False
    assert to_baostock_symbol("600519.XSHG") == "sh.600519"
    assert to_jq_symbol("sh.600519") == "600519.XSHG"


def test_akshare_industry_daily_maps_publish_date(configs: dict, monkeypatch) -> None:
    adapter = AkshareAdapter(configs["data_sources"])

    monkeypatch.setattr(
        adapter,
        "_call_with_cache",
        lambda namespace, loader, *cache_parts: pd.DataFrame(
            [{"指数代码": "851312", "指数名称": "棉纺", "发布日期": "2024-12-02", "市盈率": 18.06, "市净率": 0.94}]
        ),
    )

    frame = adapter.get_industry_daily("2024-12-01", "2024-12-31", "三级行业")

    assert frame.loc[0, "industry_code"] == "851312"
    assert frame.loc[0, "industry_name"] == "棉纺"
    assert frame.loc[0, "date"] == "2024-12-02"
    assert frame.loc[0, "pe_ttm"] == 18.06
    assert frame.loc[0, "pb"] == 0.94


def test_akshare_stock_list_includes_listed_date_from_exchange_tables(configs: dict, monkeypatch) -> None:
    adapter = AkshareAdapter(configs["data_sources"])

    def fake_call(namespace: str, loader, *cache_parts) -> pd.DataFrame:
        if namespace == "stock_list_sh_main":
            return pd.DataFrame([{"证券代码": "600000", "证券简称": "浦发银行", "上市日期": "1999-11-10"}])
        if namespace == "stock_list_sh_star":
            return pd.DataFrame(columns=["证券代码", "证券简称", "上市日期"])
        if namespace == "stock_list_sz_a":
            return pd.DataFrame([{"A股代码": "000001", "A股简称": "平安银行", "A股上市日期": "1991-04-03"}])
        raise AssertionError(f"Unexpected namespace: {namespace}")

    monkeypatch.setattr(adapter, "_call_with_cache", fake_call)

    frame = adapter.get_stock_list("2026-04-03")

    assert frame.to_dict(orient="records") == [
        {"code": "000001.sz", "name": "平安银行", "listed_date": "1991-04-03", "as_of_date": "2026-04-03"},
        {"code": "600000.sh", "name": "浦发银行", "listed_date": "1999-11-10", "as_of_date": "2026-04-03"},
    ]


def test_akshare_industry_members_falls_back_on_empty_primary(configs: dict, monkeypatch) -> None:
    adapter = AkshareAdapter(configs["data_sources"])

    def fake_call(namespace: str, loader, *cache_parts) -> pd.DataFrame:
        if namespace == "industry_members":
            return pd.DataFrame(columns=["股票代码", "股票简称"])
        if namespace == "industry_members_fallback":
            return pd.DataFrame([{"证券代码": "000596", "证券名称": "古井贡酒"}])
        raise AssertionError(f"Unexpected namespace: {namespace}")

    monkeypatch.setattr(adapter, "_call_with_cache", fake_call)

    frame = adapter.get_industry_members("801120", as_of_date="2024-12-31")

    assert frame.to_dict(orient="records") == [
        {
            "industry_code": "801120",
            "code": "000596.sz",
            "name": "古井贡酒",
            "as_of_date": "2024-12-31",
        }
    ]


def test_baostock_valuation_history_returns_metric_column(configs: dict, monkeypatch) -> None:
    adapter = BaoStockAdapter(configs["data_sources"])

    class FakeBS:
        @staticmethod
        def query_history_k_data_plus(symbol, fields, start_date, end_date, frequency, adjustflag):
            assert symbol == "sh.600519"
            assert fields == "date,code,pbMRQ"
            assert start_date == "2016-01-01"
            assert end_date == "2024-12-31"
            assert frequency == "d"
            assert adjustflag == "3"
            return object()

    monkeypatch.setattr(adapter, "_bs", lambda: FakeBS())
    monkeypatch.setattr(
        adapter,
        "_result_to_frame",
        lambda result: pd.DataFrame([{"date": "2024-12-02", "code": "sh.600519", "pbMRQ": "8.059005"}]),
    )

    frame = adapter.get_stock_valuation_history("600519.sh", "pb", "2024-12-31")

    assert frame.to_dict(orient="records") == [
        {
            "code": "600519.sh",
            "date": "2024-12-02",
            "metric": "pb",
            "value": 8.059005,
        }
    ]


def test_baostock_financials_uses_keyword_queries_and_computes_cfo(configs: dict, monkeypatch) -> None:
    adapter = BaoStockAdapter(configs["data_sources"])

    class FakeBS:
        @staticmethod
        def query_profit_data(*, code, year, quarter):
            assert code == "sh.600519"
            assert year == 2024
            if quarter != 3:
                return pd.DataFrame(columns=["code", "pubDate", "statDate", "roeAvg", "netProfit"])
            return pd.DataFrame(
                [
                    {
                        "code": "sh.600519",
                        "pubDate": "2024-10-26",
                        "statDate": "2024-09-30",
                        "roeAvg": "0.26833",
                        "netProfit": "63031462239.55",
                    }
                ]
            )

        @staticmethod
        def query_cash_flow_data(*, code, year, quarter):
            if quarter != 3:
                return pd.DataFrame(columns=["code", "pubDate", "statDate", "CFOToNP"])
            return pd.DataFrame(
                [{"code": "sh.600519", "pubDate": "2024-10-26", "statDate": "2024-09-30", "CFOToNP": "0.704749"}]
            )

        @staticmethod
        def query_dupont_data(*, code, year, quarter):
            if quarter != 3:
                return pd.DataFrame(columns=["code", "pubDate", "statDate"])
            return pd.DataFrame([{"code": "sh.600519", "pubDate": "2024-10-26", "statDate": "2024-09-30"}])

        @staticmethod
        def query_balance_data(*, code, year, quarter):
            if quarter != 3:
                return pd.DataFrame(columns=["code", "pubDate", "statDate", "liabilityToAsset"])
            return pd.DataFrame(
                [{"code": "sh.600519", "pubDate": "2024-10-26", "statDate": "2024-09-30", "liabilityToAsset": "0.001363"}]
            )

    monkeypatch.setattr(adapter, "_bs", lambda: FakeBS())
    monkeypatch.setattr(adapter, "_result_to_frame", lambda result: result)

    frame = adapter.get_financials("600519.sh", start_date="2024-01-01", end_date="2024-12-31")

    row = frame.to_dict(orient="records")[0]
    assert row["code"] == "600519.sh"
    assert row["date"] == "2024-09-30"
    assert row["report_date"] == "2024-09-30"
    assert row["announcement_date"] == "2024-10-26"
    assert row["roe"] == pytest.approx(0.26833)
    assert row["net_profit"] == pytest.approx(63031462239.55)
    assert row["cfo"] == pytest.approx(44421359981.86063)
    assert row["debt_to_assets"] == pytest.approx(0.001363)
    assert row["is_st"] is False


def test_efinance_base_info_supports_market_caps_and_st_flags(configs: dict, monkeypatch) -> None:
    adapter = EfinanceAdapter(configs["data_sources"])

    class FakeStock:
        @staticmethod
        def get_base_info(codes):
            assert codes == ["600519", "000752"]
            return pd.DataFrame(
                [
                    {"股票代码": "600519", "股票名称": "贵州茅台", "总市值": 1_828_164_241_474.2},
                    {"股票代码": "000752", "股票名称": "ST西发", "总市值": 2_600_000_000.0},
                ]
            )

    class FakeEf:
        stock = FakeStock()

    monkeypatch.setattr(adapter, "_ef", lambda: FakeEf())

    market_caps = adapter.get_market_caps(["600519.sh", "000752.sz"], "2024-12-31")
    st_flags = adapter.get_st_flags(["600519.sh", "000752.sz"], "2024-12-31")

    assert market_caps.to_dict(orient="records") == [
        {"code": "600519.sh", "date": "2024-12-31", "market_cap_billion": 18281.642414742},
        {"code": "000752.sz", "date": "2024-12-31", "market_cap_billion": 26.0},
    ]
    assert st_flags.to_dict(orient="records") == [
        {"code": "600519.sh", "date": "2024-12-31", "is_st": False},
        {"code": "000752.sz", "date": "2024-12-31", "is_st": True},
    ]
