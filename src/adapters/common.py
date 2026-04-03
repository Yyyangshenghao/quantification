from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.utils.config import resolve_path
from src.utils.exceptions import DataSourceError


MARKET_ALIAS_MAP = {
    "sh": "sh",
    "xshg": "sh",
    "sz": "sz",
    "xshe": "sz",
    "bj": "bj",
    "bjse": "bj",
}


PRICE_COLUMN_MAP = {
    "日期": "date",
    "date": "date",
    "代码": "code",
    "symbol": "code",
    "开盘": "open",
    "open": "open",
    "最高": "high",
    "high": "high",
    "最低": "low",
    "low": "low",
    "收盘": "close",
    "close": "close",
    "成交量": "volume",
    "volume": "volume",
    "成交额": "amount",
    "amount": "amount",
    "换手率": "turn",
    "turn": "turn",
    "是否ST": "is_st",
}


def retry_call(fn: Callable[[], pd.DataFrame], attempts: int, backoff_seconds: float) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = fn()
            if not isinstance(result, pd.DataFrame):
                raise DataSourceError("Adapter must return pandas DataFrame.")
            return result
        except Exception as exc:  # pragma: no cover - exercised against live sources
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(backoff_seconds * attempt)
    raise DataSourceError(str(last_error)) from last_error


def normalize_frame(frame: pd.DataFrame, extra_columns: dict[str, Any] | None = None) -> pd.DataFrame:
    renamed = frame.rename(columns=PRICE_COLUMN_MAP)
    if extra_columns:
        for key, value in extra_columns.items():
            renamed[key] = value
    if "date" in renamed.columns:
        renamed["date"] = pd.to_datetime(renamed["date"]).dt.strftime("%Y-%m-%d")
    return renamed


def normalize_symbol(symbol: str) -> str:
    value = str(symbol).strip()
    if not value:
        return value
    lowered = value.lower()
    if "." in lowered:
        left, right = lowered.split(".", 1)
        if left in MARKET_ALIAS_MAP:
            return f"{right}.{MARKET_ALIAS_MAP[left]}"
        if right in MARKET_ALIAS_MAP:
            return f"{left}.{MARKET_ALIAS_MAP[right]}"
        return lowered
    if lowered.startswith("6"):
        return f"{lowered}.sh"
    if lowered.startswith(("0", "3")):
        return f"{lowered}.sz"
    if lowered.startswith(("4", "8", "9")):
        return f"{lowered}.bj"
    return lowered


def is_a_share_equity_symbol(symbol: str) -> bool:
    normalized = normalize_symbol(symbol)
    if "." not in normalized:
        return False
    code, market = normalized.split(".", 1)
    if not code.isdigit() or len(code) != 6:
        return False
    if market == "sh":
        return code.startswith("6")
    if market == "sz":
        return code.startswith(("0", "3"))
    if market == "bj":
        return code.startswith(("4", "8", "9"))
    return False


def to_baostock_symbol(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    if "." not in normalized:
        raise DataSourceError(f"Cannot convert symbol to BaoStock format: {symbol}")
    code, market = normalized.split(".", 1)
    return f"{market}.{code}"


def to_jq_symbol(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    if "." not in normalized:
        raise DataSourceError(f"Cannot convert symbol to JQData format: {symbol}")
    code, market = normalized.split(".", 1)
    mapping = {"sh": "XSHG", "sz": "XSHE", "bj": "BJSE"}
    return f"{code}.{mapping[market]}"


def cache_file(base_dir: str | Path, namespace: str, *parts: object) -> Path:
    digest = hashlib.md5("|".join(map(str, parts)).encode("utf-8"), usedforsecurity=False).hexdigest()
    path = resolve_path(base_dir) / namespace / f"{digest}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_optional_module(module_name: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise DataSourceError(f"Optional dependency not installed: {module_name}") from exc
