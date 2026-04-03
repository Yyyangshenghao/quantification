from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd


DATE_FMT = "%Y-%m-%d"


def normalize_date(value: str | date | datetime | pd.Timestamp) -> str:
    if isinstance(value, str):
        return pd.Timestamp(value).strftime(DATE_FMT)
    if isinstance(value, pd.Timestamp):
        return value.strftime(DATE_FMT)
    if isinstance(value, datetime):
        return value.date().strftime(DATE_FMT)
    if isinstance(value, date):
        return value.strftime(DATE_FMT)
    raise TypeError(f"Unsupported date value: {value!r}")


def ensure_timestamp(value: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(normalize_date(value))


def date_range(start_date: str, end_date: str) -> Iterable[str]:
    start = ensure_timestamp(start_date)
    end = ensure_timestamp(end_date)
    current = start
    while current <= end:
        yield current.strftime(DATE_FMT)
        current += timedelta(days=1)
