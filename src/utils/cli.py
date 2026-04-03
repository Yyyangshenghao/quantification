from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import resolve_path


def add_common_date_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="End date in YYYY-MM-DD format.")
    parser.add_argument("--as-of-date", help="As-of date in YYYY-MM-DD format.")


def ensure_dir(path_like: str | Path) -> Path:
    path = resolve_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_json(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _sanitize_json(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_sanitize_json(item) for item in payload]
    if isinstance(payload, tuple):
        return [_sanitize_json(item) for item in payload]
    if payload is None:
        return None
    if isinstance(payload, (str, int, bool)):
        return payload
    try:
        if pd.isna(payload):
            return None
    except Exception:
        pass
    if hasattr(payload, "item"):
        try:
            return _sanitize_json(payload.item())
        except Exception:
            return payload
    if isinstance(payload, float) and (payload != payload or payload in {float("inf"), float("-inf")}):
        return None
    return payload


def write_json(path_like: str | Path, payload: Any) -> Path:
    path = resolve_path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_sanitize_json(payload), handle, ensure_ascii=False, indent=2)
    return path
