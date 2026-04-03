from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.utils.config import resolve_path


def clear_directories(paths: list[str | Path]) -> list[Path]:
    cleared: list[Path] = []
    for path_like in paths:
        path = resolve_path(path_like)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        cleared.append(path)
    return cleared


def configured_cache_directories(data_cfg: dict) -> list[str]:
    cache_cfg = data_cfg.get("cache", {})
    directories = [str(item) for item in cache_cfg.get("directories", []) if str(item).strip()]
    if directories:
        return directories
    raw_dir = str(data_cfg.get("storage", {}).get("raw_dir", "data/raw"))
    return [str(Path(raw_dir) / "akshare")]


def write_feature_dataset(frame: pd.DataFrame, output_path: str | Path) -> list[Path]:
    path = resolve_path(output_path)
    if path.suffix == ".parquet":
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return [path]

    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    if frame.empty:
        empty_path = path / "empty.parquet"
        frame.to_parquet(empty_path, index=False)
        return [empty_path]

    years = pd.to_datetime(frame["date"], errors="coerce").dt.year.fillna(0).astype(int)
    written: list[Path] = []
    for year, group in frame.assign(_feature_year=years).groupby("_feature_year", sort=True):
        filename = "unknown.parquet" if int(year) <= 0 else f"{int(year)}.parquet"
        target = path / filename
        group.drop(columns=["_feature_year"]).to_parquet(target, index=False)
        written.append(target)
    return written
