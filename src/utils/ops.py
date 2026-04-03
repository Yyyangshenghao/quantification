from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.cli import write_json
from src.utils.config import resolve_path


DATE_COLUMNS = ("date", "effective_from", "report_date", "announcement_date")


def sha256_file(path_like: str | Path) -> str:
    path = resolve_path(path_like)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_hash(cwd: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(resolve_path(cwd)) if cwd else None,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def write_run_manifest(
    output_path: str | Path,
    *,
    as_of_date: str,
    data_source: dict[str, Any],
    config_paths: list[str | Path],
    input_paths: list[str | Path],
    extra: dict[str, Any] | None = None,
) -> Path:
    manifest = {
        "as_of_date": as_of_date,
        "git_commit": git_commit_hash(),
        "data_source": data_source,
        "config_hashes": {str(path): sha256_file(path) for path in config_paths},
        "input_hashes": {str(path): sha256_file(path) for path in input_paths},
    }
    if extra:
        manifest.update(extra)
    return write_json(output_path, manifest)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
    raise ValueError(f"Unsupported table format for quality report: {path}")


def table_quality(path_like: str | Path, as_of_date: str) -> dict[str, Any]:
    path = resolve_path(path_like)
    if not path.exists():
        return {
            "path": str(path_like),
            "exists": False,
            "rows": 0,
            "missing_rate": None,
            "date_min": None,
            "date_max": None,
            "stale": True,
        }
    frame = _read_table(path)
    date_column = next((column for column in DATE_COLUMNS if column in frame.columns), None)
    if date_column:
        dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
        date_min = dates.min().strftime("%Y-%m-%d") if not dates.empty else None
        date_max = dates.max().strftime("%Y-%m-%d") if not dates.empty else None
        stale = bool(date_max and date_max < as_of_date)
    else:
        date_min = None
        date_max = None
        stale = False
    denominator = max(frame.shape[0] * max(frame.shape[1], 1), 1)
    missing_rate = round(float(frame.isna().sum().sum()) / float(denominator), 6)
    return {
        "path": str(path_like),
        "exists": True,
        "rows": int(len(frame)),
        "missing_rate": missing_rate,
        "date_min": date_min,
        "date_max": date_max,
        "stale": stale,
    }


def write_data_quality_report(output_path: str | Path, table_paths: list[str | Path], as_of_date: str) -> Path:
    payload = {
        "as_of_date": as_of_date,
        "tables": [table_quality(path, as_of_date) for path in table_paths],
    }
    return write_json(output_path, payload)


def write_provider_health_report(
    output_path: str | Path,
    *,
    as_of_date: str,
    entries: list[dict[str, Any]],
    source: str,
) -> Path:
    payload = {
        "as_of_date": as_of_date,
        "source": source,
        "providers": entries,
    }
    return write_json(output_path, payload)
