from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import resolve_path
from src.utils.exceptions import DataSourceError


class ParquetStore:
    def __init__(self, duckdb_path: str | Path, compression: str = "zstd") -> None:
        self.duckdb_path = resolve_path(duckdb_path)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        self.compression = compression

    def write_frame(self, path_like: str | Path, frame: pd.DataFrame) -> Path:
        path = resolve_path(path_like)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression=self.compression)
        return path

    def read_frame(self, path_like: str | Path) -> pd.DataFrame:
        path = resolve_path(path_like)
        if not path.exists():
            raise DataSourceError(f"Parquet file not found: {path}")
        return pd.read_parquet(path)

    def query(self, sql: str) -> pd.DataFrame:
        try:
            import duckdb
        except ImportError as exc:
            raise DataSourceError("duckdb is required for SQL querying.") from exc
        with duckdb.connect(str(self.duckdb_path)) as connection:
            return connection.sql(sql).df()

    def write_sql_view(self, view_name: str, parquet_path: str | Path) -> None:
        try:
            import duckdb
        except ImportError as exc:
            raise DataSourceError("duckdb is required for SQL querying.") from exc
        path = resolve_path(parquet_path)
        with duckdb.connect(str(self.duckdb_path)) as connection:
            connection.execute(
                f"create or replace view {view_name} as select * from read_parquet('{path}')"
            )
