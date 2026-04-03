from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.storage import clear_directories, configured_cache_directories, write_feature_dataset


def test_configured_cache_directories_defaults_to_akshare_raw_cache() -> None:
    data_cfg = {"storage": {"raw_dir": "data/raw"}, "cache": {}}
    assert configured_cache_directories(data_cfg) == ["data/raw/akshare"]


def test_clear_directories_removes_cache_contents_but_not_sibling_main_tables(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    cache_dir = raw_dir / "akshare"
    cache_dir.mkdir(parents=True)
    (cache_dir / "stale.parquet").write_text("cache", encoding="utf-8")
    main_table = raw_dir / "price_daily.parquet"
    main_table.parent.mkdir(parents=True, exist_ok=True)
    main_table.write_text("main", encoding="utf-8")

    clear_directories([cache_dir])

    assert cache_dir.exists()
    assert list(cache_dir.iterdir()) == []
    assert main_table.read_text(encoding="utf-8") == "main"


def test_write_feature_dataset_writes_yearly_partitioned_directory(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["600000.sh", "600000.sh", "601088.sh"],
            "date": ["2025-12-31", "2026-01-02", "2026-04-03"],
            "close": [10.0, 10.2, 38.5],
        }
    )
    output_dir = tmp_path / "daily_features"

    written = write_feature_dataset(frame, output_dir)

    assert sorted(path.name for path in written) == ["2025.parquet", "2026.parquet"]
    loaded = pd.read_parquet(output_dir).sort_values(["date", "symbol"]).reset_index(drop=True)
    expected = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(loaded, expected)


def test_write_feature_dataset_supports_single_parquet_file(tmp_path: Path) -> None:
    frame = pd.DataFrame({"symbol": ["600000.sh"], "date": ["2026-04-03"], "close": [10.0]})
    output_file = tmp_path / "daily_features.parquet"

    written = write_feature_dataset(frame, output_file)

    assert written == [output_file]
    loaded = pd.read_parquet(output_file)
    pd.testing.assert_frame_equal(loaded, frame)
