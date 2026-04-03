#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import resolve_path
from src.utils.ops import write_data_quality_report, write_provider_health_report, write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deterministic demo case and verify outputs.")
    parser.add_argument("--refresh-expected", action="store_true", help="Overwrite fixture expected outputs from the current demo run.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip comparing current outputs with fixture expected outputs.")
    return parser.parse_args()


def _load_json(path_like: str | Path) -> object:
    return json.loads(resolve_path(path_like).read_text(encoding="utf-8"))


def _copy_fixture_file(source: str | Path, target: str | Path) -> None:
    shutil.copyfile(resolve_path(source), resolve_path(target))


def _run_command(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=str(ROOT), check=True)


def _compare_json(expected_path: str | Path, actual_path: str | Path) -> None:
    expected = _load_json(expected_path)
    actual = _load_json(actual_path)
    if expected != actual:
        raise AssertionError(f"JSON mismatch: {expected_path} != {actual_path}")


def _compare_yaml(expected_path: str | Path, actual_path: str | Path) -> None:
    expected = yaml.safe_load(resolve_path(expected_path).read_text(encoding="utf-8"))
    actual = yaml.safe_load(resolve_path(actual_path).read_text(encoding="utf-8"))
    if expected != actual:
        raise AssertionError(f"YAML mismatch: {expected_path} != {actual_path}")


def main() -> int:
    args = parse_args()
    fixture_root = resolve_path("fixtures/demo_case")
    meta = _load_json(fixture_root / "meta.json")
    as_of_date = str(meta["as_of_date"])
    os.environ["QUANTILE_FIXED_GENERATED_AT"] = str(meta["generated_at"])

    data_dir = resolve_path("data/demo_case")
    data_dir.mkdir(parents=True, exist_ok=True)
    features_file = data_dir / "daily_features.parquet"
    benchmark_file = data_dir / "benchmark_daily.parquet"
    financials_file = data_dir / "financials_effective.parquet"
    pd.DataFrame(_load_json(fixture_root / "input/features.json")).to_parquet(features_file, index=False)
    pd.DataFrame(_load_json(fixture_root / "input/benchmark.json")).to_parquet(benchmark_file, index=False)
    pd.DataFrame(_load_json(fixture_root / "input/financials.json")).to_parquet(financials_file, index=False)

    _copy_fixture_file(fixture_root / "input/account.yml", "config/account.yml")
    _copy_fixture_file(fixture_root / "input/positions.yml", "config/positions.yml")
    _copy_fixture_file(fixture_root / "input/universe_seed.yml", "config/universe.yml")

    _run_command("scripts/refresh_universe.py", "--as-of-date", as_of_date, "--features-file", str(features_file), "--apply")
    _run_command(
        "scripts/prepare_snapshot.py",
        "--as-of-date",
        as_of_date,
        "--features-file",
        str(features_file),
        "--benchmark-file",
        str(benchmark_file),
        "--financials-file",
        str(financials_file),
    )
    _run_command("scripts/render_report.py", "--snapshot-json", "data/snapshots/latest.json")

    write_run_manifest(
        "data/curated/run_manifest.json",
        as_of_date=as_of_date,
        data_source=meta["data_source"],
        config_paths=[
            "config/account.yml",
            "config/positions.yml",
            "config/universe.yml",
            "config/strategy.yml",
            "config/metric_map.yml",
            "config/universe_rules.yml",
        ],
        input_paths=[
            fixture_root / "input/account.yml",
            fixture_root / "input/positions.yml",
            fixture_root / "input/universe_seed.yml",
            fixture_root / "input/features.json",
            fixture_root / "input/benchmark.json",
            fixture_root / "input/financials.json",
        ],
        extra={"mode": "demo_case"},
    )
    write_provider_health_report(
        "provider_health/latest.json",
        as_of_date=as_of_date,
        entries=[
            {
                "method": "demo_fixture",
                "adapter": meta["data_source"]["provider"],
                "success": True,
                "error": None,
                "attempt_index": 1,
                "rows": 4,
            }
        ],
        source="demo_fixture",
    )
    write_data_quality_report(
        "data_quality/latest.json",
        [features_file, benchmark_file, financials_file, "reports/daily/orders_latest.json"],
        as_of_date,
    )

    expected_dir = fixture_root / "expected"
    expected_dir.mkdir(parents=True, exist_ok=True)
    if args.refresh_expected:
        shutil.copyfile(resolve_path("config/universe.yml"), expected_dir / "universe.yml")
        shutil.copyfile(resolve_path("reports/daily/orders_latest.json"), expected_dir / "orders_latest.json")

    if not args.skip_verify:
        _compare_yaml(expected_dir / "universe.yml", "config/universe.yml")
        _compare_json(expected_dir / "orders_latest.json", "reports/daily/orders_latest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
