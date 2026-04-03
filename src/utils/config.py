from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.utils.exceptions import ConfigError


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return repo_root() / path


@lru_cache(maxsize=32)
def load_yaml(path_like: str | Path) -> dict[str, Any]:
    path = resolve_path(path_like)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a mapping: {path}")
    return data


def load_yaml_optional(path_like: str | Path) -> dict[str, Any]:
    path = resolve_path(path_like)
    if not path.exists():
        return {}
    return load_yaml(path)


def reset_config_cache() -> None:
    load_yaml.cache_clear()


def load_project_configs() -> dict[str, dict[str, Any]]:
    return {
        "data_sources": load_yaml("config/data_sources.yml"),
        "strategy": load_yaml("config/strategy.yml"),
        "metric_map": load_yaml("config/metric_map.yml"),
        "positions": load_yaml("config/positions.yml"),
        "universe": load_yaml("config/universe.yml"),
        "universe_rules": load_yaml("config/universe_rules.yml"),
        "account": load_yaml_optional("config/account.yml"),
    }
