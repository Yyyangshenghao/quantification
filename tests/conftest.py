from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_project_configs, reset_config_cache  # noqa: E402


@pytest.fixture()
def configs() -> dict:
    reset_config_cache()
    return copy.deepcopy(load_project_configs())
