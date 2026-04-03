from __future__ import annotations

from src.strategy.grid import compute_grid_step


def test_grid_step_is_clipped(configs: dict) -> None:
    grid_cfg = configs["strategy"]["execution"]["grid_execution"]
    assert compute_grid_step(atr20=1.0, close=10.0, grid_cfg=grid_cfg) == 0.08
    assert compute_grid_step(atr20=0.1, close=10.0, grid_cfg=grid_cfg) == 0.04
