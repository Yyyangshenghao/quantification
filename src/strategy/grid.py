from __future__ import annotations


def compute_grid_step(atr20: float, close: float, grid_cfg: dict) -> float:
    raw = float(grid_cfg["atr_multiplier"]) * float(atr20) / float(close)
    return max(float(grid_cfg["min_step"]), min(float(grid_cfg["max_step"]), raw))


def can_add_grid_tranche(row: dict, position: dict, grid_cfg: dict) -> bool:
    if int(position.get("remaining_tranches", 0)) <= 0:
        return False
    if not bool(row.get("thesis_still_valid", False)):
        return False
    if float(row.get("close", 0)) / float(row.get("ma250", 1)) < float(grid_cfg["hard_add_ban"]["close_to_ma250_min"]):
        return False
    if float(row.get("ma120_slope_20d", 0)) < float(grid_cfg["hard_add_ban"]["ma120_slope_20d_min"]):
        return False
    grid_step = compute_grid_step(float(row["atr20"]), float(row["close"]), grid_cfg)
    return float(row["close"]) <= float(position["last_fill_price"]) * (1.0 - grid_step)


def can_reduce_grid_tranche(row: dict, position: dict, grid_cfg: dict) -> bool:
    if int(position.get("extra_tranches", 0)) <= 0:
        return False
    grid_step = compute_grid_step(float(row["atr20"]), float(row["close"]), grid_cfg)
    return bool(
        float(row["close"]) >= float(position["last_fill_price"]) * (1.0 + grid_step)
        and float(row["stock_q_blended"]) >= float(grid_cfg["reduce_extra_min_stock_q_blended"])
    )
