from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.utils.config import resolve_path


LEDGER_COLUMNS = [
    "trade_date",
    "symbol",
    "name",
    "side",
    "shares",
    "price",
    "commission",
    "stamp_duty",
    "other_fees",
    "source",
    "notes",
]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(numeric):
        return default
    return numeric


def _infer_tranches(current_weight: float, tranche_weights: dict[int, float]) -> int:
    if current_weight <= 0:
        return 0
    ordered = sorted(tranche_weights.items(), key=lambda item: (abs(item[1] - current_weight), item[0]))
    return int(ordered[0][0]) if ordered else 0


def normalize_fills(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    rename_map = {
        "date": "trade_date",
        "成交日期": "trade_date",
        "代码": "symbol",
        "证券代码": "symbol",
        "名称": "name",
        "证券名称": "name",
        "方向": "side",
        "买卖方向": "side",
        "数量": "shares",
        "成交数量": "shares",
        "价格": "price",
        "成交价格": "price",
        "手续费": "commission",
        "印花税": "stamp_duty",
    }
    normalized = normalized.rename(columns={key: value for key, value in rename_map.items() if key in normalized.columns})
    for column in LEDGER_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = "" if column in {"trade_date", "symbol", "name", "side", "source", "notes"} else 0.0
    normalized["trade_date"] = pd.to_datetime(normalized["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    normalized["symbol"] = normalized["symbol"].astype(str).str.strip().str.lower()
    normalized["name"] = normalized["name"].astype(str).str.strip()
    normalized["side"] = (
        normalized["side"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"B": "BUY", "S": "SELL", "买入": "BUY", "卖出": "SELL"})
    )
    for column in ("shares", "price", "commission", "stamp_duty", "other_fees"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0.0)
    normalized["shares"] = normalized["shares"].astype(int)
    normalized["source"] = normalized["source"].astype(str).str.strip().replace({"": "manual_import"})
    normalized["notes"] = normalized["notes"].astype(str).str.strip()
    normalized = normalized[LEDGER_COLUMNS]
    normalized = normalized[normalized["trade_date"].notna()]
    normalized = normalized[normalized["symbol"] != ""]
    normalized = normalized[normalized["side"].isin(["BUY", "SELL"])]
    normalized = normalized[normalized["shares"] > 0]
    normalized = normalized.sort_values(["trade_date", "symbol", "side", "price", "shares"]).reset_index(drop=True)
    return normalized


def append_fills_to_ledger(fills: pd.DataFrame, ledger_path: str | Path) -> pd.DataFrame:
    ledger_file = resolve_path(ledger_path)
    ledger_file.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_fills(fills)
    if ledger_file.exists():
        existing = pd.read_csv(ledger_file)
        existing = normalize_fills(existing)
        normalized = pd.concat([existing, normalized], ignore_index=True)
    normalized = normalized.drop_duplicates(
        subset=["trade_date", "symbol", "side", "shares", "price", "commission", "stamp_duty", "other_fees"],
        keep="last",
    ).sort_values(["trade_date", "symbol", "side", "price", "shares"])
    normalized.to_csv(ledger_file, index=False)
    return normalized.reset_index(drop=True)


def rebuild_positions_from_ledger(
    trades: pd.DataFrame,
    *,
    latest_prices: pd.DataFrame | None = None,
    latest_total_equity: float = 0.0,
    tranche_weights: dict[int, float] | None = None,
) -> list[dict]:
    if trades.empty:
        return []
    tranche_weights = tranche_weights or {0: 0.0}
    latest_prices = latest_prices.copy() if latest_prices is not None else pd.DataFrame(columns=["code", "close"])
    if not latest_prices.empty:
        latest_prices = latest_prices.sort_values("date" if "date" in latest_prices.columns else latest_prices.columns[0]).copy()
        latest_prices = latest_prices.drop_duplicates(subset=["code"], keep="last").set_index("code")
    positions: list[dict] = []
    for symbol, group in normalize_fills(trades).groupby("symbol", sort=True):
        shares = 0
        avg_cost = 0.0
        last_fill_price = 0.0
        name = str(group["name"].iloc[-1] or symbol)
        for row in group.sort_values(["trade_date", "symbol"]).to_dict(orient="records"):
            side = row["side"]
            trade_shares = int(row["shares"])
            trade_price = float(row["price"])
            fees = float(row.get("commission", 0.0)) + float(row.get("other_fees", 0.0))
            if side == "BUY":
                total_cost = avg_cost * shares + trade_shares * trade_price + fees
                shares += trade_shares
                avg_cost = total_cost / shares if shares else 0.0
            else:
                shares = max(0, shares - trade_shares)
                if shares == 0:
                    avg_cost = 0.0
            last_fill_price = trade_price
        if shares <= 0:
            continue
        latest_close = float(latest_prices.loc[symbol, "close"]) if not latest_prices.empty and symbol in latest_prices.index else 0.0
        current_weight = round((shares * latest_close) / latest_total_equity, 6) if latest_total_equity > 0 and latest_close > 0 else 0.0
        positions.append(
            {
                "symbol": symbol,
                "name": name,
                "current_shares": int(shares),
                "avg_cost": round(avg_cost, 4),
                "current_weight": current_weight,
                "current_position_tranches": _infer_tranches(current_weight, tranche_weights),
                "extra_tranches": 0,
                "last_fill_price": round(last_fill_price, 4),
            }
        )
    return positions


def write_positions_yaml(
    output_path: str | Path,
    *,
    positions: list[dict],
    default_tranches: int,
    latest_total_equity: float,
) -> Path:
    payload = {
        "portfolio": {
            "manual_only": True,
            "default_tranches": int(default_tranches),
            "cash_weight": round(max(0.0, 1.0 - sum(_safe_float(item.get("current_weight")) for item in positions)), 6)
            if latest_total_equity > 0
            else 1.0,
            "notes": "建议由 data/ledger/trades.csv 重建当前持仓；程序只生成手动执行建议，不自动下单。",
        },
        "positions": positions,
    }
    output_file = resolve_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return output_file
