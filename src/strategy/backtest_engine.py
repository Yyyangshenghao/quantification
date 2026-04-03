from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import pandas as pd

from src.strategy.regime import determine_market_regime
from src.strategy.signals import SignalEngine
from src.utils.config import resolve_path


@dataclass
class Position:
    symbol: str
    industry: str
    bucket: str
    shares: float = 0.0
    avg_cost: float = 0.0
    current_position_tranches: int = 0
    current_weight: float = 0.0
    extra_tranches: int = 0
    last_fill_price: float = 0.0
    current_shares: int = 0


@dataclass
class BacktestResult:
    metrics: dict
    nav: pd.DataFrame
    trade_list: pd.DataFrame
    stock_attribution: pd.DataFrame
    industry_attribution: pd.DataFrame
    approximate_backtest: bool = False


class BacktestEngine:
    def __init__(
        self,
        strategy_cfg: dict,
        universe_rules_cfg: dict | None = None,
        account_cfg: dict | None = None,
        initial_cash: float | None = None,
        historical_universe_dir: str | Path | None = "data/curated/universe_history",
    ) -> None:
        self.strategy_cfg = strategy_cfg
        self.universe_rules_cfg = universe_rules_cfg or {}
        self.account_cfg = account_cfg or {}
        account_initial = float(self.account_cfg.get("account", {}).get("initial_capital", 1_000_000.0))
        self.initial_cash = float(initial_cash if initial_cash is not None else account_initial)
        self.signal_engine = SignalEngine(strategy_cfg, self.universe_rules_cfg, self.account_cfg)
        self.historical_universe_dir = resolve_path(historical_universe_dir) if historical_universe_dir else None
        execution_cfg = self.account_cfg.get("execution", {}) if isinstance(self.account_cfg, dict) else {}
        self.fee_rate = float(execution_cfg.get("commission_rate", self.strategy_cfg["execution"].get("fee_rate", 0.0)))
        self.tax_rate = float(execution_cfg.get("stamp_duty_rate_sell", self.strategy_cfg["execution"].get("stamp_tax_rate", 0.0)))
        slippage_bps = execution_cfg.get("slippage_bps")
        if slippage_bps is None:
            self.slippage_rate = float(self.strategy_cfg["execution"].get("slippage_rate", 0.0))
        else:
            self.slippage_rate = float(slippage_bps) / 10_000.0

    def run(self, features: pd.DataFrame, benchmark: pd.DataFrame, bucket: str = "combined") -> BacktestResult:
        features = features.sort_values(["date", "symbol"]).copy()
        features = self._prepare_backtest_features(features)
        features, approximate_backtest = self._overlay_historical_universe(features)
        benchmark = benchmark.sort_values("date").copy()
        dates = sorted(features["date"].unique())
        if len(dates) < 2:
            raise ValueError("Backtest requires at least two trading days.")

        cash = self.initial_cash
        positions: dict[str, Position] = {}
        nav_records: list[dict] = []
        trades: list[dict] = []

        for index, signal_date in enumerate(dates[:-1]):
            fill_date = dates[index + 1]
            todays = features[features["date"] == signal_date].copy()
            if bucket != "combined":
                todays = todays[todays["bucket"] == bucket].copy()
            benchmark_until_today = benchmark[benchmark["date"] <= signal_date]
            regime = determine_market_regime(benchmark_until_today, self.strategy_cfg)
            positions_df = self._positions_frame(positions)
            todays = self._apply_positions_to_features(todays, positions_df)
            nav = self._portfolio_value(cash, positions, todays.set_index("symbol"), mark_field="close")
            account_state = {
                "orders_degraded": False,
                "current_cash": cash,
                "reserved_cash": 0.0,
                "latest_total_equity": nav,
                "current_invested_value": max(0.0, nav - cash),
            }
            decisions = self.signal_engine.generate(todays, positions_df, regime, safe_mode=False, account_state=account_state)
            next_day = features[features["date"] == fill_date].set_index("symbol")
            for decision in decisions:
                action = decision["action_enum"]
                if action not in {"BUY_1", "BUY_2", "BUY_3", "REDUCE", "SELL_ALL"}:
                    continue
                if decision["symbol"] not in next_day.index:
                    continue
                if "open" in next_day.columns and pd.notna(next_day.loc[decision["symbol"], "open"]):
                    fill_open = float(next_day.loc[decision["symbol"], "open"])
                else:
                    fill_open = float(next_day.loc[decision["symbol"], "close"])
                fill_price = fill_open * (
                    1 + self.slippage_rate
                    if action in {"BUY_1", "BUY_2", "BUY_3"}
                    else 1 - self.slippage_rate
                )
                trade_value = abs(float(decision.get("target_order_value", 0.0)))
                if trade_value <= 0:
                    trade_value = abs(float(decision["target_position_change"])) * nav
                realized_pnl = 0.0
                requested_shares = abs(int(decision.get("delta_shares", 0) or 0))

                if action in {"BUY_1", "BUY_2", "BUY_3"}:
                    shares = float(requested_shares) if requested_shares > 0 else (trade_value / fill_price if fill_price else 0.0)
                    trade_value = shares * fill_price
                    fee = trade_value * self.fee_rate
                    cash -= trade_value + fee
                    position = positions.get(
                        decision["symbol"],
                        Position(symbol=decision["symbol"], industry=str(decision["industry"]), bucket=str(decision["bucket"])),
                    )
                    total_cost = position.avg_cost * position.shares + trade_value
                    position.shares += shares
                    position.avg_cost = total_cost / position.shares if position.shares else 0.0
                    position.current_position_tranches = int(decision["target_position_tranches"])
                    position.current_weight = float(decision["target_weight"])
                    position.last_fill_price = fill_price
                    position.current_shares = int(round(position.shares))
                    positions[decision["symbol"]] = position
                else:
                    if decision["symbol"] not in positions:
                        continue
                    position = positions[decision["symbol"]]
                    fee = trade_value * self.fee_rate
                    tax = trade_value * self.tax_rate
                    if action == "SELL_ALL":
                        shares = position.shares
                        trade_value = shares * fill_price
                        fee = trade_value * self.fee_rate
                        tax = trade_value * self.tax_rate
                        cash += trade_value - fee - tax
                        realized_pnl = trade_value - position.avg_cost * shares - fee - tax
                        positions.pop(decision["symbol"], None)
                    else:
                        shares = min(position.shares, float(requested_shares) if requested_shares > 0 else (trade_value / fill_price if fill_price else 0.0))
                        realized_value = shares * fill_price
                        fee = realized_value * self.fee_rate
                        tax = realized_value * self.tax_rate
                        cash += realized_value - fee - tax
                        realized_pnl = realized_value - position.avg_cost * shares - fee - tax
                        position.shares -= shares
                        position.current_position_tranches = int(decision["target_position_tranches"])
                        position.current_weight = float(decision["target_weight"])
                        position.last_fill_price = fill_price
                        position.current_shares = int(round(position.shares))
                        if position.shares <= 1e-9:
                            positions.pop(decision["symbol"], None)
                        else:
                            positions[decision["symbol"]] = position

                trades.append(
                    {
                        "signal_date": signal_date,
                        "fill_date": fill_date,
                        "symbol": decision["symbol"],
                        "industry": decision["industry"],
                        "bucket": decision["bucket"],
                        "action": action,
                        "fill_price": fill_price,
                        "realized_pnl": realized_pnl,
                    }
                )

            nav_records.append(
                {
                    "date": fill_date,
                    "nav": self._portfolio_value(cash, positions, next_day, mark_field="close"),
                }
            )

        nav_frame = pd.DataFrame(nav_records)
        trade_frame = pd.DataFrame(trades)
        metrics = self._metrics(nav_frame, trade_frame)
        stock_attr = trade_frame.groupby("symbol", as_index=False)["realized_pnl"].sum() if not trade_frame.empty else pd.DataFrame(columns=["symbol", "realized_pnl"])
        industry_attr = trade_frame.groupby("industry", as_index=False)["realized_pnl"].sum() if not trade_frame.empty else pd.DataFrame(columns=["industry", "realized_pnl"])
        return BacktestResult(metrics, nav_frame, trade_frame, stock_attr, industry_attr, approximate_backtest=approximate_backtest)

    @staticmethod
    def _prepare_backtest_features(features: pd.DataFrame) -> pd.DataFrame:
        prepared = features.copy()
        defaults = {
            "in_effective_universe": True,
            "holding_state": "NONE",
            "current_position_tranches": 0,
            "current_weight": 0.0,
            "extra_tranches": 0,
            "last_fill_price": 0.0,
            "current_shares": 0,
            "data_stale": False,
            "fundamental_break": False,
            "cycle_peak_trap": False,
            "quality_pass": True,
        }
        for column, default in defaults.items():
            if column not in prepared.columns:
                prepared[column] = default
            else:
                prepared[column] = prepared[column].fillna(default)
        if "universe_final_score" not in prepared.columns and "final_score" in prepared.columns:
            prepared["universe_final_score"] = prepared["final_score"]
        return prepared

    def _overlay_historical_universe(self, features: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        if self.historical_universe_dir is None or not self.historical_universe_dir.exists():
            return features, False
        histories: list[dict] = []
        for path in sorted(self.historical_universe_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            effective_from = payload.get("effective_from")
            effective_to = payload.get("effective_to")
            stocks = payload.get("stocks", [])
            if not effective_from or not stocks:
                continue
            histories.append(
                {
                    "effective_from": pd.Timestamp(effective_from),
                    "effective_to": pd.Timestamp(effective_to) if effective_to else None,
                    "symbols": {str(item.get("symbol")) for item in stocks if item.get("symbol")},
                }
            )
        if not histories:
            return features, False
        prepared = features.copy()
        approximate_backtest = False
        for date in sorted(pd.to_datetime(prepared["date"].dropna().unique())):
            matches = [
                item
                for item in histories
                if item["effective_from"] <= date and (item["effective_to"] is None or date <= item["effective_to"])
            ]
            if not matches:
                approximate_backtest = True
                continue
            selected = max(matches, key=lambda item: item["effective_from"])
            mask = prepared["date"] == date.strftime("%Y-%m-%d")
            prepared.loc[mask, "in_effective_universe"] = prepared.loc[mask, "symbol"].astype(str).isin(selected["symbols"])
        return prepared, approximate_backtest

    @staticmethod
    def _positions_frame(positions: dict[str, Position]) -> pd.DataFrame:
        if not positions:
            return pd.DataFrame(
                columns=["symbol", "current_position_tranches", "current_weight", "extra_tranches", "last_fill_price", "current_shares"]
            )
        return pd.DataFrame(
            [
                {
                    "symbol": position.symbol,
                    "current_position_tranches": position.current_position_tranches,
                    "current_weight": position.current_weight,
                    "extra_tranches": position.extra_tranches,
                    "last_fill_price": position.last_fill_price,
                    "current_shares": position.current_shares,
                }
                for position in positions.values()
            ]
        )

    @staticmethod
    def _apply_positions_to_features(features: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
        if features.empty or positions.empty:
            return features
        merged = features.merge(
            positions,
            how="left",
            on="symbol",
            suffixes=("", "_position"),
        )
        for column in ("current_position_tranches", "current_weight", "extra_tranches", "last_fill_price", "current_shares"):
            position_column = f"{column}_position"
            if position_column not in merged.columns:
                continue
            merged[column] = merged[position_column].combine_first(merged[column])
            merged = merged.drop(columns=[position_column])
        return merged

    @staticmethod
    def _portfolio_value(cash: float, positions: dict[str, Position], price_frame: pd.DataFrame, mark_field: str) -> float:
        value = cash
        for symbol, position in positions.items():
            if symbol in price_frame.index:
                value += position.shares * float(price_frame.loc[symbol, mark_field])
        return value

    def _metrics(self, nav_frame: pd.DataFrame, trade_frame: pd.DataFrame) -> dict:
        if nav_frame.empty:
            return {"cagr": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "sharpe": 0.0}
        nav = nav_frame["nav"].astype(float)
        returns = nav.pct_change().dropna()
        years = max(len(nav_frame) / 252, 1 / 252)
        cagr = (nav.iloc[-1] / self.initial_cash) ** (1 / years) - 1
        drawdown = nav / nav.cummax() - 1
        wins = (
            (trade_frame["realized_pnl"] > 0).sum() / max((trade_frame["action"].isin(["REDUCE", "SELL_ALL"])).sum(), 1)
            if not trade_frame.empty
            else 0.0
        )
        sharpe = (
            returns.mean() / returns.std(ddof=0) * sqrt(252)
            if not returns.empty and returns.std(ddof=0) > 0
            else 0.0
        )
        return {
            "cagr": float(cagr),
            "max_drawdown": float(drawdown.min()),
            "win_rate": float(wins),
            "sharpe": float(sharpe),
        }
