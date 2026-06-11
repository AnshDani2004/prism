"""Event-driven backtester for prediction market strategies."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.sizing import KellySizer
from src.market.interface import ContractResolver

logger = logging.getLogger(__name__)

EXPERIMENT_LOG = Path("config/experiment_log.jsonl")


@dataclass
class Trade:
    """Single executed trade with full metadata."""

    game_id: str
    contract_id: str
    signal_time: pd.Timestamp
    execution_time: pd.Timestamp
    side: str  # 'buy_yes' or 'sell_yes'
    n_contracts: int
    signal_price: float
    fill_price: float
    fee: float
    model_prob: float
    edge: float
    edge_type: str
    home_won: bool
    pnl: float


@dataclass
class WalkForwardFold:
    """Train/test window for walk-forward validation."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class BacktestResults:
    """Container for backtest output."""

    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    initial_capital: float = 10_000.0

    @property
    def total_pnl(self) -> float:
        return float(sum(t.pnl for t in self.trades))

    @property
    def final_capital(self) -> float:
        return self.initial_capital + self.total_pnl


class PredictionMarketBacktester:
    """
    Event-driven backtester for prediction market strategies.

    Design principles:
    - No lookahead: trades execute only after signal_time + entry_delay
    - Fills at the touch (buy at ask, sell at bid), not mid
    - Kalshi fee formula: ceil_to_cent(0.07 * n * p * (1-p))
    - Liquidity cap: max fraction of displayed touch volume
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        max_position_size: float = 0.05,
        fee_rate: float = 0.07,
        max_volume_participation: float = 0.10,
        entry_delay_seconds: float = 5.0,
        edge_threshold: float = 0.05,
        experiment_log_path: Path | str = EXPERIMENT_LOG,
    ) -> None:
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.fee_rate = fee_rate
        self.max_volume_participation = max_volume_participation
        self.entry_delay_seconds = entry_delay_seconds
        self.edge_threshold = edge_threshold
        self.experiment_log_path = Path(experiment_log_path)
        self.resolver = ContractResolver()
        self.walk_forward_folds: list[WalkForwardFold] = []

    @staticmethod
    def kalshi_fee(n_contracts: int, price: float, fee_rate: float = 0.07) -> float:
        """fee = ceil_to_cent(fee_rate * n * p * (1-p))"""
        raw = fee_rate * n_contracts * price * (1.0 - price)
        return math.ceil(raw * 100) / 100.0

    def _signal_timestamp(self, row: pd.Series) -> pd.Timestamp:
        sport = str(row.get("sport", "NFL"))
        game_date = row.get("game_date", pd.Timestamp("2020-01-01"))
        secs = int(row["seconds_remaining"])
        wc = self.resolver.game_clock_to_wall(game_date, secs, sport)
        return pd.Timestamp(wc, tz="UTC")

    def _price_at_time(
        self,
        prices: pd.DataFrame,
        timestamp: pd.Timestamp,
        side: str,
        spread_override: float | None = None,
    ) -> tuple[float, float | None]:
        """Return (fill_price, touch_volume) at or before timestamp."""
        if prices.empty:
            return 0.5, None

        prices = prices.copy()
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
        at_time = prices[prices["timestamp"] <= timestamp]
        if at_time.empty:
            row = prices.iloc[0]
        else:
            row = at_time.iloc[-1]

        mid = self.resolver.compute_implied_probability(row, str(row.get("market_source", "kalshi")))
        bid = float(row["yes_bid"]) if pd.notna(row.get("yes_bid")) else mid - 0.01
        ask = float(row["yes_ask"]) if pd.notna(row.get("yes_ask")) else mid + 0.01

        if spread_override is not None:
            half = spread_override / 2
            bid, ask = mid - half, mid + half

        fill = ask if side == "buy_yes" else bid
        volume = float(row["volume"]) if pd.notna(row.get("volume")) else None
        return float(np.clip(fill, 0.01, 0.99)), volume

    def simulate_trade(
        self,
        capital: float,
        model_prob: float,
        fill_price: float,
        side: str,
        sizer: KellySizer,
        touch_volume: float | None = None,
    ) -> tuple[int, float]:
        """Return (n_contracts, fee) for a single trade attempt."""
        if side == "buy_yes":
            fraction = sizer.kelly_fraction(model_prob, fill_price)
        else:
            fraction = sizer.kelly_fraction_sell(model_prob, fill_price)

        fraction = min(fraction, self.max_position_size)
        n_contracts = sizer.contracts_from_capital(capital, fraction, fill_price)

        if touch_volume is not None and touch_volume > 0:
            max_contracts = int(self.max_volume_participation * touch_volume)
            n_contracts = min(n_contracts, max_contracts)

        fee = self.kalshi_fee(n_contracts, fill_price, self.fee_rate) if n_contracts > 0 else 0.0
        return n_contracts, fee

    def _settle_trade(
        self,
        side: str,
        n_contracts: int,
        fill_price: float,
        fee: float,
        home_won: bool,
    ) -> float:
        """PnL for a resolved binary contract."""
        if n_contracts == 0:
            return 0.0
        if side == "buy_yes":
            payout = n_contracts * (1.0 if home_won else 0.0)
            cost = n_contracts * fill_price + fee
            return payout - cost
        # sell_yes: collected premium, pay 1 if home wins
        premium = n_contracts * fill_price - fee
        liability = n_contracts * (1.0 if home_won else 0.0)
        return premium - liability

    def run(
        self,
        edge_signals: pd.DataFrame,
        market_prices: pd.DataFrame,
        game_outcomes: pd.DataFrame,
        sizer: KellySizer,
        fee_rate: float | None = None,
        spread_override: float | None = None,
    ) -> BacktestResults:
        """
        Main event-driven backtest loop.

        Processes signals in chronological order without lookahead.
        """
        fee_rate = fee_rate if fee_rate is not None else self.fee_rate
        self.fee_rate = fee_rate

        if edge_signals.empty:
            return BacktestResults(initial_capital=self.initial_capital)

        signals = edge_signals.copy()
        if "signal_time" not in signals.columns:
            signals["signal_time"] = signals.apply(self._signal_timestamp, axis=1)
        signals = signals.sort_values("signal_time")

        outcome_map = dict(
            zip(game_outcomes["game_id"], game_outcomes["home_won"], strict=False)
        )
        capital = self.initial_capital
        trades: list[Trade] = []
        equity_rows: list[dict[str, object]] = []

        for _, sig in signals.iterrows():
            edge = float(sig["edge"])
            if abs(edge) < self.edge_threshold:
                continue

            side = "buy_yes" if edge > 0 else "sell_yes"
            signal_time = pd.Timestamp(sig["signal_time"])
            execution_time = signal_time + pd.Timedelta(seconds=self.entry_delay_seconds)

            contract_id = str(sig["contract_id"])
            market_source = str(sig["market_source"])
            prices = market_prices[
                (market_prices["contract_id"] == contract_id)
                & (market_prices["market_source"] == market_source)
            ]
            fill_price, volume = self._price_at_time(
                prices, execution_time, side, spread_override=spread_override
            )

            model_prob = float(sig["model_prob"])
            if side == "buy_yes":
                current_edge = model_prob - fill_price
            else:
                current_edge = fill_price - model_prob

            if abs(current_edge) < self.edge_threshold:
                continue  # edge evaporated after delay

            n_contracts, fee = self.simulate_trade(
                capital, model_prob, fill_price, side, sizer, touch_volume=volume
            )
            if n_contracts == 0:
                continue

            game_id = str(sig["game_id"])
            home_won = bool(outcome_map.get(game_id, False))
            pnl = self._settle_trade(side, n_contracts, fill_price, fee, home_won)
            capital += pnl

            trades.append(
                Trade(
                    game_id=game_id,
                    contract_id=contract_id,
                    signal_time=signal_time,
                    execution_time=execution_time,
                    side=side,
                    n_contracts=n_contracts,
                    signal_price=float(sig["market_price"]),
                    fill_price=fill_price,
                    fee=fee,
                    model_prob=model_prob,
                    edge=edge,
                    edge_type=str(sig.get("edge_type", "unknown")),
                    home_won=home_won,
                    pnl=pnl,
                )
            )
            equity_rows.append(
                {
                    "timestamp": execution_time,
                    "capital": capital,
                    "pnl": pnl,
                    "game_id": game_id,
                }
            )

        equity = pd.DataFrame(equity_rows)
        results = BacktestResults(
            trades=trades,
            equity_curve=equity,
            initial_capital=self.initial_capital,
        )
        self._log_experiment()
        logger.info(
            "Backtest complete: %d trades, PnL=%.2f, final=%.2f",
            len(trades),
            results.total_pnl,
            results.final_capital,
        )
        return results

    def _log_experiment(self) -> None:
        """Append strategy config to experiment log for deflated Sharpe."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "edge_threshold": self.edge_threshold,
            "max_position_size": self.max_position_size,
            "fee_rate": self.fee_rate,
            "entry_delay_seconds": self.entry_delay_seconds,
            "max_volume_participation": self.max_volume_participation,
        }
        self.experiment_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.experiment_log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def build_walk_forward_folds(
        self,
        game_dates: pd.Series,
        n_folds: int = 3,
        train_ratio: float = 0.7,
    ) -> list[WalkForwardFold]:
        """Build non-overlapping walk-forward train/test windows."""
        dates = pd.to_datetime(game_dates).sort_values().unique()
        if len(dates) < n_folds + 1:
            return []

        folds: list[WalkForwardFold] = []
        chunk = len(dates) // (n_folds + 1)
        for i in range(n_folds):
            train_end_idx = (i + 1) * chunk
            test_end_idx = min((i + 2) * chunk, len(dates))
            train_start = pd.Timestamp(dates[0])
            train_end = pd.Timestamp(dates[train_end_idx - 1])
            test_start = pd.Timestamp(dates[train_end_idx])
            test_end = pd.Timestamp(dates[test_end_idx - 1])
            if train_end < test_start:
                folds.append(
                    WalkForwardFold(
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                    )
                )
        self.walk_forward_folds = folds
        return folds
