"""Adverse selection detection and edge realizability analysis."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from src.data.database import PrismDatabase
from src.market.interface import ContractResolver
from src.utils.pandas_typing import as_float, as_int, as_timestamp

logger = logging.getLogger(__name__)


def _to_utc(ts: pd.Timestamp | datetime) -> pd.Timestamp:
    """Normalize naive or tz-aware timestamps to UTC."""
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is not None:
        return stamp.tz_convert("UTC")
    return stamp.tz_localize("UTC")


class AdverseSelectionDetector:
    """
    Detects whether model edges are exploitable or adverse.

    Measures correction latency (how fast markets reprice after scoring events)
    and whether edges persist long enough to trade after realistic delays.
    """

    def __init__(
        self,
        db: PrismDatabase | None = None,
        resolver: ContractResolver | None = None,
    ) -> None:
        self.db = db or PrismDatabase()
        self.resolver = resolver or ContractResolver(db=self.db)

    def load_prices(self, contract_id: str, market_source: str) -> pd.DataFrame:
        prices = self.resolver.load_market_prices(contract_id, market_source)
        if not prices.empty:
            prices = prices.sort_values("timestamp").copy()
            prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
            prices["implied_prob"] = prices.apply(
                lambda r: self.resolver.compute_implied_probability(r, market_source),
                axis=1,
            )
        return prices

    def compute_price_impact(
        self,
        contract_id: str,
        event_timestamp: pd.Timestamp,
        market_source: str = "kalshi",
        window_seconds: int = 60,
    ) -> float:
        """
        Price change in the first window_seconds after a scoring event.

        Positive impact means price moved toward the eventual outcome.
        Returns signed change in implied probability.
        """
        prices = self.load_prices(contract_id, market_source)
        if prices.empty:
            return 0.0

        event_ts = _to_utc(event_timestamp)
        before = prices[prices["timestamp"] <= event_ts]
        after = prices[
            (prices["timestamp"] > event_ts)
            & (prices["timestamp"] <= event_ts + pd.Timedelta(seconds=window_seconds))
        ]

        if before.empty or after.empty:
            return 0.0

        p_before = float(before.iloc[-1]["implied_prob"])
        p_after = float(after.iloc[-1]["implied_prob"])
        return p_after - p_before

    def correction_latency_distribution(
        self,
        game_id: str,
        contract_id: str,
        market_source: str,
        impact_threshold: float = 0.01,
        max_window: int = 120,
    ) -> pd.DataFrame:
        """
        For each scoring event, measure seconds until price moves by impact_threshold.
        """
        states = self.resolver.load_game_states(game_id)
        prices = self.load_prices(contract_id, market_source)
        if states.empty or prices.empty:
            return pd.DataFrame()

        sport = str(states["sport"].iloc[0])
        scoring = states[states["is_scoring_event"].fillna(False)].copy()
        records: list[dict[str, object]] = []

        for row in scoring.itertuples(index=False):
            wc = self.resolver.game_clock_to_wall(
                as_timestamp(row.game_date),
                as_float(row.seconds_remaining),
                sport,
            )
            event_ts = _to_utc(wc)
            p0 = self.resolver.compute_implied_probability(
                prices[prices["timestamp"] <= event_ts].iloc[-1]
                if not prices[prices["timestamp"] <= event_ts].empty
                else prices.iloc[0],
                market_source,
            )

            latency = max_window
            for secs in range(1, max_window + 1):
                window = prices[
                    (prices["timestamp"] > event_ts)
                    & (prices["timestamp"] <= event_ts + pd.Timedelta(seconds=secs))
                ]
                if window.empty:
                    continue
                p1 = float(window.iloc[-1]["implied_prob"])
                if abs(p1 - p0) >= impact_threshold:
                    latency = secs
                    break

            records.append(
                {
                    "game_id": game_id,
                    "seconds_remaining": as_int(row.seconds_remaining),
                    "event_timestamp": event_ts,
                    "correction_latency_seconds": latency,
                    "price_impact": self.compute_price_impact(
                        contract_id, event_ts, market_source, window_seconds=max_window
                    ),
                    "sport": sport,
                }
            )

        return pd.DataFrame(records)

    def test_edge_realizability(
        self,
        edge_signals: pd.DataFrame,
        contract_id: str,
        market_source: str,
        entry_delay_seconds: float = 5.0,
    ) -> pd.DataFrame:
        """
        Would edges have been realizable after entry_delay_seconds?

        Re-checks market price after delay; edge is realizable if it persists
        with the same sign and at least half the original magnitude.
        """
        if edge_signals.empty:
            return pd.DataFrame()

        prices = self.load_prices(contract_id, market_source)
        if prices.empty:
            return pd.DataFrame()

        results = edge_signals.copy()
        realizable: list[bool] = []
        delayed_edge: list[float] = []

        delayed_prices: list[float] = []
        for row in edge_signals.itertuples(index=False):
            sport = str(getattr(row, "sport", "NFL"))
            game_date = getattr(row, "game_date", pd.Timestamp("2023-01-01"))
            wc = self.resolver.game_clock_to_wall(
                as_timestamp(game_date),
                as_float(row.seconds_remaining),
                sport,
            )
            signal_ts = _to_utc(wc)
            delayed_ts = signal_ts + pd.Timedelta(seconds=entry_delay_seconds)

            at_delay = prices[prices["timestamp"] <= delayed_ts]
            if at_delay.empty:
                realizable.append(False)
                delayed_edge.append(0.0)
                delayed_prices.append(float("nan"))
                continue

            delayed_price = float(at_delay.iloc[-1]["implied_prob"])
            delayed_prices.append(delayed_price)
            orig_edge = as_float(row.edge)
            model_prob = as_float(row.model_prob)
            new_edge = model_prob - delayed_price
            delayed_edge.append(new_edge)

            same_sign = (orig_edge == 0) or (np.sign(new_edge) == np.sign(orig_edge))
            persists = abs(new_edge) >= 0.5 * abs(orig_edge)
            realizable.append(bool(same_sign and persists))

        results["delayed_market_price"] = delayed_prices
        results["delayed_edge"] = delayed_edge
        results["realizable"] = realizable
        return results

    def summarize_realizability(self, realizability: pd.DataFrame) -> dict[str, float]:
        """Summary stats for edge realizability analysis."""
        if realizability.empty:
            return {"n_signals": 0, "realizable_rate": 0.0, "mean_delayed_edge": 0.0}
        return {
            "n_signals": float(len(realizability)),
            "realizable_rate": float(realizability["realizable"].mean()),
            "mean_delayed_edge": float(realizability["delayed_edge"].mean()),
            "mean_abs_edge": float(realizability["edge"].abs().mean()),
        }
