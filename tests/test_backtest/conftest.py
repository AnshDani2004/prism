"""Fixtures for backtest tests."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtest.engine import PredictionMarketBacktester
from src.backtest.sizing import KellySizer


@pytest.fixture
def sizer() -> KellySizer:
    return KellySizer(kelly_fraction=0.25, max_position_size=0.05)


@pytest.fixture
def backtester() -> PredictionMarketBacktester:
    return PredictionMarketBacktester(
        initial_capital=10_000.0,
        edge_threshold=0.03,
        entry_delay_seconds=5.0,
    )


@pytest.fixture
def sample_edge_signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2"],
            "contract_id": ["c1", "c1", "c2"],
            "market_source": ["kalshi", "kalshi", "kalshi"],
            "seconds_remaining": [1800, 900, 1200],
            "sport": ["NFL", "NFL", "NFL"],
            "game_date": [date(2023, 1, 15)] * 3,
            "model_prob": [0.65, 0.70, 0.40],
            "market_price": [0.55, 0.58, 0.50],
            "edge": [0.10, 0.12, -0.10],
            "edge_type": ["inplay_staleness", "inplay_staleness", "inplay_drift"],
            "signal_time": pd.to_datetime(
                [
                    "2023-01-15 19:00:00",
                    "2023-01-15 19:15:00",
                    "2023-01-16 19:10:00",
                ],
                utc=True,
            ),
        }
    )


@pytest.fixture
def sample_market_prices_bt() -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-15 18:55", periods=30, freq="1min", tz="UTC")
    prices = []
    for i, ts in enumerate(timestamps):
        mid = 0.52 + i * 0.005
        prices.append(
            {
                "contract_id": "c1",
                "market_source": "kalshi",
                "timestamp": ts,
                "yes_price": mid,
                "no_price": 1 - mid,
                "yes_bid": mid - 0.01,
                "yes_ask": mid + 0.01,
                "volume": 200.0,
            }
        )
    ts2 = pd.date_range("2023-01-16 18:55", periods=30, freq="1min", tz="UTC")
    for i, ts in enumerate(ts2):
        mid = 0.48 - i * 0.002
        prices.append(
            {
                "contract_id": "c2",
                "market_source": "kalshi",
                "timestamp": ts,
                "yes_price": mid,
                "no_price": 1 - mid,
                "yes_bid": mid - 0.01,
                "yes_ask": mid + 0.01,
                "volume": 150.0,
            }
        )
    return pd.DataFrame(prices)


@pytest.fixture
def sample_game_outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "home_won": [True, False],
        }
    )
