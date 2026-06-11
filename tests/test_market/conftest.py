"""Fixtures for market interface tests."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.market.adverse_selection import AdverseSelectionDetector
from src.market.edge import EdgeCalculator
from src.market.interface import ContractResolver
from src.models.base import WinProbabilityModel


class FixedProbModel(WinProbabilityModel):
    """Model that returns a fixed column or constant probability."""

    model_name = "fixed"

    def __init__(self, prob: float = 0.55, column: str | None = None) -> None:
        self.prob = prob
        self.column = column

    def fit(self, game_states: pd.DataFrame, outcomes: pd.Series) -> FixedProbModel:
        return self

    def predict(self, game_states: pd.DataFrame) -> np.ndarray:
        if self.column and self.column in game_states.columns:
            return game_states[self.column].to_numpy(dtype=float)
        return np.full(len(game_states), self.prob)


@pytest.fixture
def fixed_model() -> FixedProbModel:
    return FixedProbModel()


@pytest.fixture
def mirror_model() -> FixedProbModel:
    """Model that echoes market_price column as prediction (efficient market)."""
    return FixedProbModel(column="market_price")


@pytest.fixture
def market_game_states() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2023_01_KC_PHI"] * 4,
            "sport": ["NFL"] * 4,
            "season": [2023] * 4,
            "game_date": [date(2023, 1, 15)] * 4,
            "home_team": ["KC"] * 4,
            "away_team": ["PHI"] * 4,
            "seconds_remaining": [3600, 2700, 1800, 900],
            "game_period": [1, 2, 3, 4],
            "score_differential": [0, 7, 7, 14],
            "home_score": [0, 7, 7, 14],
            "away_score": [0, 0, 0, 0],
            "possession": ["KC"] * 4,
            "is_scoring_event": [False, True, False, True],
            "event_type": [None, "touchdown", None, "touchdown"],
        }
    )


@pytest.fixture
def market_prices_timeline() -> pd.DataFrame:
    """Prices that lag after scoring events (staleness simulation)."""
    timestamps = pd.date_range("2023-01-15 18:00", periods=8, freq="3min", tz="UTC")
    return pd.DataFrame(
        {
            "contract_id": ["KXNFL-KC-PHI"] * 8,
            "market_source": ["kalshi"] * 8,
            "timestamp": timestamps,
            "yes_price": [0.52, 0.53, 0.54, 0.55, 0.56, 0.62, 0.63, 0.70],
            "no_price": [0.48, 0.47, 0.46, 0.45, 0.44, 0.38, 0.37, 0.30],
            "yes_bid": [0.51, 0.52, 0.53, 0.54, 0.55, 0.61, 0.62, 0.69],
            "yes_ask": [0.53, 0.54, 0.55, 0.56, 0.57, 0.63, 0.64, 0.71],
            "volume": [100.0] * 8,
        }
    )


@pytest.fixture
def resolver(prism_db) -> ContractResolver:
    return ContractResolver(db=prism_db)


@pytest.fixture
def edge_calc(prism_db) -> EdgeCalculator:
    return EdgeCalculator(db=prism_db)


@pytest.fixture
def adverse_detector(prism_db) -> AdverseSelectionDetector:
    return AdverseSelectionDetector(db=prism_db)


@pytest.fixture
def populated_market_db(prism_db, market_game_states, market_prices_timeline, sample_contracts_df):
    """DB with game states, prices, contracts, and mapping."""
    prism_db.upsert_dataframe(
        "game_states",
        market_game_states,
        primary_key=["game_id", "seconds_remaining"],
    )
    prism_db.upsert_dataframe(
        "market_prices",
        market_prices_timeline,
        primary_key=["contract_id", "market_source", "timestamp"],
    )
    contracts = sample_contracts_df.iloc[[0]].copy()
    contracts["contract_id"] = "KXNFL-KC-PHI"
    prism_db.upsert_dataframe(
        "contracts", contracts, primary_key=["contract_id", "market_source"]
    )
    mapping = pd.DataFrame(
        {
            "game_id": ["2023_01_KC_PHI"],
            "contract_id": ["KXNFL-KC-PHI"],
            "market_source": ["kalshi"],
            "match_confidence": [0.95],
            "match_method": ["exact"],
        }
    )
    prism_db.upsert_dataframe(
        "game_contract_map",
        mapping,
        primary_key=["game_id", "contract_id", "market_source"],
    )
    return prism_db
