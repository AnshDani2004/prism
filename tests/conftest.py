"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import date

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.data.database import PrismDatabase
from src.data.mapping import GameContractMapper
from src.data.nfl import NFLAdapter


@pytest.fixture
def test_db() -> duckdb.DuckDBPyConnection:
    """Fresh in-memory DuckDB per test — never uses production data."""
    con = duckdb.connect(":memory:")
    db = PrismDatabase(db_path=":memory:")
    db._conn = con  # noqa: SLF001 — test fixture
    db.initialize_schema()
    return con


@pytest.fixture
def prism_db(test_db: duckdb.DuckDBPyConnection) -> PrismDatabase:
    """PrismDatabase backed by in-memory DuckDB."""
    db = PrismDatabase(db_path=":memory:")
    db._conn = test_db  # noqa: SLF001
    return db


@pytest.fixture
def nfl_adapter(prism_db: PrismDatabase) -> NFLAdapter:
    return NFLAdapter(db=prism_db)


@pytest.fixture
def sample_nfl_pbp() -> pd.DataFrame:
    """Minimal NFL play-by-play for unit tests."""
    return pd.DataFrame(
        {
            "game_id": ["2023_01_KC_PHI"] * 8,
            "season": [2023] * 8,
            "game_date": [date(2023, 1, 15)] * 8,
            "home_team": ["KC"] * 8,
            "away_team": ["PHI"] * 8,
            "qtr": [1, 1, 1, 2, 2, 3, 4, 4],
            "game_seconds_remaining": [3600, 3500, 3400, 2700, 2600, 1800, 900, 0],
            "score_differential": [0, 0, 7, 7, 7, 14, 14, 14],
            "posteam": ["KC", "KC", "KC", "PHI", "PHI", "KC", "KC", "KC"],
            "home_score": [0, 0, 7, 7, 7, 14, 14, 14],
            "away_score": [0, 0, 0, 0, 0, 0, 0, 0],
            "play_type": [
                "pass",
                "run",
                "touchdown",
                "pass",
                "pass",
                "touchdown",
                "pass",
                "end_game",
            ],
            "touchdown": [0, 0, 1, 0, 0, 1, 0, 0],
            "field_goal_result": [None] * 8,
            "extra_point_result": [None] * 8,
            "two_point_conv_result": [None] * 8,
            "safety": [0] * 8,
        }
    )


@pytest.fixture
def sample_nfl_game(sample_nfl_pbp: pd.DataFrame, nfl_adapter: NFLAdapter) -> pd.DataFrame:
    return nfl_adapter.extract_game_states(sample_nfl_pbp)


@pytest.fixture
def sample_games_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2023_01_15_KC_PHI", "2023_01_07_BUF_CIN"],
            "sport": ["NFL", "NFL"],
            "game_date": [date(2023, 1, 15), date(2023, 1, 7)],
            "home_team": ["KC", "CIN"],
            "away_team": ["PHI", "BUF"],
        }
    )


@pytest.fixture
def sample_contracts_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contract_id": ["KXNFL-KC-PHI-0115", "KXNFL-BUF-CIN-0107"],
            "market_source": ["kalshi", "kalshi"],
            "sport": ["NFL", "NFL"],
            "home_team": ["KC", "CIN"],
            "away_team": ["PHI", "BUF"],
            "game_date": [date(2023, 1, 15), date(2023, 1, 7)],
            "contract_type": ["moneyline", "moneyline"],
            "resolved_outcome": ["home", "away"],
            "resolution_price": [0.65, 0.42],
        }
    )


@pytest.fixture
def mapper(prism_db: PrismDatabase) -> GameContractMapper:
    return GameContractMapper(db=prism_db)


@pytest.fixture
def sample_market_prices() -> pd.DataFrame:
    """Synthetic Kalshi prices for testing."""
    timestamps = pd.date_range("2023-01-15 18:00", periods=5, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "contract_id": ["KXNFL-KC-PHI-0115"] * 5,
            "market_source": ["kalshi"] * 5,
            "timestamp": timestamps,
            "yes_price": [0.55, 0.58, 0.62, 0.65, 0.64],
            "no_price": [0.45, 0.42, 0.38, 0.35, 0.36],
            "yes_bid": [0.54, 0.57, 0.61, 0.64, 0.63],
            "yes_ask": [0.56, 0.59, 0.63, 0.66, 0.65],
            "volume": [100.0, 150.0, 200.0, 180.0, 120.0],
        }
    )
