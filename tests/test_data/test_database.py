"""Tests for DuckDB interface."""

import pandas as pd

from src.data.database import PrismDatabase


def test_schema_initialization(prism_db: PrismDatabase):
    tables = prism_db.query_df("SHOW TABLES")["name"].tolist()
    assert "game_states" in tables
    assert "contracts" in tables
    assert "market_prices" in tables


def test_upsert_dataframe(prism_db: PrismDatabase):
    df = pd.DataFrame(
        {
            "game_id": ["g1"],
            "sport": ["NFL"],
            "season": [2023],
            "game_date": [pd.Timestamp("2023-01-15").date()],
            "home_team": ["KC"],
            "away_team": ["PHI"],
            "seconds_remaining": [3600],
            "game_period": [1],
            "score_differential": [0],
            "home_score": [0],
            "away_score": [0],
            "possession": [None],
            "is_scoring_event": [False],
            "event_type": [None],
        }
    )
    n = prism_db.upsert_dataframe("game_states", df, primary_key=["game_id", "seconds_remaining"])
    assert n == 1
    assert prism_db.count("game_states") == 1

    df2 = df.copy()
    df2["home_score"] = [7]
    df2["score_differential"] = [7]
    prism_db.upsert_dataframe("game_states", df2, primary_key=["game_id", "seconds_remaining"])
    row = prism_db.query_df("SELECT home_score FROM game_states WHERE game_id='g1'")
    assert row["home_score"].iloc[0] == 7


def test_phase1_checkpoint_empty(prism_db: PrismDatabase):
    results = prism_db.phase1_checkpoint()
    assert results["score_consistency_check"] == 0
    assert results["nfl_game_states"] == 0


def test_close_connection(prism_db: PrismDatabase):
    prism_db.close()
    assert prism_db._conn is None  # noqa: SLF001


def test_insert_dataframe(prism_db: PrismDatabase):
    df = pd.DataFrame(
        {
            "contract_id": ["c1"],
            "market_source": ["kalshi"],
            "sport": ["NFL"],
            "home_team": ["KC"],
            "away_team": ["PHI"],
            "game_date": [pd.Timestamp("2023-01-15").date()],
            "contract_type": ["moneyline"],
            "resolved_outcome": [None],
            "resolution_price": [None],
        }
    )
    n = prism_db.insert_dataframe("contracts", df)
    assert n == 1
