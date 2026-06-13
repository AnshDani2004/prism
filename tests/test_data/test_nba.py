"""Tests for NBA data pipeline (bulk LeagueGameLog adapter)."""
from datetime import date
import pandas as pd
import pytest
from src.data.nba import NBAAdapter, period_seconds_remaining


def test_period_seconds_remaining_regulation():
    assert period_seconds_remaining(1, 720) == 2880
    assert period_seconds_remaining(4, 0) == 0


def test_period_seconds_remaining_overtime():
    assert period_seconds_remaining(5, 300) == 300


def test_extract_game_states_sampling():
    adapter = NBAAdapter()
    # Simulate LeagueGameLog rows (one per team per game)
    pbp = pd.DataFrame({
        "GAME_ID": ["0022300001", "0022300001"],
        "season": [2023, 2023],
        "GAME_DATE": ["2023-10-24", "2023-10-24"],
        "TEAM_ABBREVIATION": ["LAL", "DEN"],
        "MATCHUP": ["LAL vs. DEN", "DEN @ LAL"],
        "PTS": [112, 107],
        "is_home": [True, False],
    })
    states = adapter.extract_game_states(pbp)
    assert not states.empty
    assert states["sport"].iloc[0] == "NBA"
    assert (states["score_differential"] == states["home_score"] - states["away_score"]).all()
    # Should have start + 3 quarter boundaries + final = 5 states per game
    assert len(states) == 5


def test_extract_game_states_reduces_plays():
    adapter = NBAAdapter()
    pbp = pd.DataFrame({
        "GAME_ID": ["0022300002", "0022300002"],
        "season": [2023, 2023],
        "GAME_DATE": ["2023-11-01", "2023-11-01"],
        "TEAM_ABBREVIATION": ["GSW", "BOS"],
        "MATCHUP": ["GSW vs. BOS", "BOS @ GSW"],
        "PTS": [120, 115],
        "is_home": [True, False],
    })
    states = adapter.extract_game_states(pbp)
    assert len(states) == 5
    assert states["seconds_remaining"].iloc[0] == 2880
    assert states["seconds_remaining"].iloc[-1] == 0


def test_validate_nba_states():
    adapter = NBAAdapter()
    states = pd.DataFrame({
        "game_id": ["g1", "g1"],
        "seconds_remaining": [2880, 1440],
        "score_differential": [0, 5],
        "home_score": [0, 10],
        "away_score": [0, 5],
    })
    assert adapter.validate_game_states(states) is True


def test_validate_fails_inconsistent_scores():
    adapter = NBAAdapter()
    states = pd.DataFrame({
        "game_id": ["g1"],
        "seconds_remaining": [100],
        "score_differential": [10],
        "home_score": [5],
        "away_score": [0],
    })
    with pytest.raises(ValueError, match="inconsistent"):
        adapter.validate_game_states(states)
