"""Tests for NBA data pipeline."""

from datetime import date

import pandas as pd
import pytest

from src.data.nba import NBAAdapter, parse_game_clock, period_seconds_remaining


def test_parse_game_clock():
    assert parse_game_clock("12:00") == 720
    assert parse_game_clock("2:30") == 150
    assert parse_game_clock("0:00") == 0
    assert parse_game_clock(None) is None
    assert parse_game_clock("invalid") is None


def test_period_seconds_remaining_regulation():
    assert period_seconds_remaining(1, 720) == 2880
    assert period_seconds_remaining(4, 0) == 0


def test_period_seconds_remaining_overtime():
    assert period_seconds_remaining(5, 300) == 300


def test_extract_game_states_sampling():
    adapter = NBAAdapter()
    pbp = pd.DataFrame(
        {
            "game_id": ["0022300001"] * 6,
            "season": [2023] * 6,
            "game_date": [date(2023, 10, 24)] * 6,
            "home_team": ["LAL"] * 6,
            "away_team": ["DEN"] * 6,
            "PERIOD": [1, 1, 1, 1, 2, 2],
            "PCTIMESTRING": ["12:00", "10:00", "8:00", "7:30", "12:00", "6:00"],
            "SCORE": ["0 - 0", "2 - 0", "2 - 2", "2 - 2", "2 - 2", "5 - 2"],
            "HOMEDESCRIPTION": [
                None,
                "James makes 2pt",
                None,
                "Murray makes 3pt shot",
                None,
                "Davis makes layup",
            ],
            "VISITORDESCRIPTION": [None] * 6,
            "PLAYER1_TEAM_ABBREVIATION": ["LAL", "LAL", "DEN", "DEN", "LAL", "LAL"],
        }
    )
    states = adapter.extract_game_states(pbp)
    assert not states.empty
    assert states["sport"].iloc[0] == "NBA"
    assert (states["score_differential"] == states["home_score"] - states["away_score"]).all()


def test_validate_nba_states():
    adapter = NBAAdapter()
    states = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "seconds_remaining": [2880, 1440],
            "score_differential": [0, 5],
            "home_score": [0, 10],
            "away_score": [0, 5],
        }
    )
    assert adapter.validate_game_states(states) is True


def test_validate_fails_inconsistent_scores():
    adapter = NBAAdapter()
    states = pd.DataFrame(
        {
            "game_id": ["g1"],
            "seconds_remaining": [100],
            "score_differential": [10],
            "home_score": [5],
            "away_score": [0],
        }
    )
    with pytest.raises(ValueError, match="inconsistent"):
        adapter.validate_game_states(states)
