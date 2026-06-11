"""Tests for NFL load_pbp with mocked nfl_data_py."""

from unittest.mock import patch

import pandas as pd
import pytest

from src.data.nfl import NFLAdapter


@patch("src.data.nfl.nfl.import_pbp_data")
def test_load_pbp(mock_import, nfl_adapter, sample_nfl_pbp):
    mock_import.return_value = sample_nfl_pbp.copy()
    result = nfl_adapter.load_pbp([2023])
    assert len(result) == len(sample_nfl_pbp)
    assert (result["score_differential"] == result["home_score"] - result["away_score"]).all()


@patch("src.data.nfl.nfl.import_pbp_data")
def test_load_pbp_drops_null_seconds(mock_import, nfl_adapter, sample_nfl_pbp):
    pbp = sample_nfl_pbp.copy()
    pbp.loc[0, "game_seconds_remaining"] = None
    mock_import.return_value = pbp
    result = nfl_adapter.load_pbp([2023])
    assert len(result) == len(pbp) - 1


def test_event_type_field_goal(nfl_adapter):
    row = pd.Series(
        {
            "play_type": "field_goal",
            "touchdown": 0,
            "field_goal_result": "made",
            "extra_point_result": None,
            "two_point_conv_result": None,
            "safety": 0,
        }
    )
    assert nfl_adapter._event_type(row) == "field_goal"


def test_validate_invalid_period(nfl_adapter, sample_nfl_game):
    bad = sample_nfl_game.copy()
    bad.loc[0, "game_period"] = 6
    with pytest.raises(ValueError, match="invalid periods"):
        nfl_adapter.validate_game_states(bad)
