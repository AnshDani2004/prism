"""Tests for NFL data pipeline."""

import pandas as pd
import pytest

from src.data.nfl import NFLAdapter


def test_extract_game_states_reduces_plays(nfl_adapter: NFLAdapter, sample_nfl_pbp: pd.DataFrame):
    states = nfl_adapter.extract_game_states(sample_nfl_pbp)
    assert len(states) < len(sample_nfl_pbp)
    assert len(states) >= 4  # quarter starts + scoring events


def test_scoring_events_flagged(nfl_adapter: NFLAdapter, sample_nfl_pbp: pd.DataFrame):
    states = nfl_adapter.extract_game_states(sample_nfl_pbp)
    scoring = states[states["is_scoring_event"]]
    assert len(scoring) >= 2
    assert set(scoring["event_type"].dropna()) >= {"touchdown"}


def test_score_differential_consistent(nfl_adapter: NFLAdapter, sample_nfl_game: pd.DataFrame):
    diff = sample_nfl_game["home_score"] - sample_nfl_game["away_score"]
    assert (sample_nfl_game["score_differential"] == diff).all()


def test_seconds_remaining_monotonic(nfl_adapter: NFLAdapter, sample_nfl_game: pd.DataFrame):
    secs = sample_nfl_game["seconds_remaining"].values
    assert all(secs[i] >= secs[i + 1] for i in range(len(secs) - 1))


def test_validate_passes(nfl_adapter: NFLAdapter, sample_nfl_game: pd.DataFrame):
    assert nfl_adapter.validate_game_states(sample_nfl_game) is True


def test_validate_fails_on_bad_scores(nfl_adapter: NFLAdapter, sample_nfl_game: pd.DataFrame):
    bad = sample_nfl_game.copy()
    bad.loc[0, "score_differential"] = 999
    with pytest.raises(ValueError, match="inconsistent score"):
        nfl_adapter.validate_game_states(bad)


def test_validate_fails_on_duplicate_keys(nfl_adapter: NFLAdapter, sample_nfl_game: pd.DataFrame):
    duped = pd.concat([sample_nfl_game, sample_nfl_game.iloc[[-1]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        nfl_adapter.validate_game_states(duped)


def test_validate_fails_on_negative_scores(nfl_adapter: NFLAdapter, sample_nfl_game: pd.DataFrame):
    bad = sample_nfl_game.copy()
    bad.loc[0, "home_score"] = -1
    bad.loc[0, "score_differential"] = bad.loc[0, "home_score"] - bad.loc[0, "away_score"]
    with pytest.raises(ValueError, match="negative scores"):
        nfl_adapter.validate_game_states(bad)
