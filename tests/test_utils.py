"""Tests for utility modules."""

import logging

import pandas as pd
import pytest

from config.settings import Settings
from src.data.base import SportDataAdapter
from src.data.nfl import NFLAdapter
from src.utils.logging import setup_logging
from src.utils.validation import assert_column_in_range, assert_no_duplicates


def test_setup_logging():
    settings = Settings(LOG_LEVEL="DEBUG")
    setup_logging(settings)
    assert logging.getLogger().level == logging.DEBUG


def test_assert_no_duplicates_raises():
    df = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
    with pytest.raises(ValueError, match="duplicate"):
        assert_no_duplicates(df, ["a", "b"], "test")


def test_assert_no_duplicates_empty():
    assert_no_duplicates(pd.DataFrame(), ["a"], "test")


def test_assert_column_in_range_raises():
    df = pd.DataFrame({"x": [0.5, 1.5]})
    with pytest.raises(ValueError, match="outside"):
        assert_column_in_range(df, "x", 0.0, 1.0, "test")


def test_sport_adapter_ingest(nfl_adapter, sample_nfl_pbp, monkeypatch):
    monkeypatch.setattr(nfl_adapter, "load_pbp", lambda seasons: sample_nfl_pbp)
    states = nfl_adapter.ingest([2023])
    assert len(states) > 0
    assert nfl_adapter.db.count("game_states") > 0


def test_sport_adapter_is_abstract():
    assert SportDataAdapter.__abstractmethods__
