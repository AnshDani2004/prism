"""Tests for contract resolver and price alignment."""

import numpy as np
import pandas as pd
import pytest

from src.market.interface import ContractResolver


def test_implied_probability_kalshi_mid(resolver: ContractResolver):
    row = pd.Series({"yes_bid": 0.54, "yes_ask": 0.56, "yes_price": 0.50})
    prob = resolver.compute_implied_probability(row, "kalshi")
    assert prob == pytest.approx(0.55, abs=0.001)
    assert 0 <= prob <= 1


def test_implied_probability_polymarket(resolver: ContractResolver):
    row = pd.Series({"yes_price": 0.62, "no_price": 0.38})
    prob = resolver.compute_implied_probability(row, "polymarket")
    assert prob == pytest.approx(0.62, abs=0.001)


def test_implied_probability_sportsbook_vig_removed(resolver: ContractResolver):
    # Raw implied probs sum to 1.08 (8% overround)
    row = pd.Series({"yes_price": 0.55, "no_price": 0.53})
    prob = resolver.compute_implied_probability(row, "sportsbook")
    assert prob == pytest.approx(0.55 / (0.55 + 0.53), abs=0.001)
    assert prob < 0.55  # vig removal lowers naive estimate


def test_implied_probability_in_unit_interval(resolver: ContractResolver, market_prices_timeline):
    for _, row in market_prices_timeline.iterrows():
        prob = resolver.compute_implied_probability(row, "kalshi")
        assert 0 <= prob <= 1


def test_align_dataframes(
    resolver: ContractResolver,
    market_game_states: pd.DataFrame,
    market_prices_timeline: pd.DataFrame,
):
    aligned = resolver.align_dataframes(
        market_game_states, market_prices_timeline, "kalshi"
    )
    assert not aligned.empty
    assert "implied_prob" in aligned.columns
    assert "market_price" in aligned.columns
    assert aligned["implied_prob"].notna().all()


def test_align_prices_to_game_states_db(populated_market_db):
    resolver = ContractResolver(db=populated_market_db)
    aligned = resolver.align_prices_to_game_states(
        "2023_01_KC_PHI", "KXNFL-KC-PHI", "kalshi"
    )
    assert len(aligned) == 4
    assert aligned["implied_prob"].between(0, 1).all()


def test_game_clock_to_wall_monotonic(resolver: ContractResolver):
    times = [
        resolver.game_clock_to_wall("2023-01-15", secs, "NFL")
        for secs in [3600, 1800, 0]
    ]
    assert times[0] < times[1] < times[2]
