"""Tests for edge calculator."""

import numpy as np
import pandas as pd

from src.market.edge import EdgeCalculator
from src.market.interface import ContractResolver
from tests.test_market.conftest import FixedProbModel


def test_edge_zero_for_efficient_market(
    resolver: ContractResolver,
    market_game_states: pd.DataFrame,
    market_prices_timeline: pd.DataFrame,
    mirror_model,
):
    aligned = resolver.align_dataframes(market_game_states, market_prices_timeline, "kalshi")
    calc = EdgeCalculator()
    edges = calc.compute_edge_from_aligned(
        aligned,
        mirror_model,
        game_id="2023_01_KC_PHI",
        contract_id="KXNFL-KC-PHI",
        market_source="kalshi",
    )
    assert abs(edges["edge"].mean()) < 0.01


def test_classify_edge_type_pre_game(edge_calc: EdgeCalculator):
    row = pd.Series({"seconds_remaining": 3600, "seconds_since_last_score": 0, "sport": "NFL"})
    assert edge_calc.classify_edge_type(row) == "pre_game"


def test_classify_edge_type_staleness(edge_calc: EdgeCalculator):
    row = pd.Series({"seconds_remaining": 1800, "seconds_since_last_score": 10, "sport": "NFL"})
    assert edge_calc.classify_edge_type(row) == "inplay_staleness"


def test_classify_edge_type_drift(edge_calc: EdgeCalculator):
    row = pd.Series({"seconds_remaining": 900, "seconds_since_last_score": 120, "sport": "NFL"})
    assert edge_calc.classify_edge_type(row) == "inplay_drift"


def test_compute_edge_series_db(populated_market_db, fixed_model: FixedProbModel):
    calc = EdgeCalculator(db=populated_market_db)
    edges = calc.compute_edge_series(
        "2023_01_KC_PHI",
        fixed_model,
        "kalshi",
        contract_id="KXNFL-KC-PHI",
    )
    assert not edges.empty
    assert "edge" in edges.columns
    assert "seconds_since_last_score" in edges.columns
    assert edges["edge"].notna().all()


def test_persist_edge_signals(populated_market_db, fixed_model: FixedProbModel):
    calc = EdgeCalculator(db=populated_market_db)
    edges = calc.compute_edge_series(
        "2023_01_KC_PHI", fixed_model, "kalshi", contract_id="KXNFL-KC-PHI"
    )
    n = calc.persist_edge_signals(edges)
    assert n == len(edges)
    assert populated_market_db.count("edge_signals") == len(edges)


def test_seconds_since_last_score(edge_calc: EdgeCalculator, market_game_states, market_prices_timeline):
    resolver = ContractResolver()
    aligned = resolver.align_dataframes(market_game_states, market_prices_timeline, "kalshi")
    gaps = edge_calc._seconds_since_last_score(aligned)
    assert (gaps >= 0).all()
    scoring_rows = aligned[aligned["is_scoring_event"]]
    for idx in scoring_rows.index:
        assert gaps.loc[idx] == 0.0
