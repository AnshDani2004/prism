"""Tests for adverse selection detector."""

import pandas as pd

from src.market.adverse_selection import AdverseSelectionDetector


def test_correction_latency_positive(populated_market_db):
    detector = AdverseSelectionDetector(db=populated_market_db)
    event_ts = pd.Timestamp("2023-01-15 18:09:00", tz="UTC")
    latency = detector.compute_price_impact(
        "KXNFL-KC-PHI", event_ts, market_source="kalshi", window_seconds=60
    )
    assert abs(latency) > 0.01


def test_correction_latency_distribution(populated_market_db):
    detector = AdverseSelectionDetector(db=populated_market_db)
    dist = detector.correction_latency_distribution(
        "2023_01_KC_PHI", "KXNFL-KC-PHI", "kalshi"
    )
    assert not dist.empty
    assert "correction_latency_seconds" in dist.columns
    assert (dist["correction_latency_seconds"] > 0).any()


def test_edge_realizability(populated_market_db, fixed_model):
    from src.market.edge import EdgeCalculator

    calc = EdgeCalculator(db=populated_market_db)
    edges = calc.compute_edge_series(
        "2023_01_KC_PHI", fixed_model, "kalshi", contract_id="KXNFL-KC-PHI"
    )
    detector = AdverseSelectionDetector(db=populated_market_db)
    result = detector.test_edge_realizability(
        edges, "KXNFL-KC-PHI", "kalshi", entry_delay_seconds=5.0
    )
    assert "realizable" in result.columns
    assert "delayed_edge" in result.columns
    summary = detector.summarize_realizability(result)
    assert summary["n_signals"] > 0


def test_summarize_realizability_empty():
    detector = AdverseSelectionDetector()
    summary = detector.summarize_realizability(pd.DataFrame())
    assert summary["n_signals"] == 0
