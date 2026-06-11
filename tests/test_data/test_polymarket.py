"""Tests for Polymarket adapter."""

from unittest.mock import patch

import pandas as pd

from src.data.polymarket import PolymarketAdapter


@patch("src.data.polymarket.PolymarketAdapter._get")
def test_get_price_history(mock_get, prism_db):
    mock_get.return_value = {
        "history": [
            {"t": 1705348800, "p": "0.55"},
            {"t": 1705349100, "p": "0.58"},
        ]
    }
    adapter = PolymarketAdapter(db=prism_db)
    result = adapter.get_price_history("token123")
    assert len(result) == 2
    assert (result["yes_price"] >= 0).all() and (result["yes_price"] <= 1).all()


@patch("src.data.polymarket.PolymarketAdapter._get")
def test_get_price_history_uses_cache(mock_get, prism_db, sample_market_prices):
    prices = sample_market_prices.copy()
    prices["market_source"] = "polymarket"
    prices["contract_id"] = "token123"
    prism_db.upsert_dataframe(
        "market_prices",
        prices,
        primary_key=["contract_id", "market_source", "timestamp"],
    )
    adapter = PolymarketAdapter(db=prism_db)
    result = adapter.get_price_history("token123")
    assert len(result) == 5
    mock_get.assert_not_called()


@patch("src.data.polymarket.PolymarketAdapter._get")
def test_get_resolved_sports_markets(mock_get, prism_db):
    mock_get.return_value = [
        {
            "id": "market1",
            "question": "Chiefs vs Eagles",
            "endDate": "2023-01-15T00:00:00Z",
            "outcomes": '["Chiefs", "Eagles"]',
            "clobTokenIds": '["tok1", "tok2"]',
            "umaResolutionStatus": "resolved",
            "outcome": "Chiefs",
        }
    ]
    adapter = PolymarketAdapter(db=prism_db)
    contracts = adapter.get_resolved_sports_markets("NFL")
    assert len(contracts) == 1
    assert contracts[0]["sport"] == "NFL"
