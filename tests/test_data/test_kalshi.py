"""Tests for Kalshi adapter."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.kalshi import KalshiAdapter, KalshiAuthError


def test_cents_to_prob():
    assert KalshiAdapter.cents_to_prob(50) == 0.5
    assert KalshiAdapter.cents_to_prob(0) == 0.0
    assert KalshiAdapter.cents_to_prob(100) == 1.0


def test_auth_raises_without_credentials(prism_db):
    adapter = KalshiAdapter(db=prism_db)
    with pytest.raises(KalshiAuthError):
        adapter._auth_headers("GET", "/markets")  # noqa: SLF001


@patch("src.data.kalshi.requests.request")
def test_get_price_history_caches(mock_request, prism_db, sample_market_prices):
    adapter = KalshiAdapter(db=prism_db)
    prism_db.upsert_dataframe(
        "market_prices",
        sample_market_prices,
        primary_key=["contract_id", "market_source", "timestamp"],
    )
    result = adapter.get_price_history("KXNFL-KC-PHI-0115")
    assert len(result) == 5
    mock_request.assert_not_called()
    assert (result["yes_price"] >= 0).all() and (result["yes_price"] <= 1).all()


@patch("src.data.kalshi.requests.request")
def test_get_price_history_fetches_and_stores(mock_request, prism_db):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "candlesticks": [
            {
                "end_period_ts": 1705348800,
                "yes_bid": {"close": 54},
                "yes_ask": {"close": 56},
                "volume": 100,
            }
        ]
    }
    mock_request.return_value = mock_response

    adapter = KalshiAdapter(db=prism_db)
    adapter.settings.kalshi_api_key_id = "test-key"
    adapter.settings.kalshi_private_key_path = None

    # Mock private key loading
    with patch.object(adapter, "_auth_headers", return_value={}):
        result = adapter.get_price_history("TEST-TICKER")

    assert len(result) == 1
    assert result["yes_price"].iloc[0] == pytest.approx(0.55, abs=0.01)
    assert result["yes_bid"].iloc[0] == pytest.approx(0.54, abs=0.01)
    assert result["yes_ask"].iloc[0] == pytest.approx(0.56, abs=0.01)


@patch("src.data.kalshi.requests.request")
def test_get_sports_markets(mock_request, prism_db):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.side_effect = [
        {
            "markets": [
                {
                    "ticker": "NFL-KC-PHI",
                    "title": "NFL Chiefs vs Eagles",
                    "category": "Sports",
                    "result": "yes",
                    "last_price": 65,
                    "close_time": "2023-01-15T23:00:00Z",
                }
            ],
            "cursor": None,
        }
    ]
    mock_request.return_value = mock_response

    adapter = KalshiAdapter(db=prism_db)
    with patch.object(adapter, "_auth_headers", return_value={}):
        contracts = adapter.get_sports_markets("NFL", 2023)

    assert len(contracts) == 1
    assert contracts[0]["contract_id"] == "NFL-KC-PHI"
    assert prism_db.count("contracts") == 1
