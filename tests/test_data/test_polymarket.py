"""Tests for Polymarket adapter."""
from datetime import date
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from src.data.polymarket import PolymarketAdapter, extract_teams, parse_end_date


def test_extract_teams_nfl():
    home, away = extract_teams("Will the Falcons beat the Panthers?", "NFL")
    assert home == "ATL"
    assert away == "CAR"


def test_extract_teams_nba():
    home, away = extract_teams("Will the Warriors beat the Lakers?", "NBA")
    assert home == "GSW"
    assert away == "LAL"


def test_parse_end_date_valid():
    result = parse_end_date("2023-01-15T00:00:00Z")
    assert result == date(2023, 1, 15)


def test_parse_end_date_none():
    assert parse_end_date(None) is None


def test_get_resolved_sports_markets():
    adapter = PolymarketAdapter()
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {
            "id": "123",
            "title": "Will the Chiefs beat the Eagles?",
            "endDate": "2023-02-12T00:00:00Z",
            "markets": [{"outcomePrices": "[0.9, 0.1]", "outcomes": "[\"Yes\", \"No\"]"}],
        }
    ]
    mock_response.raise_for_status = MagicMock()
    with patch("src.data.polymarket.requests.get", return_value=mock_response):
        # Second call returns empty to stop pagination
        mock_response.json.side_effect = [
            [{"id": "123", "title": "Will the Chiefs beat the Eagles?",
              "endDate": "2023-02-12T00:00:00Z",
              "markets": [{"outcomePrices": "[0.9, 0.1]", "outcomes": "[\"Yes\", \"No\"]"}]}],
            [],
        ]
        events = adapter.get_resolved_sports_markets("NFL")
    assert len(events) == 1
    assert events[0]["id"] == "123"


def test_get_price_history():
    adapter = PolymarketAdapter()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "history": [
            {"t": 1700000000, "p": 0.55},
            {"t": 1700003600, "p": 0.60},
        ]
    }
    with patch("src.data.polymarket.requests.get", return_value=mock_response):
        df = adapter.get_price_history("token_abc")
    assert not df.empty
    assert "yes_price" in df.columns
    assert df["yes_price"].iloc[0] == pytest.approx(0.55)


def test_get_price_history_uses_cache():
    db = MagicMock()
    cached = pd.DataFrame({
        "timestamp": [pd.Timestamp("2023-01-01", tz="UTC")],
        "yes_price": [0.72],
    })
    db.query_df.return_value = cached
    adapter = PolymarketAdapter(db=db)
    with patch("src.data.polymarket.requests.get") as mock_get:
        result = adapter.get_price_history("token_xyz")
        mock_get.assert_not_called()
    assert result["yes_price"].iloc[0] == pytest.approx(0.72)
