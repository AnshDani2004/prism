"""
Polymarket historical data pipeline via Gamma + CLOB APIs.

Gamma API: market metadata (https://gamma-api.polymarket.com)
CLOB API: price history (https://clob.polymarket.com/prices-history)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import pandas as pd
import requests

from src.data.database import PrismDatabase

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


class PolymarketAdapter:
    """Polymarket market discovery and price history adapter."""

    MARKET_SOURCE = "polymarket"

    SPORT_TAGS: dict[str, str] = {
        "NFL": "nfl",
        "NBA": "nba",
    }

    def __init__(self, db: PrismDatabase | None = None, request_delay: float = 0.3):
        self.db = db or PrismDatabase()
        self.request_delay = request_delay
        self.session = requests.Session()

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_resolved_sports_markets(self, sport: str) -> list[dict[str, Any]]:
        """
        Query Gamma API for resolved sports markets.

        Paginates with limit/offset. Caches contract metadata to DuckDB.
        """
        tag = self.SPORT_TAGS.get(sport.upper(), sport.lower())
        contracts: list[dict[str, Any]] = []
        offset = 0
        limit = 100

        while True:
            params = {
                "closed": "true",
                "tag": tag,
                "limit": limit,
                "offset": offset,
            }
            try:
                markets = self._get(f"{GAMMA_API}/markets", params=params)
            except requests.HTTPError as exc:
                logger.error("Polymarket markets fetch failed: %s", exc)
                break

            if not markets:
                break

            for market in markets:
                question = market.get("question", "")
                outcomes = market.get("outcomes")
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)
                tokens = market.get("clobTokenIds")
                if isinstance(tokens, str):
                    tokens = json.loads(tokens)

                resolved_outcome = None
                resolution_price = None
                if market.get("umaResolutionStatus") == "resolved":
                    winner = market.get("outcome")
                    if winner:
                        resolved_outcome = str(winner).lower()

                end_date = market.get("endDate") or market.get("closedTime")
                contract = {
                    "contract_id": str(market.get("id", market.get("conditionId", ""))),
                    "market_source": self.MARKET_SOURCE,
                    "sport": sport.upper(),
                    "home_team": "",
                    "away_team": "",
                    "game_date": pd.to_datetime(end_date).date() if end_date else None,
                    "contract_type": "moneyline",
                    "resolved_outcome": resolved_outcome,
                    "resolution_price": resolution_price,
                    "_question": question,
                    "_token_ids": tokens,
                }
                contracts.append(contract)

            if len(markets) < limit:
                break
            offset += limit
            time.sleep(self.request_delay)

        persist = [
            {k: v for k, v in c.items() if not k.startswith("_")}
            for c in contracts
            if c.get("game_date") is not None
        ]
        if persist:
            df = pd.DataFrame(persist)
            self.db.upsert_dataframe(
                "contracts", df, primary_key=["contract_id", "market_source"]
            )
        logger.info("Polymarket: loaded %d resolved %s markets", len(contracts), sport)
        return contracts

    def get_price_history(self, token_id: str) -> pd.DataFrame:
        """
        Query CLOB API prices-history for an outcome token.

        Returns DataFrame: timestamp, price (in [0,1]).
        Caches to DuckDB immediately; only fetches each token once.
        """
        cached = self.db.query_df(
            """
            SELECT timestamp, yes_price, no_price, volume
            FROM market_prices
            WHERE contract_id = ? AND market_source = ?
            ORDER BY timestamp
            """,
            [token_id, self.MARKET_SOURCE],
        )
        if not cached.empty:
            return cached

        try:
            data = self._get(
                f"{CLOB_API}/prices-history",
                params={"market": token_id, "interval": "max"},
            )
        except requests.HTTPError:
            logger.warning("No price history for token %s", token_id)
            return pd.DataFrame()

        history = data.get("history", [])
        rows: list[dict[str, Any]] = []
        for point in history:
            ts = pd.to_datetime(point["t"], unit="s", utc=True)
            price = float(point["p"])
            rows.append(
                {
                    "contract_id": token_id,
                    "market_source": self.MARKET_SOURCE,
                    "timestamp": ts,
                    "yes_price": price,
                    "no_price": 1.0 - price,
                    "yes_bid": None,
                    "yes_ask": None,
                    "volume": None,
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            self.db.upsert_dataframe(
                "market_prices",
                df,
                primary_key=["contract_id", "market_source", "timestamp"],
            )
        return df
