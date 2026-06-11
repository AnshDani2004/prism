"""
Kalshi REST API pipeline with RSA-PSS request signing.

Docs: https://docs.kalshi.com/getting_started/quick_start_authenticated_requests
"""

from __future__ import annotations

import base64
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import pandas as pd
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.types import PrivateKeyTypes

from config.settings import Settings
from src.data.database import PrismDatabase

logger = logging.getLogger(__name__)


class KalshiAuthError(Exception):
    """Raised when Kalshi credentials are missing or invalid."""


class KalshiAdapter:
    """Kalshi market data adapter with signed REST requests."""

    MARKET_SOURCE = "kalshi"

    def __init__(self, db: PrismDatabase | None = None, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.db = db or PrismDatabase(settings=self.settings)
        self.base_url = self.settings.kalshi_base_url.rstrip("/")
        self._private_key: PrivateKeyTypes | None = None

    @property
    def private_key(self) -> rsa.RSAPrivateKey:
        """Load RSA private key from PEM file."""
        if self._private_key is None:
            key_path = self.settings.kalshi_private_key_path
            if key_path is None or not Path(key_path).exists():
                raise KalshiAuthError(
                    "KALSHI_PRIVATE_KEY_PATH not set or file missing. "
                    "Set credentials in .env to fetch live Kalshi data."
                )
            pem = Path(key_path).read_bytes()
            loaded = serialization.load_pem_private_key(pem, password=None)
            if not isinstance(loaded, rsa.RSAPrivateKey):
                raise KalshiAuthError("Kalshi requires an RSA private key")
            self._private_key = loaded
        return cast(rsa.RSAPrivateKey, self._private_key)

    def _sign_request(self, timestamp_ms: str, method: str, path: str) -> str:
        """RSA-PSS sign timestamp + method + path (no query string)."""
        path_no_query = path.split("?")[0]
        message = f"{timestamp_ms}{method.upper()}{path_no_query}".encode()
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Build authenticated request headers."""
        api_key_id = self.settings.kalshi_api_key_id
        if not api_key_id:
            raise KalshiAuthError("KALSHI_API_KEY_ID not set in environment")
        timestamp_ms = str(int(datetime.now(UTC).timestamp() * 1000))
        full_path = urlparse(self.base_url + path).path
        signature = self._sign_request(timestamp_ms, method, full_path)
        return {
            "KALSHI-ACCESS-KEY": api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        """Execute HTTP request against Kalshi API."""
        url = f"{self.base_url}{path}"
        headers = self._auth_headers(method, path) if authenticated else {}
        response = requests.request(
            method,
            url,
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload

    @staticmethod
    def cents_to_prob(cents: float | int | None) -> float:
        """Convert Kalshi cents quote to [0, 1] probability."""
        if cents is None:
            return 0.5
        return float(cents) / 100.0

    def get_sports_markets(self, sport: str, season: int) -> list[dict[str, Any]]:
        """
        Fetch resolved sports markets for a given sport and season.

        Paginates /markets endpoint and caches to DuckDB contracts table.
        """
        sport_upper = sport.upper()
        contracts: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            params: dict[str, Any] = {"limit": 200, "status": "settled"}
            if cursor:
                params["cursor"] = cursor
            try:
                data = self._request("GET", "/markets", params=params)
            except requests.HTTPError as exc:
                logger.error("Kalshi markets fetch failed: %s", exc)
                break

            markets = data.get("markets", [])
            for market in markets:
                title = (market.get("title") or "").upper()
                if sport_upper not in title and sport_upper not in (
                    market.get("category") or ""
                ).upper():
                    continue
                ticker = market.get("ticker", "")
                result = market.get("result", "")
                resolved = "yes" if result == "yes" else "no" if result == "no" else None
                resolution_price = market.get("last_price")
                if resolution_price is not None:
                    resolution_price = self.cents_to_prob(resolution_price)

                contract = {
                    "contract_id": ticker,
                    "market_source": self.MARKET_SOURCE,
                    "sport": sport_upper,
                    "home_team": "",  # filled by mapping
                    "away_team": "",
                    "game_date": pd.to_datetime(
                        market.get("close_time") or market.get("expiration_time")
                    ).date()
                    if market.get("close_time") or market.get("expiration_time")
                    else None,
                    "contract_type": "moneyline",
                    "resolved_outcome": resolved,
                    "resolution_price": resolution_price,
                }
                contracts.append(contract)

            cursor = data.get("cursor")
            if not cursor or not markets:
                break
            time.sleep(0.2)

        if contracts:
            df = pd.DataFrame(contracts)
            df = df.dropna(subset=["game_date"])
            self.db.upsert_dataframe(
                "contracts", df, primary_key=["contract_id", "market_source"]
            )
        logger.info("Kalshi: loaded %d %s contracts for season %d", len(contracts), sport, season)
        return contracts

    def get_price_history(self, market_ticker: str) -> pd.DataFrame:
        """
        Fetch price time series for a single contract.

        Returns DataFrame: timestamp, yes_bid, yes_ask, yes_mid, volume.
        All prices converted from cents to [0,1] probability scale.
        """
        cached = self.db.query_df(
            """
            SELECT timestamp, yes_price, no_price, yes_bid, yes_ask, volume
            FROM market_prices
            WHERE contract_id = ? AND market_source = ?
            ORDER BY timestamp
            """,
            [market_ticker, self.MARKET_SOURCE],
        )
        if not cached.empty:
            logger.debug("Cache hit for Kalshi prices: %s", market_ticker)
            return cached

        try:
            data = self._request(
                "GET",
                f"/markets/{market_ticker}/candlesticks",
                params={"period_interval": 1},
            )
        except requests.HTTPError:
            logger.warning("No candlestick data for %s", market_ticker)
            return pd.DataFrame()

        candles = data.get("candlesticks", [])
        rows: list[dict[str, Any]] = []
        for candle in candles:
            ts = pd.to_datetime(candle.get("end_period_ts"), unit="s", utc=True)
            yes_bid = self.cents_to_prob(candle.get("yes_bid", {}).get("close"))
            yes_ask = self.cents_to_prob(candle.get("yes_ask", {}).get("close"))
            yes_mid = (yes_bid + yes_ask) / 2
            rows.append(
                {
                    "contract_id": market_ticker,
                    "market_source": self.MARKET_SOURCE,
                    "timestamp": ts,
                    "yes_price": yes_mid,
                    "no_price": 1.0 - yes_mid,
                    "yes_bid": yes_bid,
                    "yes_ask": yes_ask,
                    "volume": candle.get("volume"),
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
