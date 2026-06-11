"""Contract resolver and game-clock to wall-clock price alignment."""

from __future__ import annotations

import logging
from datetime import datetime, time

import numpy as np
import pandas as pd

from src.data.database import PrismDatabase

logger = logging.getLogger(__name__)

SPORT_TOTAL_SECONDS = {"NFL": 3600, "NBA": 2880}
DEFAULT_KICKOFF = time(18, 0)  # nominal UTC kickoff for alignment


class ContractResolver:
    """
    Aligns game state timestamps to market price timestamps.

    Game states are indexed by seconds_remaining (game clock). Market prices
    use wall-clock timestamps. We bridge the two by:
      1. Estimating wall-clock kickoff from game_date
      2. Mapping elapsed game time to wall clock
      3. Using scoring events as anchor refinements
      4. Asof-joining market prices to each game state
    """

    def __init__(
        self,
        db: PrismDatabase | None = None,
        kickoff_time: time = DEFAULT_KICKOFF,
    ) -> None:
        self.db = db or PrismDatabase()
        self.kickoff_time = kickoff_time

    def _total_seconds(self, sport: str) -> int:
        return SPORT_TOTAL_SECONDS.get(sport.upper(), 3600)

    def game_clock_to_wall(
        self,
        game_date: pd.Timestamp | datetime,
        seconds_remaining: int | float,
        sport: str,
    ) -> pd.Timestamp:
        """Convert game clock to estimated wall-clock timestamp."""
        total = self._total_seconds(sport)
        elapsed = total - int(seconds_remaining)
        base = pd.Timestamp(game_date).normalize() + pd.Timedelta(
            hours=self.kickoff_time.hour,
            minutes=self.kickoff_time.minute,
        )
        return base + pd.Timedelta(seconds=elapsed)

    def compute_implied_probability(
        self,
        price_row: pd.Series | dict[str, float | None],
        market_source: str,
    ) -> float:
        """
        Convert market quote to implied home-win probability in [0, 1].

        Kalshi: mid of yes_bid/yes_ask when available, else yes_price.
        Polymarket: yes_price (already in [0, 1]).
        Sportsbook: remove vig via proportional normalization (caller passes
        home/away raw implied probs as yes_price/no_price).
        """
        source = market_source.lower()
        yes_price = _get_float(price_row, "yes_price")
        yes_bid = _get_float(price_row, "yes_bid")
        yes_ask = _get_float(price_row, "yes_ask")
        no_price = _get_float(price_row, "no_price")

        if source == "kalshi":
            if yes_bid is not None and yes_ask is not None:
                return float(np.clip((yes_bid + yes_ask) / 2.0, 0.0, 1.0))
            return float(np.clip(yes_price if yes_price is not None else 0.5, 0.0, 1.0))

        if source == "polymarket":
            return float(np.clip(yes_price if yes_price is not None else 0.5, 0.0, 1.0))

        if source in {"sportsbook", "odds"}:
            # Proportional vig removal: normalize raw implied probabilities
            q_home = yes_price if yes_price is not None else 0.5
            q_away = no_price if no_price is not None else 0.5
            total = q_home + q_away
            if total <= 0:
                return 0.5
            return float(np.clip(q_home / total, 0.0, 1.0))

        return float(np.clip(yes_price if yes_price is not None else 0.5, 0.0, 1.0))

    def load_game_states(self, game_id: str) -> pd.DataFrame:
        return self.db.query_df(
            "SELECT * FROM game_states WHERE game_id = ? ORDER BY seconds_remaining DESC",
            [game_id],
        )

    def load_market_prices(self, contract_id: str, market_source: str) -> pd.DataFrame:
        prices = self.db.query_df(
            """
            SELECT * FROM market_prices
            WHERE contract_id = ? AND market_source = ?
            ORDER BY timestamp
            """,
            [contract_id, market_source],
        )
        if not prices.empty:
            prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
        return prices

    def align_prices_to_game_states(
        self,
        game_id: str,
        contract_id: str,
        market_source: str,
    ) -> pd.DataFrame:
        """
        Merge game states with nearest market price at each state's wall-clock time.

        Returns DataFrame with game state columns plus market_price, yes_bid,
        yes_ask, market_timestamp, and implied_prob.
        """
        states = self.load_game_states(game_id)
        prices = self.load_market_prices(contract_id, market_source)

        if states.empty or prices.empty:
            logger.warning("Empty states or prices for game=%s contract=%s", game_id, contract_id)
            return pd.DataFrame()

        sport = str(states["sport"].iloc[0])
        states = states.copy()
        states["wall_clock"] = states.apply(
            lambda r: self.game_clock_to_wall(r["game_date"], r["seconds_remaining"], sport),
            axis=1,
        )
        states["wall_clock"] = pd.to_datetime(states["wall_clock"], utc=True)

        # Refine anchors at scoring events: snap to nearest market tick
        scoring_mask = states["is_scoring_event"].fillna(False)
        if scoring_mask.any():
            for idx in states.index[scoring_mask]:
                wc = states.at[idx, "wall_clock"]
                nearest_idx = (prices["timestamp"] - wc).abs().idxmin()
                states.at[idx, "wall_clock"] = prices.at[nearest_idx, "timestamp"]

        prices = prices.sort_values("timestamp")
        states = states.sort_values("wall_clock")

        merged = pd.merge_asof(
            states,
            prices,
            left_on="wall_clock",
            right_on="timestamp",
            direction="backward",
        )
        merged["implied_prob"] = merged.apply(
            lambda r: self.compute_implied_probability(r, market_source),
            axis=1,
        )
        merged["market_price"] = merged["implied_prob"]
        return merged

    def align_dataframes(
        self,
        states: pd.DataFrame,
        prices: pd.DataFrame,
        market_source: str,
        sport: str | None = None,
    ) -> pd.DataFrame:
        """Align pre-loaded states and prices (for testing without DB)."""
        if states.empty or prices.empty:
            return pd.DataFrame()

        sport = sport or str(states["sport"].iloc[0])
        states = states.copy()
        prices = prices.copy()
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)

        states["wall_clock"] = states.apply(
            lambda r: self.game_clock_to_wall(r["game_date"], r["seconds_remaining"], sport),
            axis=1,
        )
        states["wall_clock"] = pd.to_datetime(states["wall_clock"], utc=True)
        states = states.sort_values("wall_clock")
        prices = prices.sort_values("timestamp")

        merged = pd.merge_asof(
            states,
            prices,
            left_on="wall_clock",
            right_on="timestamp",
            direction="backward",
        )
        merged["implied_prob"] = merged.apply(
            lambda r: self.compute_implied_probability(r, market_source),
            axis=1,
        )
        merged["market_price"] = merged["implied_prob"]
        return merged


def _get_float(row: pd.Series | dict[str, float | None], key: str) -> float | None:
    if isinstance(row, dict):
        val = row.get(key)
    else:
        val = row.get(key) if key in row.index else None
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return float(val)
