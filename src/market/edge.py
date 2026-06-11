"""Edge calculator: model probability vs market implied probability."""

from __future__ import annotations

import logging

import pandas as pd

from src.data.database import PrismDatabase
from src.market.interface import ContractResolver
from src.models.base import WinProbabilityModel
from src.utils.pandas_typing import as_float, as_str

logger = logging.getLogger(__name__)

STALENESS_WINDOW_SECONDS = 30


class EdgeCalculator:
    """
    Computes edge = model_probability - market_implied_probability at each
    aligned game state and persists signals to DuckDB.
    """

    def __init__(
        self,
        db: PrismDatabase | None = None,
        resolver: ContractResolver | None = None,
        staleness_window: int = STALENESS_WINDOW_SECONDS,
    ) -> None:
        self.db = db or PrismDatabase()
        self.resolver = resolver or ContractResolver(db=self.db)
        self.staleness_window = staleness_window

    def _seconds_since_last_score(self, aligned: pd.DataFrame) -> pd.Series:
        """Seconds of game clock elapsed since the previous scoring event."""
        aligned = aligned.sort_values("seconds_remaining", ascending=False)
        last_score_clock: float | None = None
        gaps: list[float] = []
        total = 3600  # default NFL; per-row sport handled below

        for row in aligned.itertuples(index=False):
            total = 3600 if as_str(row.sport).upper() == "NFL" else 2880
            secs = as_float(row.seconds_remaining)
            is_score = bool(getattr(row, "is_scoring_event", False))
            if is_score:
                gap = 0.0
                last_score_clock = secs
            else:
                gap = total if last_score_clock is None else last_score_clock - secs
            gaps.append(max(0.0, gap))

        sorted_idx = aligned.sort_values("seconds_remaining", ascending=False).index
        return pd.Series(gaps, index=sorted_idx)

    def classify_edge_type(self, edge_row: pd.Series) -> str:
        """
        Classify edge observation type.

        - pre_game: before game starts (full clock remaining)
        - inplay_staleness: within staleness window after last score
        - inplay_drift: 30s+ after last score
        - cross_market: reserved for Kalshi vs Polymarket disagreement (tagged externally)
        """
        if edge_row.get("edge_type") == "cross_market":
            return "cross_market"

        sport = str(edge_row.get("sport", "NFL")).upper()
        total = 3600 if sport == "NFL" else 2880
        secs = float(edge_row.get("seconds_remaining", 0))
        if secs >= total - 60:
            return "pre_game"

        since_score = float(edge_row.get("seconds_since_last_score", float("inf")))
        if since_score <= self.staleness_window:
            return "inplay_staleness"
        return "inplay_drift"

    def compute_edge_series(
        self,
        game_id: str,
        model: WinProbabilityModel,
        market_source: str,
        contract_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Compute edge time series for a matched (game, contract) pair.

        Returns DataFrame with model_prob, market_price, edge, is_scoring_event,
        and seconds_since_last_score.
        """
        if contract_id is None:
            mapping = self.db.query_df(
                """
                SELECT contract_id FROM game_contract_map
                WHERE game_id = ? AND market_source = ?
                ORDER BY match_confidence DESC LIMIT 1
                """,
                [game_id, market_source],
            )
            if mapping.empty:
                logger.warning("No contract mapping for game %s source %s", game_id, market_source)
                return pd.DataFrame()
            contract_id = str(mapping["contract_id"].iloc[0])

        aligned = self.resolver.align_prices_to_game_states(game_id, contract_id, market_source)
        if aligned.empty:
            return pd.DataFrame()

        model_probs = model.predict(aligned)
        aligned = aligned.copy()
        aligned["model_prob"] = model_probs
        aligned["market_price"] = aligned["implied_prob"]
        aligned["edge"] = aligned["model_prob"] - aligned["market_price"]
        aligned["seconds_since_last_score"] = self._seconds_since_last_score(aligned)
        aligned["edge_type"] = aligned.apply(self.classify_edge_type, axis=1)
        aligned["contract_id"] = contract_id
        aligned["market_source"] = market_source
        aligned["game_id"] = game_id

        return aligned[
            [
                "game_id",
                "contract_id",
                "market_source",
                "seconds_remaining",
                "sport",
                "game_date",
                "model_prob",
                "market_price",
                "edge",
                "edge_type",
                "is_scoring_event",
                "seconds_since_last_score",
            ]
        ]

    def compute_edge_from_aligned(
        self,
        aligned: pd.DataFrame,
        model: WinProbabilityModel,
        game_id: str,
        contract_id: str,
        market_source: str,
    ) -> pd.DataFrame:
        """Compute edge from pre-aligned states+prices (for testing)."""
        if aligned.empty:
            return pd.DataFrame()

        aligned = aligned.copy()
        aligned["model_prob"] = model.predict(aligned)
        aligned["market_price"] = aligned["implied_prob"]
        aligned["edge"] = aligned["model_prob"] - aligned["market_price"]
        aligned["seconds_since_last_score"] = self._seconds_since_last_score(aligned)
        aligned["edge_type"] = aligned.apply(self.classify_edge_type, axis=1)
        aligned["game_id"] = game_id
        aligned["contract_id"] = contract_id
        aligned["market_source"] = market_source
        return aligned

    def compute_all_edges(
        self,
        model: WinProbabilityModel,
        market_source: str,
        min_confidence: float = 0.8,
    ) -> pd.DataFrame:
        """Compute edges for all high-confidence game-contract mappings."""
        mappings = self.db.query_df(
            "SELECT * FROM game_contract_map WHERE match_confidence >= ?",
            [min_confidence],
        )
        mappings = mappings[mappings["market_source"] == market_source]
        frames: list[pd.DataFrame] = []

        for _, row in mappings.iterrows():
            series = self.compute_edge_series(
                str(row["game_id"]),
                model,
                market_source,
                contract_id=str(row["contract_id"]),
            )
            if not series.empty:
                frames.append(series)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        self.persist_edge_signals(result)
        logger.info("Computed %d edge observations for %s", len(result), market_source)
        return result

    def persist_edge_signals(self, edges: pd.DataFrame) -> int:
        """Write edge signals to DuckDB."""
        if edges.empty:
            return 0
        persist = edges[
            [
                "game_id",
                "contract_id",
                "market_source",
                "seconds_remaining",
                "model_prob",
                "market_price",
                "edge",
                "edge_type",
            ]
        ].copy()
        return self.db.insert_dataframe("edge_signals", persist)

    def cross_market_edges(
        self,
        game_id: str,
        model: WinProbabilityModel,
    ) -> pd.DataFrame:
        """
        Compare Kalshi and Polymarket prices for the same game.

        Tags rows where sources disagree as cross_market edge type.
        """
        frames: list[pd.DataFrame] = []
        for source in ("kalshi", "polymarket"):
            series = self.compute_edge_series(game_id, model, source)
            if not series.empty:
                frames.append(series)

        if len(frames) < 2:
            return pd.DataFrame()

        kalshi = frames[0].set_index("seconds_remaining")
        poly = frames[1].set_index("seconds_remaining")
        common = kalshi.index.intersection(poly.index)
        if len(common) == 0:
            return pd.DataFrame()

        cross = kalshi.loc[common].copy()
        cross["market_price_kalshi"] = kalshi.loc[common, "market_price"]
        cross["market_price_poly"] = poly.loc[common, "market_price"]
        cross["edge"] = cross["market_price_kalshi"] - cross["market_price_poly"]
        cross["edge_type"] = "cross_market"
        cross["model_prob"] = model.predict(cross.reset_index())
        return cross.reset_index()
