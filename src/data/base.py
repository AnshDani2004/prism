"""Abstract base classes for sport data adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.data.database import PrismDatabase


class SportDataAdapter(ABC):
    """Base class for play-by-play ingestion pipelines."""

    sport: str

    def __init__(self, db: PrismDatabase | None = None) -> None:
        from src.data.database import PrismDatabase as _PrismDatabase

        self.db = db or _PrismDatabase()

    @abstractmethod
    def load_pbp(self, seasons: list[int]) -> pd.DataFrame:
        """Load raw play-by-play data."""

    @abstractmethod
    def extract_game_states(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """Reduce play-by-play to meaningful game state snapshots."""

    @abstractmethod
    def validate_game_states(self, states: pd.DataFrame) -> bool:
        """Validate extracted game states; raise on failure."""

    def ingest(self, seasons: list[int]) -> pd.DataFrame:
        """Full pipeline: load, extract, validate, persist."""
        import logging

        logger = logging.getLogger(__name__)
        logger.info("Ingesting %s seasons %s", self.sport, seasons)

        pbp = self.load_pbp(seasons)
        states = self.extract_game_states(pbp)
        self.validate_game_states(states)
        self.db.upsert_dataframe("game_states", states, primary_key=["game_id", "seconds_remaining"])

        n_games = states["game_id"].nunique()
        n_scoring = int(states["is_scoring_event"].sum())
        logger.info(
            "%s ingest complete: %d games, %d states, %d scoring events",
            self.sport,
            n_games,
            len(states),
            n_scoring,
        )
        return states
