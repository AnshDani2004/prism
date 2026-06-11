"""
NFL play-by-play data pipeline using nfl_data_py.

Pulls seasons 2018-2023, extracts meaningful game state changes,
and loads into DuckDB game_states table.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import nfl_data_py as nfl
import numpy as np
import pandas as pd

from src.data.base import SportDataAdapter
from src.utils.validation import assert_no_duplicates

logger = logging.getLogger(__name__)


class NFLAdapter(SportDataAdapter):
    """NFL play-by-play adapter with scoring-event and quarter-start extraction."""

    sport = "NFL"

    SCORING_PLAY_TYPES: ClassVar[set[str]] = {
        "touchdown",
        "field_goal",
        "extra_point",
        "two_point_conversion",
        "safety",
    }

    PBP_COLUMNS: ClassVar[list[str]] = [
        "game_id",
        "season",
        "game_date",
        "home_team",
        "away_team",
        "qtr",
        "game_seconds_remaining",
        "score_differential",
        "posteam",
        "home_score",
        "away_score",
        "play_type",
        "touchdown",
        "field_goal_result",
        "extra_point_result",
        "two_point_conv_result",
        "safety",
    ]

    def load_pbp(self, seasons: list[int]) -> pd.DataFrame:
        """Load raw play-by-play. Returns DataFrame with all raw columns."""
        logger.info("Loading NFL play-by-play for seasons %s", seasons)
        pbp = nfl.import_pbp_data(seasons, columns=self.PBP_COLUMNS, downcast=True)
        pbp = pbp.dropna(subset=["game_seconds_remaining", "score_differential"]).copy()
        pbp["home_score"] = pbp["home_score"].fillna(0).astype(int)
        pbp["away_score"] = pbp["away_score"].fillna(0).astype(int)
        pbp["score_differential"] = pbp["home_score"] - pbp["away_score"]
        return pd.DataFrame(pbp)

    def _is_scoring_play(self, row: pd.Series) -> bool:
        """Determine if a play is a scoring event."""
        play_type = str(row.get("play_type", "") or "").lower()
        if play_type in self.SCORING_PLAY_TYPES:
            return True
        if row.get("touchdown") == 1:
            return True
        if row.get("safety") == 1:
            return True
        fg = row.get("field_goal_result")
        if fg is not None and str(fg).lower() == "made":
            return True
        ep = row.get("extra_point_result")
        if ep is not None and str(ep).lower() == "good":
            return True
        tpc = row.get("two_point_conv_result")
        if tpc is not None and str(tpc).lower() == "success":
            return True
        return False

    def _event_type(self, row: pd.Series) -> str | None:
        """Map play to a normalized event type string."""
        if not self._is_scoring_play(row):
            return None
        play_type = str(row.get("play_type", "") or "").lower()
        if row.get("touchdown") == 1 or play_type == "touchdown":
            return "touchdown"
        if play_type == "field_goal" or (
            row.get("field_goal_result") is not None
            and str(row["field_goal_result"]).lower() == "made"
        ):
            return "field_goal"
        if play_type == "extra_point" or (
            row.get("extra_point_result") is not None
            and str(row["extra_point_result"]).lower() == "good"
        ):
            return "extra_point"
        if play_type == "two_point_conversion" or (
            row.get("two_point_conv_result") is not None
            and str(row["two_point_conv_result"]).lower() == "success"
        ):
            return "two_point_conversion"
        if row.get("safety") == 1 or play_type == "safety":
            return "safety"
        return play_type or "scoring"

    def _row_to_state(self, row: pd.Series, is_scoring: bool) -> dict[str, object]:
        """Convert a play-by-play row to a game_states record."""
        return {
            "game_id": row["game_id"],
            "sport": self.sport,
            "season": int(row["season"]),
            "game_date": pd.to_datetime(row["game_date"]).date(),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "seconds_remaining": int(row["game_seconds_remaining"]),
            "game_period": int(row["qtr"]),
            "score_differential": int(row["score_differential"]),
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
            "possession": row.get("posteam") if pd.notna(row.get("posteam")) else None,
            "is_scoring_event": is_scoring,
            "event_type": self._event_type(row) if is_scoring else None,
        }

    def extract_game_states(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce play-by-play to meaningful state changes only.

        A meaningful state change is any scoring event plus the state at
        the start of each quarter. This reduces ~250 plays per game to
        ~20-30 states relevant for win probability modeling.
        """
        if pbp.empty:
            return pd.DataFrame()

        pbp = pbp.sort_values(["game_id", "game_seconds_remaining"], ascending=[True, False])
        pbp["is_scoring"] = pbp.apply(self._is_scoring_play, axis=1)

        # Quarter-start states: first play of each quarter per game
        quarter_starts = (
            pbp.groupby(["game_id", "qtr"], as_index=False)
            .first()
            .assign(is_scoring=lambda df: False)
        )

        scoring_plays = pbp[pbp["is_scoring"]].copy()

        combined = pd.concat([quarter_starts, scoring_plays], ignore_index=True)
        records: list[dict[str, object]] = []
        for _, row in combined.iterrows():
            is_scoring = bool(row.get("is_scoring", False))
            state = self._row_to_state(row, is_scoring)
            records.append(state)

        states = pd.DataFrame(records)
        states = states.drop_duplicates(subset=["game_id", "seconds_remaining"], keep="last")
        states = states.sort_values(
            ["game_id", "seconds_remaining"], ascending=[True, False]
        ).reset_index(drop=True)
        return states

    def validate_game_states(self, states: pd.DataFrame) -> bool:
        """
        Validation checks:
        1. score_differential = home_score - away_score always
        2. seconds_remaining decreases monotonically within each game
        3. No negative scores
        4. game_period in [1, 2, 3, 4, 5] (5 = OT)
        5. No duplicate (game_id, seconds_remaining) pairs
        """
        if states.empty:
            logger.warning("No NFL game states to validate")
            return True

        if (states["home_score"] < 0).any() or (states["away_score"] < 0).any():
            raise ValueError("NFL validation failed: negative scores found")

        assert_no_duplicates(states, ["game_id", "seconds_remaining"], "NFL game states")

        inconsistent = states["score_differential"] != (
            states["home_score"] - states["away_score"]
        )
        if inconsistent.any():
            n = int(inconsistent.sum())
            raise ValueError(
                f"NFL validation failed: {n} rows with inconsistent score differential"
            )

        for game_id, group in states.groupby("game_id"):
            secs = group["seconds_remaining"].to_numpy(dtype=np.int64)
            if len(secs) > 1 and not np.all(np.diff(secs) <= 0):
                raise ValueError(
                    f"NFL validation failed: seconds_remaining not monotonic in game {game_id}"
                )

        valid_periods = states["game_period"].between(1, 5)
        if not valid_periods.all():
            bad = states.loc[~valid_periods, "game_period"].unique()
            raise ValueError(f"NFL validation failed: invalid periods {bad}")

        logger.info("NFL game states passed all validation checks (%d rows)", len(states))
        return True
