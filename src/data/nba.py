"""
NBA game-state pipeline using nba_api LeagueGameLog (bulk, per season).

PlayByPlay endpoint is too fragile at scale (8000+ individual API calls).
Instead we use LeagueGameLog to get final scores per game, then simulate
game states using a score-interpolation model seeded from final scores.
This gives us clean pre-game and final-state rows for every game.

DECISIONS.md: PlayByPlay endpoint rejected — too slow and unreliable at scale.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from src.data.base import SportDataAdapter

if TYPE_CHECKING:
    from src.data.database import PrismDatabase
from src.utils.validation import assert_no_duplicates

logger = logging.getLogger(__name__)


def period_seconds_remaining(period: int, clock_secs: int) -> int:
    """Convert period + period clock to total sonds remaining."""
    reg_length = 12 * 60
    ot_length = 5 * 60
    periods_after = max(0, 4 - period) if period <= 4 else 0
    ot_periods_after = max(0, period - 4 - 1) if period > 4 else 0
    return clock_secs + periods_after * reg_length + ot_periods_after * ot_length


class NBAAdapter(SportDataAdapter):
    """NBA adapter using LeagueGameLog for bulk per-season data."""

    sport = "NBA"
    MAX_REGULATION_POINTS: ClassVar[int] = 160
    REQUEST_DELAY: ClassVar[float] = 1.0

    def __init__(self, db: PrismDatabase | None = None, request_delay: float = 1.0) -> None:
        super().__init__(db)
        self.request_delay = request_delay

    def load_pbp(self, seasons: list[int]) -> pd.DataFrame:
        """
        Load NBA game log for given seasons via LeagueGameLog (bulk endpoint).
        Returns one row per team per game — we deduplicate to one row per game below.
        """
        frames: list[pd.DataFrame] = []
        for season in seasons:
            logger.info("Loading NBA season %d", season)
            season_str = f"{season - 1}-{str(season)[-2:]}"
            try:
                log = leaguegamelog.LeagueGameLog(
                    season=season_str,
                    league_id="00",
                    season_type_all_star="Regular Season",
                )
                df = log.get_data_frames()[0]
                if df.empty:
                    logger.warning("No games returned for NBA season %d", season)
                    continue
                df["season"] = season
                frames.append(df)
                logger.info("Found %d NBA game rows for season %d", len(df), season)
            except Exception as exc:
                logger.warning("Failed to load NBA season %d: %s", season, exc)
            time.sleep(self.request_delay)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def extract_game_states(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """
        Convert LeagueGameLog rows into game states.

        LeagueGameLog has one row per team per game. We pair home/away rows
        to produce one game state at t=0 (pre-game) and t=final for each game.
        Additional interpolated states are generated at quarter boundaries
        using a linear score interpolation from 0 to final score.
        """
        if pbp.empty:
            return pd.DataFrame()

        # Deduplicate: LeagueGameLog returns both teams per game
        # Identify home team via MATCHUP column: "CHI vs. MIA" = home
        pbp = pbp.copy()
        pbp["is_home"] = pbp["MATCHUP"].str.contains("vs\\.", regex=True)

        home_df = pbp[pbp["is_home"]].copy()
        away_df = pbp[~pbp["is_home"]].copy()

        home_df = home_df.rename(columns={
            "TEAM_ABBREVIATION": "home_team",
            "PTS": "home_pts",
            "GAME_DATE": "game_date",
            "GAME_ID": "game_id",
        })
        away_df = away_df.rename(columns={
            "TEAM_ABBREVIATION": "away_team",
            "PTS": "away_pts",
            "GAME_ID": "game_id_away",
        })

        merged = home_df.merge(
            away_df[["game_id_away", "away_team", "away_pts"]],
            left_on="game_id",
            right_on="game_id_away",
            how="inner",
        )

        records: list[dict[str, object]] = []
        reg_total = 4 * 12 * 60  # 2880 seconds

        for _, row in merged.iterrows():
            game_id = str(row["game_id"])
            season = int(row["season"])
            try:
                game_date = pd.to_datetime(row["game_date"]).date()
            except Exception:
                continue
            home_team = str(row["home_team"])
            away_team = str(row["away_team"])
            home_final = int(row["home_pts"]) if pd.notna(row["home_pts"]) else 0
            away_final = int(row["away_pts"]) if pd.notna(row["away_pts"]) else 0

            # Anomaly check
            if home_final > self.MAX_REGULATION_POINTS or away_final > self.MAX_REGULATION_POINTS:
                logger.debug("High-scoring game flagged: %s (%d-%d)", game_id, home_final, away_final)

            # State at game start (t=0)
            records.append({
                "game_id": game_id,
                "sport": self.sport,
                "season": season,
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "seconds_remaining": reg_total,
                "game_period": 1,
                "score_differential": 0,
                "home_score": 0,
                "away_score": 0,
                "possession": None,
                "is_scoring_event": False,
                "event_type": None,
            })

            # Interpolated states at quarter boundaries (Q1 end, Q2 end, Q3 end)
            for q in range(1, 4):
                frac = q / 4.0
                h = round(home_final * frac)
                a = round(away_final * frac)
                secs = reg_total - q * 12 * 60
                records.append({
                    "game_id": game_id,
                    "sport": self.sport,
                    "season": season,
                    "game_date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "seconds_remaining": secs,
                    "game_period": q + 1,
                    "score_differential": h - a,
                    "home_score": h,
                    "away_score": a,
                    "possession": None,
                    "is_scoring_event": False,
                    "event_type": None,
                })

            # Final state (t=end)
            records.append({
                "game_id": game_id,
                "sport": self.sport,
                "season": season,
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "seconds_remaining": 0,
                "game_period": 4,
                "score_differential": home_final - away_final,
                "home_score": home_final,
                "away_score": away_final,
                "possession": None,
                "is_scoring_event": True,
                "event_type": "final",
            })

        states = pd.DataFrame(records)
        if states.empty:
            return states

        states = states.drop_duplicates(subset=["game_id", "seconds_remaining"], keep="last")
        states = states.sort_values(
            ["game_id", "seconds_remaining"], ascending=[True, False]
        ).reset_index(drop=True)
        return states

    def validate_game_states(self, states: pd.DataFrame) -> bool:
        """Validate NBA game states."""
        if states.empty:
            logger.warning("No NBA game states to validate")
            return True

        inconsistent = states["score_differential"] != (
            states["home_score"] - states["away_score"]
        )
        if inconsistent.any():
            n = int(inconsistent.sum())
            raise ValueError(
                f"NBA validation failed: {n} rows with inconsistent score differential"
            )

        for game_id, group in states.groupby("game_id"):
            secs = group["seconds_remaining"].to_numpy(dtype=np.int64)
            if len(secs) > 1 and not np.all(np.diff(secs) <= 0):
                raise ValueError(
                    f"NBA validation failed: seconds_remaining not monotonic in game {game_id}"
                )

        assert_no_duplicates(states, ["game_id", "seconds_remaining"], "NBA game states")
        logger.info("NBA game states passed all validation checks (%d rows)", len(states))
        return True
