"""
NBA play-by-play data pipeline using nba_api.

Extracts game states at scoring events and at least every 2 minutes of game clock.
"""

from __future__ import annotations

import logging
import re
import time
from typing import ClassVar

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playbyplay

from typing import TYPE_CHECKING

from src.data.base import SportDataAdapter

if TYPE_CHECKING:
    from src.data.database import PrismDatabase
from src.utils.validation import assert_no_duplicates

logger = logging.getLogger(__name__)

CLOCK_PATTERN = re.compile(r"^(\d{1,2}):(\d{2})$")


def parse_game_clock(clock_str: str | None) -> int | None:
    """Parse NBA game clock 'M:SS' to seconds remaining in period."""
    if clock_str is None or (isinstance(clock_str, float) and np.isnan(clock_str)):
        return None
    text = str(clock_str).strip()
    if text in {"", "0.0", "0:00"}:
        return 0
    match = CLOCK_PATTERN.match(text)
    if not match:
        return None
    minutes, seconds = int(match.group(1)), int(match.group(2))
    return minutes * 60 + seconds


def period_seconds_remaining(period: int, clock_secs: int) -> int:
    """Convert period + period clock to total seconds remaining in regulation/OT."""
    period_length = 12 * 60 if period <= 4 else 5 * 60
    periods_after = max(0, 4 - period) if period <= 4 else 0
    ot_periods_after = max(0, period - 4 - 1) if period > 4 else 0
    return clock_secs + periods_after * 12 * 60 + ot_periods_after * 5 * 60


class NBAAdapter(SportDataAdapter):
    """NBA play-by-play adapter with 2-minute sampling and scoring events."""

    sport = "NBA"
    SAMPLE_INTERVAL_SECONDS: ClassVar[int] = 120
    MAX_REGULATION_POINTS: ClassVar[int] = 130

    SCORING_DESCRIPTION_KEYWORDS: ClassVar[tuple[str, ...]] = (
        "makes",
        "made",
        "free throw",
        "dunk",
        "layup",
        "hook shot",
        "jump shot",
        "3pt",
        "three point",
    )

    def __init__(
        self,
        db: "PrismDatabase | None" = None,
        request_delay: float = 0.6,
    ) -> None:
        super().__init__(db)
        self.request_delay = request_delay

    def load_pbp(self, seasons: list[int]) -> pd.DataFrame:
        """
        Load NBA play-by-play for given seasons.

        Uses nba_api PlayByPlay endpoint per game. Season format: 2018 = 2017-18.
        """
        from nba_api.stats.endpoints import leaguegamefinder

        frames: list[pd.DataFrame] = []
        for season in seasons:
            logger.info("Loading NBA season %d", season)
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=f"{season - 1}-{str(season)[-2:]}",
                league_id_nullable="00",
            )
            games = finder.get_data_frames()[0]
            game_ids = games["GAME_ID"].unique()
            logger.info("Found %d NBA games for season %d", len(game_ids), season)

            for game_id in game_ids:
                try:
                    pbp = playbyplay.PlayByPlay(game_id=game_id)
                    df = pbp.get_data_frames()[0]
                    if df.empty:
                        continue
                    meta = games[games["GAME_ID"] == game_id].iloc[0]
                    df["season"] = season
                    df["game_id"] = game_id
                    df["game_date"] = pd.to_datetime(meta["GAME_DATE"]).date()
                    matchup = meta["MATCHUP"]
                    if " vs. " in matchup:
                        home, away = matchup.split(" vs. ")
                        df["home_team"] = home.strip()
                        df["away_team"] = away.strip()
                    elif " @ " in matchup:
                        away, home = matchup.split(" @ ")
                        df["home_team"] = home.strip()
                        df["away_team"] = away.strip()
                    else:
                        df["home_team"] = meta.get("TEAM_ABBREVIATION", "UNK")
                        df["away_team"] = "UNK"
                    frames.append(df)
                except Exception as exc:  # noqa: BLE001 — API failures are expected
                    logger.debug("Skipping game %s: %s", game_id, exc)
                time.sleep(self.request_delay)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _parse_scores(self, row: pd.Series) -> tuple[int, int]:
        """Extract home/away scores from SCORE column like '102 - 98'."""
        score = str(row.get("SCORE", "") or "")
        if " - " in score:
            parts = score.split(" - ")
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                pass
        return 0, 0

    def _is_scoring_event(self, row: pd.Series) -> bool:
        """Heuristic scoring detection from play description."""
        desc = str(row.get("HOMEDESCRIPTION") or row.get("VISITORDESCRIPTION") or "").lower()
        if not desc:
            return False
        if "miss" in desc:
            return False
        return any(kw in desc for kw in self.SCORING_DESCRIPTION_KEYWORDS)

    def _event_type(self, row: pd.Series) -> str | None:
        desc = str(row.get("HOMEDESCRIPTION") or row.get("VISITORDESCRIPTION") or "").lower()
        if "free throw" in desc:
            return "free_throw"
        if "3pt" in desc or "three point" in desc:
            return "3pt"
        if self._is_scoring_event(row):
            return "2pt"
        return None

    def extract_game_states(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """Extract states at scoring events and every 2 minutes of game clock."""
        if pbp.empty:
            return pd.DataFrame()

        records: list[dict[str, object]] = []
        anomaly_games: set[str] = set()

        for game_id, game_df in pbp.groupby("game_id"):
            game_df = game_df.sort_values(["PERIOD", "PCTIMESTRING"], ascending=[True, False])
            meta = game_df.iloc[0]
            home_team = meta["home_team"]
            away_team = meta["away_team"]
            season = int(meta["season"])
            game_date = meta["game_date"]

            last_sampled_clock: dict[int, int] = {}

            for _, row in game_df.iterrows():
                period = int(row["PERIOD"])
                clock_secs = parse_game_clock(row.get("PCTIMESTRING"))
                if clock_secs is None:
                    continue

                total_remaining = period_seconds_remaining(period, clock_secs)
                home_score, away_score = self._parse_scores(row)

                is_scoring = self._is_scoring_event(row)

                # Sample every 2 minutes within each period
                should_sample = False
                prev = last_sampled_clock.get(period)
                if prev is None or (prev - clock_secs) >= self.SAMPLE_INTERVAL_SECONDS:
                    should_sample = True
                    last_sampled_clock[period] = clock_secs

                if not is_scoring and not should_sample:
                    continue

                if home_score > self.MAX_REGULATION_POINTS and period <= 4:
                    anomaly_games.add(str(game_id))
                if away_score > self.MAX_REGULATION_POINTS and period <= 4:
                    anomaly_games.add(str(game_id))

                possession = None
                if pd.notna(row.get("PLAYER1_TEAM_ABBREVIATION")):
                    possession = row["PLAYER1_TEAM_ABBREVIATION"]

                records.append(
                    {
                        "game_id": game_id,
                        "sport": self.sport,
                        "season": season,
                        "game_date": game_date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "seconds_remaining": total_remaining,
                        "game_period": period,
                        "score_differential": home_score - away_score,
                        "home_score": home_score,
                        "away_score": away_score,
                        "possession": possession,
                        "is_scoring_event": is_scoring,
                        "event_type": self._event_type(row) if is_scoring else None,
                    }
                )

        if anomaly_games:
            logger.warning(
                "NBA anomaly flagged (>%d pts in regulation) for games: %s",
                self.MAX_REGULATION_POINTS,
                sorted(anomaly_games)[:10],
            )

        states = pd.DataFrame(records)
        if states.empty:
            return states

        states = states.drop_duplicates(subset=["game_id", "seconds_remaining"], keep="last")
        states = states.sort_values(
            ["game_id", "seconds_remaining"], ascending=[True, False]
        ).reset_index(drop=True)
        return states

    def validate_game_states(self, states: pd.DataFrame) -> bool:
        """
        NBA-specific validation:
        - Game clock strings parsed correctly (implicit in extraction)
        - Score differential consistent with home/away scores
        - High-scoring anomalies flagged but not dropped
        - Monotonic seconds_remaining within games
        - No duplicate (game_id, seconds_remaining)
        """
        if states.empty:
            logger.warning("No NBA game states to validate")
            return True

        inconsistent = states["score_differential"] != (
            states["home_score"] - states["away_score"]
        )
        if inconsistent.any():
            n = int(inconsistent.sum())
            raise ValueError(f"NBA validation failed: {n} rows with inconsistent score differential")

        for game_id, group in states.groupby("game_id"):
            secs = group["seconds_remaining"].to_numpy(dtype=np.int64)
            if len(secs) > 1 and not np.all(np.diff(secs) <= 0):
                raise ValueError(
                    f"NBA validation failed: seconds_remaining not monotonic in game {game_id}"
                )

        assert_no_duplicates(states, ["game_id", "seconds_remaining"], "NBA game states")
        logger.info("NBA game states passed all validation checks (%d rows)", len(states))
        return True
