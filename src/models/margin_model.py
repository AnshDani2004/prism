"""Margin-of-victory rating model with Skellam (NFL) and Gaussian (NBA) margins."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import skellam

from src.models.base import WinProbabilityModel
from src.utils.pandas_typing import as_str

logger = logging.getLogger(__name__)


class MarginRatingModel(WinProbabilityModel):
    """
    Elo-style ratings with margin-of-victory adjustment and sport-specific
    margin distributions. Replaces Dixon-Coles (soccer-specific) for American sports.
    """

    model_name = "margin_model"
    model_version = "1.0.0"

    def __init__(
        self,
        k_factor: float = 20.0,
        home_advantage_elo: float = 65.0,
        initial_rating: float = 1500.0,
        season_shrinkage: float = 0.25,
        decay_half_life_days: float = 180.0,
        nfl_lambda: float = 3.0,
        nba_sigma: float = 12.0,
        mc_samples: int = 10_000,
    ) -> None:
        self.k_factor = k_factor
        self.home_advantage_elo = home_advantage_elo
        self.initial_rating = initial_rating
        self.season_shrinkage = season_shrinkage
        self.decay_half_life_days = decay_half_life_days
        self.nfl_lambda = nfl_lambda
        self.nba_sigma = nba_sigma
        self.mc_samples = mc_samples
        self.ratings: dict[str, float] = {}
        self._reference_date: datetime | None = None

    def get_weight(self, days_ago: float) -> float:
        """Exponential time-decay weight; recent games count more."""
        if days_ago < 0:
            days_ago = 0.0
        return float(0.5 ** (days_ago / self.decay_half_life_days))

    def mov_multiplier(self, margin: float, pregame_prob: float) -> float:
        """
        FiveThirtyEight-style MOV multiplier with autocorrelation correction.

        Blowouts by heavy favorites move ratings less than underdog upsets.
        """
        margin = abs(margin)
        elo_diff = abs(pregame_prob - 0.5) * 400  # rough elo-diff proxy from prob
        autocorr = 2.2 / (elo_diff * 0.001 + 2.2)
        return float(np.log(margin + 1) * autocorr)

    def rating_update(self, margin: float, pregame_prob: float, k: float | None = None) -> float:
        """Signed Elo rating change for the home team given margin and pregame prob."""
        k = k or self.k_factor
        mult = self.mov_multiplier(margin, pregame_prob)
        actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        return k * mult * (actual - pregame_prob)

    def _shrink_to_mean(self, season: int) -> None:
        """Regress ratings toward league mean between seasons."""
        if not self.ratings:
            return
        mean_rating = float(np.mean(list(self.ratings.values())))
        for team in self.ratings:
            self.ratings[team] = (
                1 - self.season_shrinkage
            ) * self.ratings[team] + self.season_shrinkage * mean_rating
        logger.debug("Season %d shrinkage applied (factor=%.2f)", season, self.season_shrinkage)

    def _pregame_prob_elo(self, home_rating: float, away_rating: float) -> float:
        diff = home_rating + self.home_advantage_elo - away_rating
        return float(1.0 / (1.0 + 10 ** (-diff / 400)))

    def fit(
        self, game_states: pd.DataFrame, outcomes: pd.Series | None = None
    ) -> MarginRatingModel:
        """Fit decay parameter via CV and update ratings chronologically."""
        games = self.final_game_states(game_states).sort_values("game_date")
        if games.empty:
            return self

        # Fit decay half-life on training games
        self.decay_half_life_days = self._fit_decay(games)

        current_season: int | None = None
        ref_date = pd.to_datetime(games["game_date"].max())
        self._reference_date = ref_date.to_pydatetime()

        for _, row in games.iterrows():
            season = int(row["season"])
            if current_season is not None and season != current_season:
                self._shrink_to_mean(season)
            current_season = season

            home, away = str(row["home_team"]), str(row["away_team"])
            self.ratings.setdefault(home, self.initial_rating)
            self.ratings.setdefault(away, self.initial_rating)

            pre_prob = self._pregame_prob_elo(self.ratings[home], self.ratings[away])
            margin = int(row["home_score"]) - int(row["away_score"])
            days_ago = (ref_date - pd.to_datetime(row["game_date"])).days
            weight = self.get_weight(days_ago)
            delta = self.rating_update(margin, pre_prob, k=self.k_factor * weight)

            self.ratings[home] += delta
            self.ratings[away] -= delta

        logger.info(
            "Margin model fit: %d teams, decay_half_life=%.0f days",
            len(self.ratings),
            self.decay_half_life_days,
        )
        return self

    def _fit_decay(self, games: pd.DataFrame) -> float:
        """Cross-validate decay half-life on chronological game splits."""
        candidates = [90.0, 180.0, 365.0]
        best_decay = self.decay_half_life_days
        best_loss = float("inf")

        for decay in candidates:
            self.decay_half_life_days = decay
            losses: list[float] = []
            n = len(games)
            if n < 20:
                return decay
            split = int(0.8 * n)
            train, val = games.iloc[:split], games.iloc[split:]
            temp = MarginRatingModel(
                k_factor=self.k_factor,
                home_advantage_elo=self.home_advantage_elo,
                initial_rating=self.initial_rating,
                season_shrinkage=self.season_shrinkage,
                decay_half_life_days=decay,
            )
            temp.fit(train)
            probs = temp.predict(val)
            outcomes = self.home_win_outcomes(val).to_numpy()
            losses.append(float(np.mean((probs - outcomes) ** 2)))
            if losses[-1] < best_loss:
                best_loss = losses[-1]
                best_decay = decay

        return best_decay

    def margin_distribution(
        self,
        sport: str,
        rating_diff: float,
        max_margin: int = 30,
    ) -> np.ndarray:
        """
        Parametric margin PMF centered on rating differential.

        NFL: Skellam (difference of Poissons). NBA: discretized Gaussian.
        """
        if sport.upper() == "NFL":
            mu1 = self.nfl_lambda + max(rating_diff, 0) / 10
            mu2 = self.nfl_lambda - min(rating_diff, 0) / 10
            xs = np.arange(-max_margin, max_margin + 1)
            pmf = skellam.pmf(xs, mu1, mu2)
        else:
            xs = np.arange(-max_margin, max_margin + 1)
            pmf = stats.norm.pdf(xs, loc=rating_diff, scale=self.nba_sigma)
        pmf = np.maximum(pmf, 0)
        pmf = pmf / pmf.sum()
        return np.asarray(pmf, dtype=float)

    def predict(self, game_states: pd.DataFrame, seed: int | None = None) -> np.ndarray:
        """Pre-game win probability via analytic (NBA) or Monte Carlo (NFL)."""
        rng = np.random.default_rng(seed)
        probs = np.empty(len(game_states))

        for idx, row in enumerate(game_states.itertuples(index=False)):
            home = as_str(row.home_team)
            away = as_str(row.away_team)
            sport = as_str(row.sport)
            r_home = self.ratings.get(home, self.initial_rating)
            r_away = self.ratings.get(away, self.initial_rating)
            rating_diff = (r_home + self.home_advantage_elo - r_away) / 25.0

            if sport.upper() == "NBA":
                z = rating_diff * 25 / self.nba_sigma
                probs[idx] = float(stats.norm.cdf(z))
            else:
                pmf = self.margin_distribution("NFL", rating_diff)
                max_margin = (len(pmf) - 1) // 2
                margins = np.arange(-max_margin, max_margin + 1)
                if seed is not None:
                    samples = rng.choice(margins, size=self.mc_samples, p=pmf)
                    probs[idx] = float(np.mean(samples > 0))
                else:
                    probs[idx] = float(np.sum(pmf[margins > 0]))

        self.validate_predictions(probs)
        return probs
