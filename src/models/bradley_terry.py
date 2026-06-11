"""Bradley-Terry paired comparison model for pre-game win probability."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.base import WinProbabilityModel

logger = logging.getLogger(__name__)


class BradleyTerryModel(WinProbabilityModel):
    """
    Pre-game Bradley-Terry model with learnable home advantage.

    P(home wins) = exp(s_home + h) / (exp(s_home + h) + exp(s_away))
    """

    model_name = "bradley_terry"
    model_version = "1.0.0"

    def __init__(self, league_avg_strength: float = 0.0) -> None:
        self.league_avg_strength = league_avg_strength
        self.team_strengths: dict[str, float] = {}
        self.home_advantage_param: float = 0.0
        self._teams: list[str] = []

    def _neg_log_likelihood(
        self, params: np.ndarray, games: pd.DataFrame, outcomes: np.ndarray
    ) -> float:
        home_adv = params[0]
        strengths = dict(zip(self._teams, params[1:], strict=True))
        ll = 0.0
        for j, row in enumerate(games.itertuples(index=False)):
            s_home = strengths[row.home_team]  # type: ignore[attr-defined]
            s_away = strengths[row.away_team]  # type: ignore[attr-defined]
            logit = s_home + home_adv - s_away
            p_home = 1.0 / (1.0 + np.exp(-logit))
            y = outcomes[j]
            ll += y * np.log(p_home + 1e-12) + (1 - y) * np.log(1 - p_home + 1e-12)
        return -ll

    def fit(self, game_states: pd.DataFrame, outcomes: pd.Series) -> BradleyTerryModel:
        """Fit via maximum likelihood on one row per game."""
        games = self.final_game_states(game_states).reset_index(drop=True)
        if outcomes.index.equals(games.index):
            y = outcomes.to_numpy(dtype=float)
        else:
            y = self.home_win_outcomes(games).to_numpy(dtype=float)

        teams = sorted(set(games["home_team"]).union(set(games["away_team"])))
        self._teams = teams
        for team in teams:
            if team not in self.team_strengths:
                self.team_strengths[team] = self.league_avg_strength

        x0 = np.zeros(len(teams) + 1)
        x0[0] = self.home_advantage_param
        x0[1:] = [self.team_strengths[t] for t in teams]

        result = minimize(
            lambda p: self._neg_log_likelihood(p, games, y),
            x0,
            method="L-BFGS-B",
        )
        if not result.success:
            logger.warning("Bradley-Terry optimization did not converge: %s", result.message)

        self.home_advantage_param = float(result.x[0])
        self.team_strengths = {t: float(s) for t, s in zip(teams, result.x[1:], strict=True)}
        logger.info(
            "Bradley-Terry fit: %d teams, home_adv=%.3f",
            len(teams),
            self.home_advantage_param,
        )
        return self

    def predict(self, game_states: pd.DataFrame) -> np.ndarray:
        """Pre-game home win probability for each row."""
        probs = np.empty(len(game_states))
        for idx, row in enumerate(game_states.itertuples(index=False)):
            s_home = self.team_strengths.get(
                row.home_team, self.league_avg_strength  # type: ignore[attr-defined]
            )
            s_away = self.team_strengths.get(
                row.away_team, self.league_avg_strength  # type: ignore[attr-defined]
            )
            logit = s_home + self.home_advantage_param - s_away
            probs[idx] = 1.0 / (1.0 + np.exp(-logit))
        self.validate_predictions(probs)
        return probs

    def strength_diff(self, home_team: str, away_team: str) -> float:
        """Home strength minus away strength."""
        return self.team_strengths.get(home_team, 0.0) - self.team_strengths.get(
            away_team, 0.0
        )
