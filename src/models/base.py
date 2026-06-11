"""Abstract base class for all win probability models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class WinProbabilityModel(ABC):
    """
    All win probability models must implement this interface.

    Contract:
    - predict() always returns float in [0, 1] representing home team win probability
    - fit() accepts game state DataFrame and returns self
    - calibration_error() returns Expected Calibration Error (ECE)
    """

    model_name: str = "base"
    model_version: str = "0.1.0"

    @abstractmethod
    def fit(self, game_states: pd.DataFrame, outcomes: pd.Series) -> WinProbabilityModel:
        """Fit model parameters from historical game states and outcomes."""

    @abstractmethod
    def predict(self, game_states: pd.DataFrame) -> np.ndarray:
        """Return array of home win probabilities in [0, 1]."""

    @staticmethod
    def calibration_error(
        probs: np.ndarray,
        outcomes: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Expected Calibration Error (ECE) with equal-frequency binning.

        A well-calibrated model predicting p=0.7 should win ~70% of the time.
        ECE < 0.05 is considered well-calibrated.
        """
        probs = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        if len(probs) == 0:
            return 0.0

        order = np.argsort(probs)
        probs_sorted = probs[order]
        outcomes_sorted = outcomes[order]
        bin_edges = np.array_split(np.arange(len(probs)), n_bins)

        ece = 0.0
        n = len(probs)
        for indices in bin_edges:
            if len(indices) == 0:
                continue
            bin_probs = probs_sorted[indices]
            bin_outcomes = outcomes_sorted[indices]
            avg_conf = float(np.mean(bin_probs))
            avg_acc = float(np.mean(bin_outcomes))
            ece += (len(indices) / n) * abs(avg_acc - avg_conf)
        return float(ece)

    def brier_score(self, probs: np.ndarray, outcomes: np.ndarray) -> float:
        """Mean squared error between probabilities and binary outcomes."""
        probs = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        return float(np.mean((probs - outcomes) ** 2))

    def validate_predictions(self, probs: np.ndarray) -> bool:
        """All predictions must be in [0, 1], no NaN, no inf."""
        assert np.all(probs >= 0) and np.all(probs <= 1), "Probabilities out of range"
        assert not np.any(np.isnan(probs)), "NaN probabilities"
        assert not np.any(np.isinf(probs)), "Infinite probabilities"
        return True

    @staticmethod
    def final_game_states(game_states: pd.DataFrame) -> pd.DataFrame:
        """One row per game: the final observed state (lowest seconds_remaining)."""
        return (
            game_states.sort_values(["game_id", "seconds_remaining"])
            .groupby("game_id", as_index=False)
            .first()
        )

    @staticmethod
    def home_win_outcomes(final_states: pd.DataFrame) -> pd.Series:
        """Binary outcome: 1 if home team won, 0 otherwise."""
        home_wins = (final_states["home_score"] > final_states["away_score"]).astype(int)
        return pd.Series(home_wins.values, index=final_states.index)
