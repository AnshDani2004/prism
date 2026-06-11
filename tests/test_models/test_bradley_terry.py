"""Tests for Bradley-Terry model."""

import numpy as np
import pandas as pd
import pytest

from src.models.bradley_terry import BradleyTerryModel


def create_game(home: str, away: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["test"],
            "home_team": [home],
            "away_team": [away],
            "sport": ["NFL"],
            "season": [2022],
            "game_date": [pd.Timestamp("2022-09-01")],
            "seconds_remaining": [3600],
            "game_period": [1],
            "score_differential": [0],
            "home_score": [0],
            "away_score": [0],
        }
    )


def test_probabilities_in_unit_interval(synthetic_game_states, synthetic_outcomes):
    model = BradleyTerryModel()
    model.fit(synthetic_game_states, synthetic_outcomes)
    probs = model.predict(synthetic_game_states)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_symmetry(synthetic_game_states, synthetic_outcomes):
    model = BradleyTerryModel()
    model.fit(synthetic_game_states, synthetic_outcomes)
    game_ab = create_game("T01", "T02")
    game_ba = create_game("T02", "T01")
    prob_ab = model.predict(game_ab)[0]
    prob_ba = model.predict(game_ba)[0]
    assert abs(prob_ab + (1 - prob_ba)) < 1e-4


def test_home_advantage_is_positive(synthetic_game_states, synthetic_outcomes):
    model = BradleyTerryModel()
    model.fit(synthetic_game_states, synthetic_outcomes)
    assert model.home_advantage_param > 0


def test_calibration_below_threshold(synthetic_game_states, synthetic_outcomes):
    model = BradleyTerryModel()
    model.fit(synthetic_game_states, synthetic_outcomes)
    final = model.final_game_states(synthetic_game_states)
    probs = model.predict(final)
    outcomes = model.home_win_outcomes(final).to_numpy()
    ece = model.calibration_error(probs, outcomes)
    assert ece < 0.10
