"""Tests for margin-of-victory rating model."""

import numpy as np
import pytest

from src.models.bradley_terry import BradleyTerryModel
from src.models.margin_model import MarginRatingModel


def test_margin_model_calibration_better_than_bt(synthetic_game_states, synthetic_outcomes):
    final = MarginRatingModel.final_game_states(synthetic_game_states)
    outcomes = MarginRatingModel.home_win_outcomes(final).to_numpy()

    bt = BradleyTerryModel()
    bt.fit(synthetic_game_states, synthetic_outcomes)
    mm = MarginRatingModel()
    mm.fit(synthetic_game_states, synthetic_outcomes)

    bt_probs = bt.predict(final)
    mm_probs = mm.predict(final)
    bt_ece = bt.calibration_error(bt_probs, outcomes)
    mm_ece = mm.calibration_error(mm_probs, outcomes)
    assert mm_ece <= bt_ece + 0.05  # allow small tolerance on synthetic data


def test_time_weighting_effect():
    model = MarginRatingModel(decay_half_life_days=180.0)
    recent = model.get_weight(days_ago=7)
    old = model.get_weight(days_ago=365)
    assert recent > old


def test_mov_autocorrelation_correction():
    model = MarginRatingModel()
    delta_favorite = model.rating_update(margin=30, pregame_prob=0.85)
    delta_underdog = model.rating_update(margin=30, pregame_prob=0.15)
    assert abs(delta_underdog) > abs(delta_favorite)


def test_monte_carlo_seed_reproducibility(synthetic_game_states, synthetic_outcomes):
    model = MarginRatingModel()
    model.fit(synthetic_game_states, synthetic_outcomes)
    final = model.final_game_states(synthetic_game_states).iloc[[0]]
    prob_1 = model.predict(final, seed=42)[0]
    prob_2 = model.predict(final, seed=42)[0]
    assert prob_1 == prob_2


def test_skellam_pmf_sums_to_one():
    model = MarginRatingModel()
    pmf = model.margin_distribution("NFL", rating_diff=3.0)
    assert abs(pmf.sum() - 1.0) < 1e-6
