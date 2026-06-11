"""Tests for Bayesian online state-space model."""

import numpy as np
import pandas as pd
import pytest

from src.models.bayesian_online import BayesianOnlineWinProb


@pytest.fixture
def ekf_model() -> BayesianOnlineWinProb:
    return BayesianOnlineWinProb(inference="ekf", seed=42)


@pytest.fixture
def particle_model() -> BayesianOnlineWinProb:
    return BayesianOnlineWinProb(inference="particle", n_particles=500, seed=42)


@pytest.fixture
def sample_scoring_game() -> pd.DataFrame:
    """Single game with several scoring events."""
    return pd.DataFrame(
        {
            "game_id": ["g1"] * 5,
            "sport": ["NFL"] * 5,
            "season": [2021] * 5,
            "game_date": [pd.Timestamp("2021-09-01")] * 5,
            "home_team": ["KC"] * 5,
            "away_team": ["PHI"] * 5,
            "seconds_remaining": [3600, 2800, 2000, 1200, 600],
            "game_period": [1, 2, 2, 3, 4],
            "score_differential": [0, 7, 7, 14, 14],
            "home_score": [0, 7, 7, 14, 14],
            "away_score": [0, 0, 0, 0, 0],
            "is_scoring_event": [False, True, False, True, False],
            "event_type": [None, "touchdown", None, "touchdown", None],
        }
    )


def test_posterior_updates_in_correct_direction(ekf_model: BayesianOnlineWinProb):
    ekf_model.reset()
    ekf_model._posterior.score_diff = 0
    initial_prob = ekf_model.win_probability(seconds_remaining=1800)
    ekf_model.update({"team": "home", "points": 7}, time_elapsed=1800)
    updated_prob = ekf_model.win_probability(seconds_remaining=1800)
    assert updated_prob > initial_prob


def test_uncertainty_increases_with_time_remaining(ekf_model: BayesianOnlineWinProb):
    """Leading team has lower win prob with more time left (more comeback uncertainty)."""
    ekf_model.reset()
    ekf_model._posterior.score_diff = 7
    prob_early = ekf_model.win_probability(seconds_remaining=3500)
    ekf_model.reset()
    ekf_model._posterior.score_diff = 7
    prob_late = ekf_model.win_probability(seconds_remaining=60)
    assert prob_late > prob_early


def test_ekf_and_particle_agree(
    sample_scoring_game: pd.DataFrame,
):
    ekf_model = BayesianOnlineWinProb(inference="ekf", seed=42)
    particle_model = BayesianOnlineWinProb(inference="particle", n_particles=1000, seed=42)

    ekf_model.replay_game(sample_scoring_game)
    particle_model.replay_game(sample_scoring_game)

    # Compare final win prob with identical MC seed
    ekf_model._rng = np.random.default_rng(99)
    particle_model._rng = np.random.default_rng(99)
    prob_ekf = ekf_model.win_probability(600, score_diff=14)
    prob_pf = particle_model.win_probability(600, score_diff=14)
    assert abs(prob_ekf - prob_pf) < 0.03


def test_hyperparams_improve_calibration(synthetic_game_states: pd.DataFrame):
    model = BayesianOnlineWinProb(inference="ekf", seed=42)
    train = synthetic_game_states[synthetic_game_states["season"].isin([2018, 2019, 2020, 2021])]
    val = synthetic_game_states[synthetic_game_states["season"] == 2022]
    if val.empty:
        pytest.skip("No 2022 validation data")

    model.fit_hyperparams(train)
    all_final = model.final_game_states(synthetic_game_states)
    outcome_map = dict(
        zip(all_final["game_id"], model.home_win_outcomes(all_final), strict=True)
    )
    test_outcomes = val["game_id"].map(outcome_map)

    default_ece = model.evaluate_calibration_error(
        use_fitted_params=False,
        game_states=val,
        test_outcomes=test_outcomes,
    )
    fitted_ece = model.evaluate_calibration_error(
        use_fitted_params=True,
        game_states=val,
        test_outcomes=test_outcomes,
    )
    assert fitted_ece <= default_ece + 0.05


def test_analytic_jacobian_matches_numerical(ekf_model: BayesianOnlineWinProb):
    theta = 0.5
    analytic = ekf_model._H(theta)
    eps = 1e-6
    numerical = (ekf_model._h(theta + eps) - ekf_model._h(theta - eps)) / (2 * eps)
    assert abs(analytic - numerical) < 1e-4


def test_extract_scoring_events(sample_scoring_game: pd.DataFrame):
    events = BayesianOnlineWinProb.extract_scoring_events(sample_scoring_game)
    assert len(events) == 2
    assert events[0].team == "home"
    assert events[0].points == 7


def test_predict_on_game_states(sample_scoring_game: pd.DataFrame):
    model = BayesianOnlineWinProb(inference="ekf", seed=42)
    model.fit_hyperparams(sample_scoring_game)
    probs = model.predict(sample_scoring_game)
    assert len(probs) == len(sample_scoring_game)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_win_probability_bounds(ekf_model: BayesianOnlineWinProb):
    ekf_model.reset()
    for secs in [3600, 900, 60, 0]:
        p = ekf_model.win_probability(seconds_remaining=secs)
        assert 0.0 <= p <= 1.0


def test_posterior_impact_sign(ekf_model: BayesianOnlineWinProb):
    ekf_model.reset()
    impact_home = ekf_model.posterior_impact({"team": "home", "points": 3}, time_elapsed=100)
    ekf_model.reset()
    impact_away = ekf_model.posterior_impact({"team": "away", "points": 3}, time_elapsed=100)
    assert impact_home > 0
    assert impact_away < 0
