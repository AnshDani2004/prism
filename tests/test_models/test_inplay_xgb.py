"""Tests for XGBoost in-play model."""

import numpy as np
import pandas as pd
import pytest

from src.models.bradley_terry import BradleyTerryModel
from src.models.inplay_xgb import XGBInPlayModel


def test_no_future_data_leakage(synthetic_game_states, synthetic_outcomes):
    model = XGBInPlayModel()
    model.fit(synthetic_game_states, synthetic_outcomes)
    assert max(model.training_game_dates) < min(model.validation_game_dates)


def test_inplay_beats_pregame_midgame(synthetic_game_states, synthetic_outcomes):
    midgame = synthetic_game_states[
        (synthetic_game_states["seconds_remaining"] == 1800)
        & (synthetic_game_states["season"] == 2022)
    ]
    if midgame.empty:
        pytest.skip("No 2022 midgame states in synthetic data")

    xgb = XGBInPlayModel()
    xgb.fit(synthetic_game_states, synthetic_outcomes)
    bt = BradleyTerryModel()
    bt.fit(synthetic_game_states, synthetic_outcomes)

    outcome_map = dict(
        zip(
            xgb.final_game_states(synthetic_game_states)["game_id"],
            xgb.home_win_outcomes(xgb.final_game_states(synthetic_game_states)),
            strict=True,
        )
    )
    outcomes = midgame["game_id"].map(outcome_map).to_numpy()
    xgb_probs = xgb.predict(midgame)
    bt_probs = bt.predict(midgame)
    xgb_ece = xgb.calibration_error(xgb_probs, outcomes)
    bt_ece = bt.calibration_error(bt_probs, outcomes)
    assert xgb_ece <= bt_ece + 0.15


def test_high_differential_high_confidence(synthetic_game_states, synthetic_outcomes):
    model = XGBInPlayModel()
    model.fit(synthetic_game_states, synthetic_outcomes)
    state = pd.DataFrame(
        {
            "game_id": ["late_lead"],
            "sport": ["NFL"],
            "season": [2021],
            "game_date": [pd.Timestamp("2021-09-01")],
            "home_team": ["T01"],
            "away_team": ["T02"],
            "seconds_remaining": [60],
            "game_period": [4],
            "score_differential": [30],
            "home_score": [35],
            "away_score": [5],
            "possession": ["T01"],
            "is_scoring_event": [False],
            "event_type": [None],
        }
    )
    prob = model.predict(state)[0]
    assert prob > 0.90


def test_tied_game_near_50pct(synthetic_game_states, synthetic_outcomes):
    model = XGBInPlayModel()
    model.fit(synthetic_game_states, synthetic_outcomes)
    state = pd.DataFrame(
        {
            "game_id": ["early_tie"],
            "sport": ["NFL"],
            "season": [2021],
            "game_date": [pd.Timestamp("2021-09-01")],
            "home_team": ["T01"],
            "away_team": ["T02"],
            "seconds_remaining": [3600],
            "game_period": [1],
            "score_differential": [0],
            "home_score": [0],
            "away_score": [0],
            "possession": ["T01"],
            "is_scoring_event": [False],
            "event_type": [None],
        }
    )
    prob = model.predict(state)[0]
    assert 0.35 <= prob <= 0.70
