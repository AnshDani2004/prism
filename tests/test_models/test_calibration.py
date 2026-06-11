"""Tests for calibration analysis."""

import numpy as np
import pytest

from src.models.bradley_terry import BradleyTerryModel
from src.models.calibration import CalibrationAnalyzer


def test_ece_computation_correct():
    probs = np.array([0.2, 0.2, 0.8, 0.8])
    outcomes = np.array([0, 0, 1, 1])
    ece = CalibrationAnalyzer.ece(probs, outcomes, n_bins=2)
    assert ece < 0.05


def test_isotonic_calibration_reduces_ece(synthetic_game_states, synthetic_outcomes, tmp_path):
    from src.models.inplay_xgb import XGBInPlayModel

    model = XGBInPlayModel(output_dir=tmp_path)
    model.fit(synthetic_game_states, synthetic_outcomes)
    val = synthetic_game_states[synthetic_game_states["season"] == 2022]
    if val.empty:
        pytest.skip("No validation data")
    outcomes = model.home_win_outcomes(model.final_game_states(val)).to_numpy()
    raw = model.predict(val, calibrated=False)
    calibrated = model.predict(val, calibrated=True)
    raw_ece = CalibrationAnalyzer.ece(raw, outcomes[: len(raw)])
    cal_ece = CalibrationAnalyzer.ece(calibrated, outcomes[: len(calibrated)])
    assert cal_ece <= raw_ece + 0.01


def test_calibration_plot_saves_to_disk(synthetic_game_states, synthetic_outcomes, tmp_path):
    analyzer = CalibrationAnalyzer(output_dir=tmp_path)
    model = BradleyTerryModel()
    model.fit(synthetic_game_states, synthetic_outcomes)
    final = model.final_game_states(synthetic_game_states)
    probs = model.predict(final)
    outcomes = model.home_win_outcomes(final).to_numpy()
    path = analyzer.plot_reliability_diagram(probs, outcomes, model_name="bt", save=True)
    assert path is not None
    assert path.exists()


def test_brier_decomposition():
    analyzer = CalibrationAnalyzer()
    probs = np.linspace(0.1, 0.9, 100)
    outcomes = (np.random.default_rng(0).random(100) < probs).astype(float)
    decomp = analyzer.brier_decomposition(probs, outcomes)
    assert "brier" in decomp
    assert abs(decomp["brier"] - (decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"])) < 0.01
