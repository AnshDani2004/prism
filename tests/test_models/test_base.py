"""Tests for model base class."""

import numpy as np
import pytest

from src.models.base import WinProbabilityModel


def test_calibration_error_perfect():
    probs = np.array([0.1, 0.3, 0.7, 0.9])
    outcomes = np.array([0, 0, 1, 1])
    ece = WinProbabilityModel.calibration_error(probs, outcomes, n_bins=2)
    assert ece < 0.15


def test_validate_predictions():
    from src.models.bradley_terry import BradleyTerryModel

    probs = np.array([0.0, 0.5, 1.0])
    assert BradleyTerryModel().validate_predictions(probs) is True


def test_validate_predictions_rejects_nan():
    from src.models.bradley_terry import BradleyTerryModel

    with pytest.raises(AssertionError):
        BradleyTerryModel().validate_predictions(np.array([np.nan]))
