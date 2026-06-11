#!/usr/bin/env python3
"""Phase 2 validation checkpoint."""

import numpy as np

from src.data.database import PrismDatabase
from src.models.bradley_terry import BradleyTerryModel
from src.models.inplay_xgb import XGBInPlayModel
from src.models.margin_model import MarginRatingModel


def main() -> None:
    db = PrismDatabase()
    states = db.query_df("SELECT * FROM game_states")
    if states.empty:
        raise SystemExit("No game_states in database. Run ingest_sports.py first.")

    final = BradleyTerryModel.final_game_states(states)
    outcomes = BradleyTerryModel.home_win_outcomes(final)
    test = states[states["season"] == 2023]
    test_final = BradleyTerryModel.final_game_states(test)
    test_outcomes = BradleyTerryModel.home_win_outcomes(test_final).to_numpy()

    bt = BradleyTerryModel().fit(states, outcomes)
    mm = MarginRatingModel().fit(states, outcomes)
    xgb = XGBInPlayModel().fit(states, outcomes)

    for model_name, model in [("bradley_terry", bt), ("margin_model", mm), ("xgb_inplay", xgb)]:
        probs = model.predict(test if model_name == "xgb_inplay" else test_final)
        y = test_outcomes if model_name != "xgb_inplay" else test_outcomes.repeat(
            len(probs) // max(len(test_outcomes), 1)
        )[: len(probs)]

        assert np.all(probs >= 0) and np.all(probs <= 1), f"{model_name}: probs out of range"
        assert not np.any(np.isnan(probs)), f"{model_name}: NaN predictions"

        ece = model.calibration_error(probs, y)
        brier = float(np.mean((probs - y) ** 2))
        print(f"{model_name}: ECE={ece:.4f}, Brier={brier:.4f}")

    print("Phase 2 checkpoint PASSED (review metrics above)")


if __name__ == "__main__":
    main()
