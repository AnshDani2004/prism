"""XGBoost in-play win probability model with isotonic calibration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit

from src.models.base import WinProbabilityModel
from src.models.bradley_terry import BradleyTerryModel

logger = logging.getLogger(__name__)

SPORT_TOTAL_SECONDS = {"NFL": 3600, "NBA": 2880}

FEATURE_COLUMNS = [
    "score_differential",
    "score_differential_sq",
    "home_score",
    "away_score",
    "seconds_remaining",
    "pct_time_elapsed",
    "log_seconds_remaining",
    "score_diff_x_time",
    "urgency",
    "is_final_period",
    "is_overtime",
    "pregame_home_prob",
    "pregame_strength_diff",
]


class XGBInPlayModel(WinProbabilityModel):
    """
    XGBoost-based in-play win probability model.

    Trained on historical play-by-play with isotonic regression calibration
    on the validation set. Strict temporal split: train 2018-2021, val 2022, test 2023.
    """

    model_name = "xgb_inplay"
    model_version = "1.0.0"

    def __init__(
        self,
        train_seasons: tuple[int, ...] = (2018, 2019, 2020, 2021),
        val_season: int = 2022,
        test_season: int = 2023,
        output_dir: Path | str = "outputs/calibration",
    ) -> None:
        self.train_seasons = train_seasons
        self.val_season = val_season
        self.test_season = test_season
        self.output_dir = Path(output_dir)
        self._booster: xgb.XGBClassifier | None = None
        self._calibrator: IsotonicRegression | None = None
        self._pregame_model: BradleyTerryModel | None = None
        self.training_game_dates: list[pd.Timestamp] = []
        self.validation_game_dates: list[pd.Timestamp] = []
        self.feature_importances_: dict[str, float] = {}

    def _total_seconds(self, row: pd.Series) -> int:
        sport = str(row.get("sport", "NFL")).upper()
        return SPORT_TOTAL_SECONDS.get(sport, 3600)

    def engineer_features(self, game_states: pd.DataFrame) -> pd.DataFrame:
        """Build model feature matrix from raw game states."""
        df = game_states.copy()
        total = df.apply(self._total_seconds, axis=1)
        df["score_differential_sq"] = df["score_differential"] ** 2
        df["pct_time_elapsed"] = 1.0 - df["seconds_remaining"] / total
        df["log_seconds_remaining"] = np.log(df["seconds_remaining"] + 1)
        df["score_diff_x_time"] = df["score_differential"] * df["pct_time_elapsed"]
        df["urgency"] = df["score_differential"].abs() / (df["seconds_remaining"] + 1)

        sport = df["sport"].str.upper() if "sport" in df.columns else pd.Series(["NFL"] * len(df))
        df["is_final_period"] = (
            ((sport == "NFL") & (df["game_period"] == 4))
            | ((sport == "NBA") & (df["game_period"] == 4))
        ).astype(int)
        df["is_overtime"] = (df["game_period"] > 4).astype(int)

        if self._pregame_model is not None:
            pregame_df = df.drop_duplicates(subset=["game_id"])[["game_id", "home_team", "away_team"]]
            pregame_probs = self._pregame_model.predict(pregame_df)
            prob_map = dict(zip(pregame_df["game_id"], pregame_probs, strict=True))
            diff_map = {
                gid: self._pregame_model.strength_diff(h, a)
                for gid, h, a in zip(
                    pregame_df["game_id"],
                    pregame_df["home_team"],
                    pregame_df["away_team"],
                    strict=True,
                )
            }
            df["pregame_home_prob"] = df["game_id"].map(prob_map).fillna(0.5)
            df["pregame_strength_diff"] = df["game_id"].map(diff_map).fillna(0.0)
        else:
            df["pregame_home_prob"] = 0.55
            df["pregame_strength_diff"] = 0.0

        return df

    def _split_by_season(
        self, game_states: pd.DataFrame, outcomes: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Temporal train/val/test split by season."""
        final = self.final_game_states(game_states)
        outcome_map = dict(zip(final["game_id"], self.home_win_outcomes(final), strict=True))

        states = game_states.copy()
        states["_outcome"] = states["game_id"].map(outcome_map)

        train = states[states["season"].isin(self.train_seasons)]
        val = states[states["season"] == self.val_season]
        test = states[states["season"] == self.test_season]

        y_train = train["_outcome"]
        y_val = val["_outcome"]
        y_test = test["_outcome"]
        return (
            train.drop(columns="_outcome"),
            y_train,
            val.drop(columns="_outcome"),
            y_val,
            test.drop(columns="_outcome"),
            y_test,
        )

    def fit(self, game_states: pd.DataFrame, outcomes: pd.Series) -> XGBInPlayModel:
        """Train XGBoost with time-series CV; calibrate on validation set."""
        train_states, y_train, val_states, y_val, _, _ = self._split_by_season(
            game_states, outcomes
        )

        self.training_game_dates = sorted(pd.to_datetime(train_states["game_date"]).unique())
        self.validation_game_dates = sorted(pd.to_datetime(val_states["game_date"]).unique())

        # Pre-game prior from Bradley-Terry on training games only
        self._pregame_model = BradleyTerryModel()
        self._pregame_model.fit(train_states, y_train)

        x_train = self.engineer_features(train_states)
        x_val = self.engineer_features(val_states)

        best_params = self._tune_hyperparams(x_train, y_train)

        self._booster = xgb.XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
        self._booster.fit(
            x_train[FEATURE_COLUMNS],
            y_train,
            eval_set=[(x_val[FEATURE_COLUMNS], y_val)],
            verbose=False,
        )

        importances = self._booster.feature_importances_
        self.feature_importances_ = dict(
            zip(FEATURE_COLUMNS, importances.astype(float), strict=True)
        )
        self._save_feature_importances()

        raw_val = self._booster.predict_proba(x_val[FEATURE_COLUMNS])[:, 1]
        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(raw_val, y_val.to_numpy())

        logger.info(
            "XGB in-play fit: train=%d rows, val=%d rows",
            len(x_train),
            len(x_val),
        )
        return self

    def _tune_hyperparams(self, x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, object]:
        """5-fold time-series CV on training set only."""
        param_grid = [
            {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100},
            {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 200},
            {"max_depth": 5, "learning_rate": 0.1, "n_estimators": 150},
        ]
        best_params: dict[str, object] = param_grid[0]
        best_score = float("inf")
        tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(x_train) // 50)))

        for params in param_grid:
            scores: list[float] = []
            for train_idx, val_idx in tscv.split(x_train):
                model = xgb.XGBClassifier(**params, objective="binary:logistic", random_state=42)
                model.fit(
                    x_train.iloc[train_idx][FEATURE_COLUMNS],
                    y_train.iloc[train_idx],
                    verbose=False,
                )
                preds = model.predict_proba(x_train.iloc[val_idx][FEATURE_COLUMNS])[:, 1]
                scores.append(float(np.mean((preds - y_train.iloc[val_idx]) ** 2)))
            if float(np.mean(scores)) < best_score:
                best_score = float(np.mean(scores))
                best_params = params

        return best_params

    def predict(self, game_states: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """Predict home win probability; apply isotonic calibration if fitted."""
        if self._booster is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        features = self.engineer_features(game_states)
        raw = self._booster.predict_proba(features[FEATURE_COLUMNS])[:, 1]
        if calibrated and self._calibrator is not None:
            probs = self._calibrator.predict(raw)
        else:
            probs = raw
        self.validate_predictions(probs)
        return probs

    def _save_feature_importances(self) -> None:
        """Persist feature importances to output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "xgb_feature_importances.json"
        path.write_text(json.dumps(self.feature_importances_, indent=2))
        logger.info("Feature importances saved to %s", path)
