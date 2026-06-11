"""Fixtures for model tests."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


def _make_game_states(n_games: int = 80, seed: int = 42) -> pd.DataFrame:
    """Synthetic multi-season game states for model fitting."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    teams = [f"T{i:02d}" for i in range(12)]

    for g in range(n_games):
        season = 2018 + (g % 6)  # 2018-2023
        game_date = date(2018, 9, 1) + timedelta(days=g * 4)
        home = teams[g % len(teams)]
        away = teams[(g + 3) % len(teams)]
        home_boost = 3 if rng.random() < 0.55 else 0  # home-field edge in outcomes
        home_score = max(0, int(rng.poisson(22 + home_boost)))
        away_score = max(0, int(rng.poisson(22)))
        if home_score == away_score:
            home_score += rng.choice([-1, 1])
        home_score = max(0, home_score)
        away_score = max(0, away_score)

        game_id = f"{season}_{g:04d}_{home}_{away}"
        for secs in [3600, 1800, 900, 0]:
            frac = 1 - secs / 3600
            diff = int(home_score - away_score) if secs == 0 else int((home_score - away_score) * frac)
            rows.append(
                {
                    "game_id": game_id,
                    "sport": "NFL",
                    "season": season,
                    "game_date": game_date,
                    "home_team": home,
                    "away_team": away,
                    "seconds_remaining": secs,
                    "game_period": 4 if secs == 0 else max(1, 4 - secs // 900),
                    "score_differential": diff,
                    "home_score": max(0, home_score if secs == 0 else int(home_score * frac)),
                    "away_score": max(0, away_score if secs == 0 else int(away_score * frac)),
                    "possession": home,
                    "is_scoring_event": secs in {1800, 0},
                    "event_type": "touchdown" if secs in {1800, 0} else None,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_game_states() -> pd.DataFrame:
    return _make_game_states(n_games=100)


@pytest.fixture
def synthetic_outcomes(synthetic_game_states: pd.DataFrame) -> pd.Series:
    final = (
        synthetic_game_states.sort_values("seconds_remaining")
        .groupby("game_id", as_index=False)
        .first()
    )
    return (final["home_score"] > final["away_score"]).astype(int)
