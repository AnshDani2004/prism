"""Tests for game-to-contract matching."""

from datetime import date

import pandas as pd

from src.data.mapping import (
    GameContractMapper,
    date_proximity_score,
    normalize_team,
    parse_teams_from_title,
)


def test_exact_team_name_match():
    assert normalize_team("Kansas City Chiefs", "NFL") == "KC"
    assert normalize_team("KC", "NFL") == "KC"


def test_alias_resolution():
    assert normalize_team("Chiefs", "NFL") == "KC"
    assert normalize_team("Warriors", "NBA") == "GSW"


def test_parse_teams_from_title():
    home, away = parse_teams_from_title("Kansas City Chiefs vs Philadelphia Eagles", "NFL")
    assert home == "KC"
    assert away == "PHI"


def test_date_proximity_threshold():
    game_date = date(2023, 1, 7)
    close = date_proximity_score(game_date, date(2023, 1, 7))
    far = date_proximity_score(game_date, date(2023, 1, 14))
    assert close == 1.0
    assert far == 0.0


def test_no_duplicate_matches(mapper: GameContractMapper, sample_games_df, sample_contracts_df):
    matches = mapper.match_all(games=sample_games_df, contracts=sample_contracts_df)
    dupes = matches.duplicated(subset=["game_id", "market_source"])
    assert not dupes.any()


def test_confidence_threshold(mapper: GameContractMapper, sample_games_df):
    low_conf_contract = pd.DataFrame(
        {
            "contract_id": ["BAD"],
            "market_source": ["kalshi"],
            "sport": ["NFL"],
            "home_team": ["XXX"],
            "away_team": ["YYY"],
            "game_date": [date(2020, 1, 1)],
            "contract_type": ["moneyline"],
            "resolved_outcome": [None],
            "resolution_price": [None],
        }
    )
    matches = mapper.match_all(games=sample_games_df, contracts=low_conf_contract)
    assert len(matches) == 0


def test_high_confidence_match(mapper: GameContractMapper, sample_games_df, sample_contracts_df):
    titles = {
        "KXNFL-KC-PHI-0115": "Kansas City Chiefs vs Philadelphia Eagles",
        "KXNFL-BUF-CIN-0107": "Buffalo Bills vs Cincinnati Bengals",
    }
    matches = mapper.match_all(
        games=sample_games_df,
        contracts=sample_contracts_df,
        contract_titles=titles,
    )
    assert len(matches) >= 1
    assert (matches["match_confidence"] > 0.8).all()
