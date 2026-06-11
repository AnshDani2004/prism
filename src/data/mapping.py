"""
Fuzzy matching between game states and prediction market contracts.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from config.settings import Settings
from src.data.database import PrismDatabase

logger = logging.getLogger(__name__)

# Team alias dictionaries — abbreviation is canonical
NFL_ALIASES: dict[str, str] = {
    "chiefs": "KC",
    "kansas city": "KC",
    "kansas city chiefs": "KC",
    "eagles": "PHI",
    "philadelphia": "PHI",
    "philadelphia eagles": "PHI",
    "49ers": "SF",
    "niners": "SF",
    "san francisco": "SF",
    "cowboys": "DAL",
    "dallas": "DAL",
    "bills": "BUF",
    "buffalo": "BUF",
    "ravens": "BAL",
    "baltimore": "BAL",
    "bengals": "CIN",
    "cincinnati": "CIN",
    "dolphins": "MIA",
    "miami": "MIA",
    "jets": "NYJ",
    "new york jets": "NYJ",
    "giants": "NYG",
    "new york giants": "NYG",
    "patriots": "NE",
    "new england": "NE",
    "steelers": "PIT",
    "pittsburgh": "PIT",
    "browns": "CLE",
    "cleveland": "CLE",
    "titans": "TEN",
    "tennessee": "TEN",
    "colts": "IND",
    "indianapolis": "IND",
    "jaguars": "JAX",
    "jacksonville": "JAX",
    "texans": "HOU",
    "houston": "HOU",
    "broncos": "DEN",
    "denver": "DEN",
    "raiders": "LV",
    "las vegas": "LV",
    "chargers": "LAC",
    "los angeles chargers": "LAC",
    "rams": "LAR",
    "los angeles rams": "LAR",
    "seahawks": "SEA",
    "seattle": "SEA",
    "cardinals": "ARI",
    "arizona": "ARI",
    "packers": "GB",
    "green bay": "GB",
    "vikings": "MIN",
    "minnesota": "MIN",
    "bears": "CHI",
    "chicago": "CHI",
    "lions": "DET",
    "detroit": "DET",
    "saints": "NO",
    "new orleans": "NO",
    "falcons": "ATL",
    "atlanta": "ATL",
    "panthers": "CAR",
    "carolina": "CAR",
    "buccaneers": "TB",
    "bucs": "TB",
    "tampa bay": "TB",
    "commanders": "WAS",
    "washington": "WAS",
}

NBA_ALIASES: dict[str, str] = {
    "warriors": "GSW",
    "golden state": "GSW",
    "lakers": "LAL",
    "los angeles lakers": "LAL",
    "celtics": "BOS",
    "boston": "BOS",
    "heat": "MIA",
    "miami": "MIA",
    "bucks": "MIL",
    "milwaukee": "MIL",
    "nuggets": "DEN",
    "denver": "DEN",
    "suns": "PHX",
    "phoenix": "PHX",
    "mavericks": "DAL",
    "dallas": "DAL",
    "knicks": "NYK",
    "new york": "NYK",
    "nets": "BKN",
    "brooklyn": "BKN",
    "76ers": "PHI",
    "sixers": "PHI",
    "philadelphia": "PHI",
    "clippers": "LAC",
    "la clippers": "LAC",
    "thunder": "OKC",
    "oklahoma city": "OKC",
    "timberwolves": "MIN",
    "minnesota": "MIN",
    "kings": "SAC",
    "sacramento": "SAC",
    "pelicans": "NOP",
    "new orleans": "NOP",
    "grizzlies": "MEM",
    "memphis": "MEM",
    "spurs": "SAS",
    "san antonio": "SAS",
    "rockets": "HOU",
    "houston": "HOU",
    "jazz": "UTA",
    "utah": "UTA",
    "blazers": "POR",
    "portland": "POR",
    "hawks": "ATL",
    "atlanta": "ATL",
    "bulls": "CHI",
    "chicago": "CHI",
    "cavaliers": "CLE",
    "cavs": "CLE",
    "cleveland": "CLE",
    "pistons": "DET",
    "detroit": "DET",
    "pacers": "IND",
    "indiana": "IND",
    "hornets": "CHA",
    "charlotte": "CHA",
    "magic": "ORL",
    "orlando": "ORL",
    "wizards": "WAS",
    "washington": "WAS",
    "raptors": "TOR",
    "toronto": "TOR",
}


def normalize_team(name: str, sport: str) -> str:
    """Resolve team name to canonical abbreviation."""
    if not name:
        return ""
    text = name.strip().lower()
    aliases = NFL_ALIASES if sport.upper() == "NFL" else NBA_ALIASES
    if text.upper() in {v for v in aliases.values()}:
        return text.upper()
    if text in aliases:
        return aliases[text]
    # Direct abbreviation match
    if len(text) <= 4 and text.upper() == text.upper():
        return text.upper()
    for alias, abbr in aliases.items():
        if alias in text:
            return abbr
    return name.strip().upper()[:4]


def parse_teams_from_title(title: str, sport: str) -> tuple[str, str]:
    """Extract home/away team abbreviations from contract title."""
    text = title.lower()
    # Common patterns: "Team A vs Team B", "Team A @ Team B", "Team A to beat Team B"
    vs_match = re.search(r"(.+?)\s+(?:vs\.?|v\.?|@|at|to beat)\s+(.+)", text, re.I)
    if vs_match:
        team_a = normalize_team(vs_match.group(1).strip(), sport)
        team_b = normalize_team(vs_match.group(2).strip(), sport)
        return team_a, team_b
    return "", ""


def date_proximity_score(game_date: date, contract_date: date, max_days: int = 3) -> float:
    """Score date match in [0, 1]; 1.0 for same day, decays linearly."""
    delta = abs((game_date - contract_date).days)
    if delta > max_days:
        return 0.0
    return 1.0 - delta / max_days


def team_match_score(
    game_home: str,
    game_away: str,
    c_home: str,
    c_away: str,
    sport: str = "NFL",
) -> float:
    """Score team name match allowing home/away swap."""
    gh, ga = normalize_team(game_home, sport), normalize_team(game_away, sport)
    ch, ca = c_home.upper(), c_away.upper()
    direct = (fuzz.ratio(gh, ch) + fuzz.ratio(ga, ca)) / 200
    swapped = (fuzz.ratio(gh, ca) + fuzz.ratio(ga, ch)) / 200
    return max(direct, swapped)


class GameContractMapper:
    """Match games to market contracts with confidence scoring."""

    def __init__(self, db: PrismDatabase | None = None, settings: Settings | None = None):
        self.db = db or PrismDatabase()
        self.settings = settings or Settings()

    def score_pair(
        self,
        game: pd.Series,
        contract: pd.Series,
        title: str = "",
    ) -> tuple[float, str]:
        """Score a (game, contract) pair. Returns (confidence, method)."""
        sport = str(game["sport"])
        if title:
            c_home, c_away = parse_teams_from_title(title, sport)
        else:
            c_home = str(contract.get("home_team", "") or "")
            c_away = str(contract.get("away_team", "") or "")
            if not c_home and not c_away:
                c_home, c_away = parse_teams_from_title(
                    str(contract.get("contract_id", "")), sport
                )

        team_score = team_match_score(
            str(game["home_team"]),
            str(game["away_team"]),
            c_home,
            c_away,
            sport=sport,
        )
        game_date = pd.to_datetime(game["game_date"]).date()
        contract_date = pd.to_datetime(contract["game_date"]).date()
        date_score = date_proximity_score(game_date, contract_date)

        confidence = 0.7 * team_score + 0.3 * date_score
        if team_score == 1.0 and date_score == 1.0:
            method = "exact"
        elif team_score >= 0.9:
            method = "fuzzy_team"
        else:
            method = "fuzzy_date"
        return confidence, method

    def match_all(
        self,
        games: pd.DataFrame | None = None,
        contracts: pd.DataFrame | None = None,
        contract_titles: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """
        Match games to contracts above confidence threshold.

        Returns high-confidence matches; logs ambiguous and unmatched counts.
        """
        if games is None:
            games = self.db.query_df(
                "SELECT DISTINCT game_id, sport, game_date, home_team, away_team FROM game_states"
            )
        if contracts is None:
            contracts = self.db.query_df("SELECT * FROM contracts")

        contract_titles = contract_titles or {}
        threshold = self.settings.match_confidence_threshold
        ambiguous_low = self.settings.ambiguous_match_lower_bound

        matches: list[dict[str, Any]] = []
        ambiguous = 0
        unmatched_contracts = 0

        for market_source, source_contracts in contracts.groupby("market_source"):
            for _, contract in source_contracts.iterrows():
                cid = str(contract["contract_id"])
                title = contract_titles.get(cid, "")
                best_conf = 0.0
                best_game: pd.Series | None = None
                best_method = ""

                sport_contracts = games[games["sport"] == contract["sport"]]
                for _, game in sport_contracts.iterrows():
                    conf, method = self.score_pair(game, contract, title=title)
                    if conf > best_conf:
                        best_conf = conf
                        best_game = game
                        best_method = method

                if best_game is None or best_conf < ambiguous_low:
                    unmatched_contracts += 1
                    continue
                if ambiguous_low <= best_conf < threshold:
                    ambiguous += 1
                    continue

                matches.append(
                    {
                        "game_id": best_game["game_id"],
                        "contract_id": cid,
                        "market_source": market_source,
                        "match_confidence": best_conf,
                        "match_method": best_method,
                    }
                )

        result = pd.DataFrame(matches)
        if not result.empty:
            # Enforce: no game maps to more than one contract per source
            result = self._dedupe_conflicts(result)
            self.db.upsert_dataframe(
                "game_contract_map",
                result,
                primary_key=["game_id", "contract_id", "market_source"],
            )

        logger.info(
            "Mapping complete: %d matched, %d ambiguous, %d unmatched contracts",
            len(result),
            ambiguous,
            unmatched_contracts,
        )
        return result

    @staticmethod
    def _dedupe_conflicts(matches: pd.DataFrame) -> pd.DataFrame:
        """Keep highest-confidence match when a game has multiple contracts per source."""
        return (
            matches.sort_values("match_confidence", ascending=False)
            .drop_duplicates(subset=["game_id", "market_source"], keep="first")
            .reset_index(drop=True)
        )
