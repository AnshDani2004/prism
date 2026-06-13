"""
Polymarket historical data pipeline via Gamma Events API.
Correct endpoint: /events with tag= filter, NOT /markets.
"""
from __future__ import annotations
import json, logging, re, time
from datetime import date
from typing import TYPE_CHECKING, Any
import pandas as pd
import requests
from src.data.base import SportDataAdapter
if TYPE_CHECKING:
    from src.data.database import PrismDatabase

logger = logging.getLogger(__name__)
GAMMA_API = "https://gamma-api.polymarket.com"
REQUEST_DELAY = 0.5

NFL_NAME_TO_ABV = {
    "49ers": "SF", "bears": "CHI", "bengals": "CIN", "bills": "BUF",
    "broncos": "DEN", "browns": "CLE", "buccaneers": "TB", "bucs": "TB",
    "cardinals": "ARI", "chargers": "LAC", "chiefs": "KC", "colts": "IND",
    "commanders": "WAS", "cowboys": "DAL", "dolphins": "MIA", "eagles": "PHI",
    "falcons": "ATL", "giants": "NYG", "jaguars": "JAX", "jets": "NYJ",
    "lions": "DET", "packers": "GB", "panthers": "CAR", "patriots": "NE",
    "raiders": "LV", "rams": "LAR", "ravens": "BAL", "saints": "NO",
    "seahawks": "SEA", "steelers": "PIT", "texans": "HOU", "titans": "TEN",
    "vikings": "MIN", "washington": "WAS", "redskins": "WAS",
}
NBA_NAME_TO_ABV = {
    "76ers": "PHI", "bucks": "MIL", "bulls": "CHI", "cavaliers": "CLE",
    "cavs": "CLE", "celtics": "BOS", "clippers": "LAC", "grizzlies": "MEM",
    "hawks": "ATL", "heat": "MIA", "hornets": "CHA", "jazz": "UTA",
    "kings": "SAC", "knicks": "NYK", "lakers": "LAL", "magic": "ORL",
    "mavericks": "DAL", "mavs": "DAL", "nets": "BKN", "nuggets": "DEN",
    "pacers": "IND", "pelicans": "NOP", "pistons": "DET", "raptors": "TOR",
    "rockets": "HOU", "spurs": "SAS", "suns": "PHX", "thunder": "OKC",
    "timberwolves": "MIN", "wolves": "MIN", "trail blazers": "POR",
    "blazers": "POR", "warriors": "GSW", "wizards": "WAS",
}

TEAM_PATTERN = re.compile(
    r"Will (?:the )?(.+?) (?:beat|vs\.?|@|against|cover|win) (?:the )?(.+?)(?:\s+by|\s+in|\s+to|\s+win|\?|$)",
    re.IGNORECASE,
)


def name_to_abv(name: str, sport: str) -> str:
    lookup = NFL_NAME_TO_ABV if sport.upper() == "NFL" else NBA_NAME_TO_ABV
    name_lower = name.strip().lower()
    if name_lower in lookup:
        return lookup[name_lower]
    for key, abv in lookup.items():
        if key in name_lower or name_lower in key:
            return abv
    return name.strip().upper()[:3]


def extract_teams(title: str, sport: str) -> tuple[str, str]:
    clean = re.sub(r"^\(?In-Game Trading\)?\s*", "", title, flags=re.IGNORECASE)
    clean = re.sub(r"^(NFL|NBA):\s*", "", clean, flags=re.IGNORECASE)
    m = TEAM_PATTERN.search(clean)
    if m:
        t1 = name_to_abv(m.group(1).strip(), sport)
        t2 = name_to_abv(m.group(2).strip(), sport)
        return t1, t2
    return "", ""


def parse_end_date(date_str: str | None) -> date | None:
    if not date_str:
        return None
    try:
        return pd.to_datetime(date_str).date()
    except Exception:
        return None


class PolymarketAdapter(SportDataAdapter):
    sport = "POLYMARKET"

    def __init__(self, db: "PrismDatabase | None" = None) -> None:
        super().__init__(db)

    def load_pbp(self, seasons: list[int]) -> pd.DataFrame:
        return pd.DataFrame()

    def extract_game_states(self, pbp: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

    def validate_game_states(self, states: pd.DataFrame) -> bool:
        return True

    def get_resolved_sports_markets(
        self, sport: str, limit_per_page: int = 100, max_pages: int = 200
    ) -> list[dict[str, Any]]:
        tag = sport.lower()
        all_events: list[dict[str, Any]] = []
        offset = 0
        while offset < limit_per_page * max_pages:
            url = (
                f"{GAMMA_API}/events?closed=true&tag={tag}"
                f"&limit={limit_per_page}&offset={offset}"
            )
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                batch: list[dict[str, Any]] = resp.json()
                if not batch:
                    break
                all_events.extend(batch)
                logger.info(
                    "Polymarket %s: fetched %d events (total: %d)",
                    sport.upper(), len(batch), len(all_events),
                )
                if len(batch) < limit_per_page:
                    break
                offset += limit_per_page
                time.sleep(REQUEST_DELAY)
            except Exception as exc:
                logger.error(
                    "Polymarket events fetch failed at offset %d: %s", offset, exc
                )
                break
        return all_events

    def events_to_contracts(
        self, events: list[dict[str, Any]], sport: str
    ) -> pd.DataFrame:
        records = []
        for event in events:
            title = str(event.get("title", ""))
            end_date = parse_end_date(event.get("endDate"))
            if end_date is None:
                continue
            home_team, away_team = extract_teams(title, sport)
            resolved_outcome = None
            resolution_price = None
            markets = event.get("markets", [])
            if markets:
                mkt = markets[0]
                try:
                    prices = json.loads(mkt.get("outcomePrices", "[]"))
                    outcomes = json.loads(mkt.get("outcomes", "[]"))
                    if prices and outcomes:
                        yes_price = float(prices[0])
                        resolved_outcome = "yes" if yes_price > 0.5 else "no"
                        resolution_price = yes_price
                except Exception:
                    pass
            records.append({
                "contract_id": str(event.get("id", "")),
                "market_source": "polymarket",
                "sport": sport.upper(),
                "home_team": home_team,
                "away_team": away_team,
                "game_date": end_date,
                "contract_type": "spread_or_moneyline",
                "resolved_outcome": resolved_outcome,
                "resolution_price": resolution_price,
            })
        return pd.DataFrame(records) if records else pd.DataFrame()

    def ingest_sport(self, sport: str) -> pd.DataFrame:
        events = self.get_resolved_sports_markets(sport)
        if not events:
            logger.warning("No Polymarket events found for sport %s", sport)
            return pd.DataFrame()
        contracts = self.events_to_contracts(events, sport)
        logger.info(
            "Polymarket: loaded %d resolved %s contracts",
            len(contracts), sport.upper(),
        )
        return contracts

    def get_price_history(self, token_id: str) -> pd.DataFrame:
        if self.db is not None:
            cached = self.db.query_df(
                "SELECT timestamp, yes_price FROM market_prices "
                "WHERE contract_id = ? AND market_source = 'polymarket' "
                "ORDER BY timestamp",
                [token_id],
            )
            if not cached.empty:
                return cached
        url = (
            f"https://clob.polymarket.com/prices-history"
            f"?market={token_id}&interval=max&fidelity=60"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            history = data.get("history", [])
            if not history:
                return pd.DataFrame()
            df = pd.DataFrame(history)
            df = df.rename(columns={"t": "timestamp", "p": "yes_price"})
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df["contract_id"] = token_id
            df["market_source"] = "polymarket"
            df["no_price"] = 1.0 - df["yes_price"]
            df["volume"] = None
            if self.db is not None:
                self.db.upsert_dataframe(
                    "market_prices",
                    df[["contract_id", "market_source", "timestamp",
                        "yes_price", "no_price", "volume"]],
                    ["contract_id", "market_source", "timestamp"],
                )
            return df
        except Exception as exc:
            logger.error(
                "Polymarket price history failed for %s: %s", token_id, exc
            )
            return pd.DataFrame()