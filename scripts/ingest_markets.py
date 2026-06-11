#!/usr/bin/env python3
"""One-time market data ingestion and game-contract matching."""

from config.settings import Settings
from src.data.kalshi import KalshiAdapter, KalshiAuthError
from src.data.mapping import GameContractMapper
from src.data.polymarket import PolymarketAdapter
from src.utils.logging import setup_logging


def main() -> None:
    settings = Settings()
    logger = setup_logging(settings)

    polymarket = PolymarketAdapter()
    for sport in ("NFL", "NBA"):
        polymarket.get_resolved_sports_markets(sport)

    try:
        kalshi = KalshiAdapter()
        for sport in ("NFL", "NBA"):
            for season in settings.nfl_seasons:
                kalshi.get_sports_markets(sport, season)
    except KalshiAuthError as exc:
        logger.warning("Skipping Kalshi ingest (credentials not configured): %s", exc)

    mapper = GameContractMapper()
    matches = mapper.match_all()
    logger.info("Matched %d game-contract pairs", len(matches))

    results = polymarket.db.phase1_checkpoint()
    print("Market ingest complete:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
