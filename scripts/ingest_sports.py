#!/usr/bin/env python3
"""One-time sports data ingestion script."""

from config.settings import Settings
from src.data.nba import NBAAdapter
from src.data.nfl import NFLAdapter
from src.utils.logging import setup_logging


def main() -> None:
    settings = Settings()
    setup_logging(settings)

    nfl = NFLAdapter()
    nfl.ingest(settings.nfl_seasons)

    nba = NBAAdapter()
    nba.ingest(settings.nba_seasons)

    results = nfl.db.phase1_checkpoint()
    print("Sports ingest complete:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
