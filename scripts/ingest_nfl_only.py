#!/usr/bin/env python3
"""Fast path: ingest NFL only (skips slow NBA API)."""

from config.settings import Settings
from src.data.nfl import NFLAdapter
from src.utils.logging import setup_logging


def main() -> None:
    settings = Settings()
    setup_logging(settings)
    nfl = NFLAdapter()
    nfl.ingest(settings.nfl_seasons)
    print(nfl.db.phase1_checkpoint())


if __name__ == "__main__":
    main()
