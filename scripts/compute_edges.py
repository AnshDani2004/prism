#!/usr/bin/env python3
"""Compute and persist edge signals for all matched games."""

from src.data.database import PrismDatabase
from src.market.edge import EdgeCalculator
from src.models.inplay_xgb import XGBInPlayModel
from src.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    db = PrismDatabase()
    states = db.query_df("SELECT * FROM game_states")
    if states.empty:
        raise SystemExit("No game states found. Run ingest_sports.py first.")

    outcomes = XGBInPlayModel.home_win_outcomes(XGBInPlayModel.final_game_states(states))
    model = XGBInPlayModel()
    model.fit(states, outcomes)

    calc = EdgeCalculator(db=db)
    for source in ("kalshi", "polymarket"):
        edges = calc.compute_all_edges(model, market_source=source)
        print(f"{source}: {len(edges)} edge rows written")

    summary = db.phase4_checkpoint()
    print(summary)


if __name__ == "__main__":
    main()
