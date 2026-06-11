#!/usr/bin/env python3
"""Phase 4 validation checkpoint."""

from src.data.database import PrismDatabase
from src.market.edge import EdgeCalculator
from src.models.inplay_xgb import XGBInPlayModel


def main() -> None:
    db = PrismDatabase()
    states = db.query_df("SELECT * FROM game_states")
    if states.empty:
        raise SystemExit("No game_states. Run ingest_sports.py first.")

    mappings = db.count("game_contract_map", "match_confidence > 0.8")
    if mappings == 0:
        raise SystemExit("No high-confidence mappings. Run ingest_markets.py first.")

    outcomes = XGBInPlayModel.home_win_outcomes(XGBInPlayModel.final_game_states(states))
    model = XGBInPlayModel()
    model.fit(states, outcomes)

    calc = EdgeCalculator(db=db)
    for source in ("kalshi", "polymarket"):
        edges = calc.compute_all_edges(model, market_source=source)
        print(f"{source}: {len(edges)} edge observations")

    summary = db.phase4_checkpoint()
    print(summary)

    if summary.empty:
        raise SystemExit("No edge signals computed")

    total_obs = int(summary["n_observations"].sum())
    assert total_obs > 500, f"Insufficient edge signal observations: {total_obs}"

    print("Phase 4 checkpoint PASSED")


if __name__ == "__main__":
    main()
