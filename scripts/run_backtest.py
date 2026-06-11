#!/usr/bin/env python3
"""Run full backtest pipeline on stored edge signals."""

import pandas as pd

from src.backtest.engine import PredictionMarketBacktester
from src.backtest.metrics import BacktestMetrics
from src.backtest.sizing import KellySizer
from src.data.database import PrismDatabase
from src.models.inplay_xgb import XGBInPlayModel
from src.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    db = PrismDatabase()
    edges = db.query_df("SELECT * FROM edge_signals")
    prices = db.query_df("SELECT * FROM market_prices")
    states = db.query_df("SELECT * FROM game_states")

    if edges.empty:
        raise SystemExit("No edge signals. Run scripts/compute_edges.py first.")

    final = XGBInPlayModel.final_game_states(states)
    outcomes = pd.DataFrame(
        {
            "game_id": final["game_id"],
            "home_won": XGBInPlayModel.home_win_outcomes(final).astype(bool),
        }
    )
    meta = states[["game_id", "game_date", "sport", "seconds_remaining"]].drop_duplicates()
    edges = edges.merge(meta, on=["game_id", "seconds_remaining"], how="left")

    backtester = PredictionMarketBacktester()
    sizer = KellySizer()
    results = backtester.run(edges, prices, outcomes, sizer)

    metrics = BacktestMetrics()
    report = metrics.compute_all(results)
    metrics.save_plots(results)

    print("=== BACKTEST REPORT ===")
    for key, val in report.items():
        if key not in {"brier_decomposition", "sharpe_by_time_of_game"}:
            print(f"{key}: {val}")


if __name__ == "__main__":
    main()
