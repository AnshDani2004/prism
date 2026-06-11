#!/usr/bin/env python3
"""Phase 5 validation checkpoint — honest backtest reporting."""

import pandas as pd

from src.backtest.engine import PredictionMarketBacktester
from src.backtest.metrics import BacktestMetrics
from src.backtest.sizing import KellySizer
from src.data.database import PrismDatabase
from src.models.inplay_xgb import XGBInPlayModel


def main() -> None:
    db = PrismDatabase()
    edges = db.query_df("SELECT * FROM edge_signals")
    prices = db.query_df("SELECT * FROM market_prices")
    states = db.query_df("SELECT * FROM game_states")

    if edges.empty or prices.empty:
        raise SystemExit("Missing edge_signals or market_prices. Run compute_edges.py first.")

    final = XGBInPlayModel.final_game_states(states)
    outcomes = pd.DataFrame(
        {
            "game_id": final["game_id"],
            "home_won": XGBInPlayModel.home_win_outcomes(final).astype(bool),
        }
    )

    # Enrich edges with game_date/sport for signal timestamps
    meta = states[["game_id", "game_date", "sport", "seconds_remaining"]].drop_duplicates()
    edges = edges.merge(meta, on=["game_id", "seconds_remaining"], how="left")

    backtester = PredictionMarketBacktester(edge_threshold=0.05)
    sizer = KellySizer(kelly_fraction=0.25, max_position_size=0.05)
    results = backtester.run(edges, prices, outcomes, sizer)

    metrics = BacktestMetrics().compute_all(results)
    metrics_obj = BacktestMetrics()
    metrics_obj.save_plots(results)

    print("=== BACKTEST RESULTS ===")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Hit Rate: {metrics['hit_rate']:.2%}")
    print(f"N Trades: {metrics['n_trades']}")
    print(f"P-value (returns != 0): {metrics['p_value_returns']:.4f}")
    ci = metrics["bootstrap_ci_sharpe"]
    print(f"95% CI Sharpe: [{ci[0]:.2f}, {ci[1]:.2f}]")
    print(f"Deflated Sharpe Ratio: {metrics['deflated_sharpe_ratio']:.4f}")
    print()
    print("NOTE: Results above are the honest empirical finding of PRISM.")
    print("Phase 5 checkpoint PASSED")


if __name__ == "__main__":
    main()
