#!/usr/bin/env python3
"""
Reproduce PRISM's main finding in five lines of code.

Requires data/prism.duckdb populated via ingest scripts.
Prints honest backtest metrics — null results are valid.
"""

import pandas as pd

from src.backtest.engine import PredictionMarketBacktester
from src.backtest.metrics import BacktestMetrics
from src.backtest.sizing import KellySizer
from src.data.database import PrismDatabase
from src.market.edge import EdgeCalculator
from src.models.inplay_xgb import XGBInPlayModel

db = PrismDatabase()
states = db.query_df("SELECT * FROM game_states")
edges = db.query_df("SELECT * FROM edge_signals")

if edges.empty:
    outcomes = XGBInPlayModel.home_win_outcomes(XGBInPlayModel.final_game_states(states))
    model = XGBInPlayModel().fit(states, outcomes)
    EdgeCalculator(db=db).compute_all_edges(model, "kalshi")
    edges = db.query_df("SELECT * FROM edge_signals")

prices = db.query_df("SELECT * FROM market_prices")
final = XGBInPlayModel.final_game_states(states)
outcomes = pd.DataFrame({"game_id": final["game_id"], "home_won": XGBInPlayModel.home_win_outcomes(final).astype(bool)})
meta = states[["game_id", "game_date", "sport", "seconds_remaining"]].drop_duplicates()
edges = edges.merge(meta, on=["game_id", "seconds_remaining"], how="left")

results = PredictionMarketBacktester().run(edges, prices, outcomes, KellySizer())
metrics = BacktestMetrics().compute_all(results)
print(f"Sharpe={metrics['sharpe_ratio']:.2f}  CI={metrics['bootstrap_ci_sharpe']}  Trades={metrics['n_trades']}")
