#!/usr/bin/env python3
"""Generate PRISM research notebook skeletons."""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"


def nb(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


NOTEBOOKS = {
    "01_data_pipeline.ipynb": [
        md("# PRISM Notebook 01: Data Pipeline\n\nLoad NFL/NBA play-by-play and market data from DuckDB. Visualize game states and sample prices."),
        code("import duckdb\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nDB = 'data/prism.duckdb'\ncon = duckdb.connect(DB)"),
        code("summary = {\n    'nfl_states': con.execute(\"SELECT COUNT(*) FROM game_states WHERE sport='NFL'\").fetchone()[0],\n    'nba_states': con.execute(\"SELECT COUNT(*) FROM game_states WHERE sport='NBA'\").fetchone()[0],\n    'kalshi': con.execute(\"SELECT COUNT(*) FROM contracts WHERE market_source='kalshi'\").fetchone()[0],\n    'polymarket': con.execute(\"SELECT COUNT(*) FROM contracts WHERE market_source='polymarket'\").fetchone()[0],\n    'matches': con.execute('SELECT COUNT(*) FROM game_contract_map WHERE match_confidence > 0.8').fetchone()[0],\n}\npd.Series(summary).to_frame('count')"),
        code("# Sample NFL game score timeline\ngame = con.execute(\"\"\"\n    SELECT seconds_remaining, home_score, away_score, is_scoring_event\n    FROM game_states WHERE sport='NFL'\n    ORDER BY game_id, seconds_remaining DESC LIMIT 30\n\"\"\").df()\n\nfig, ax = plt.subplots(figsize=(10,4))\nax.plot(game['seconds_remaining'], game['home_score'], label='Home')\nax.plot(game['seconds_remaining'], game['away_score'], label='Away')\nax.set_xlabel('Seconds Remaining'); ax.set_ylabel('Score'); ax.legend()\nax.set_title('Sample NFL Game Score Timeline'); plt.gca().invert_xaxis()\nplt.tight_layout(); plt.show()"),
        md("## Takeaway\n\nPhase 1 checkpoint: score consistency must be 0, >10k states per sport, >100 high-confidence matches."),
    ],
    "02_baseline_models.ipynb": [
        md("# PRISM Notebook 02: Baseline Models & Calibration\n\nFit Bradley-Terry and margin models. Explain ECE with reliability diagrams."),
        code("import pandas as pd\nfrom src.data.database import PrismDatabase\nfrom src.models.bradley_terry import BradleyTerryModel\nfrom src.models.margin_model import MarginRatingModel\nfrom src.models.calibration import CalibrationAnalyzer\n\ndb = PrismDatabase()\nstates = db.query_df('SELECT * FROM game_states')\nfinal = BradleyTerryModel.final_game_states(states)\noutcomes = BradleyTerryModel.home_win_outcomes(final)"),
        code("bt = BradleyTerryModel().fit(states, outcomes)\nmm = MarginRatingModel().fit(states, outcomes)\n\ntest = final[final['season'] == 2023]\ny = BradleyTerryModel.home_win_outcomes(test).to_numpy()\n\nbt_probs = bt.predict(test)\nmm_probs = mm.predict(test)\n\nanalyzer = CalibrationAnalyzer()\nprint('Bradley-Terry ECE:', analyzer.ece(bt_probs, y))\nprint('Margin Model ECE:', analyzer.ece(mm_probs, y))\nanalyzer.plot_reliability_diagram(mm_probs, y, model_name='margin_model', save=True)"),
        md("## What is ECE?\n\n**Expected Calibration Error** measures whether predicted 70% probabilities win ~70% of the time. Markets need calibration, not just ranking."),
    ],
    "03_bayesian_online_model.ipynb": [
        md("# PRISM Notebook 03: Bayesian Online Model\n\nStep through EKF updates on a single game. Compare to particle filter."),
        code("import pandas as pd\nimport matplotlib.pyplot as plt\nfrom src.data.database import PrismDatabase\nfrom src.models.bayesian_online import BayesianOnlineWinProb\n\ndb = PrismDatabase()\nstates = db.query_df(\"SELECT * FROM game_states WHERE sport='NFL' LIMIT 500\")\ngame_id = states['game_id'].iloc[0]\ngame = states[states['game_id'] == game_id]"),
        code("ekf = BayesianOnlineWinProb(inference='ekf', seed=42)\npf = BayesianOnlineWinProb(inference='particle', n_particles=500, seed=42)\nekf.fit_hyperparams(states)\npf.process_noise = ekf.process_noise\npf.observation_noise = ekf.observation_noise\npf.base_rate = ekf.base_rate\n\nekf_probs = ekf.replay_game(game)\npf_probs = pf.replay_game(game)\nprint(f'EKF final win prob: {ekf_probs[-1]:.3f}')\nprint(f'PF  final win prob: {pf_probs[-1]:.3f}')"),
        code("events = BayesianOnlineWinProb.extract_scoring_events(game)\nprint(f'Scoring events: {len(events)}')\nfor e in events:\n    print(f'  {e.team} +{e.points} at t={e.time_elapsed:.0f}s, diff={e.score_diff}')"),
        md("## Connection to Prior Work\n\nThe particle filter here uses the same Sequential Monte Carlo algorithm as **bayesian-market-filters**, applied to sports state spaces instead of market microstructure."),
    ],
    "04_market_interface.ipynb": [
        md("# PRISM Notebook 04: Market Interface\n\nAlign model probability to market price. Visualize staleness after scoring."),
        code("import pandas as pd\nimport matplotlib.pyplot as plt\nfrom src.data.database import PrismDatabase\nfrom src.market.interface import ContractResolver\nfrom src.market.edge import EdgeCalculator\nfrom src.models.inplay_xgb import XGBInPlayModel\n\ndb = PrismDatabase()"),
        code("mapping = db.query_df('SELECT * FROM game_contract_map WHERE match_confidence > 0.8 LIMIT 1')\nif mapping.empty:\n    raise SystemExit('No mappings — run ingest_markets.py')\n\nrow = mapping.iloc[0]\nresolver = ContractResolver(db=db)\naligned = resolver.align_prices_to_game_states(row['game_id'], row['contract_id'], row['market_source'])\naligned.head()"),
        code("states = db.query_df('SELECT * FROM game_states')\nmodel = XGBInPlayModel().fit(states, XGBInPlayModel.home_win_outcomes(XGBInPlayModel.final_game_states(states)))\ncalc = EdgeCalculator(db=db)\nedges = calc.compute_edge_series(row['game_id'], model, row['market_source'], row['contract_id'])\n\nfig, ax = plt.subplots(figsize=(10,4))\nax.plot(edges['seconds_remaining'], edges['model_prob'], label='Model')\nax.plot(edges['seconds_remaining'], edges['market_price'], label='Market')\nax.set_xlabel('Seconds Remaining'); ax.set_ylabel('P(home win)'); ax.legend()\nax.set_title('Model vs Market In-Play'); plt.gca().invert_xaxis(); plt.show()"),
        md("## Staleness Hypothesis\n\nEdge should concentrate in `inplay_staleness` (≤30s after score) if markets are slow to reprice."),
    ],
    "05_alpha_backtest.ipynb": [
        md("# PRISM Notebook 05: Alpha Backtest\n\nFull walk-forward backtest with honest fees, spreads, and significance tests."),
        code("import pandas as pd\nfrom src.data.database import PrismDatabase\nfrom src.backtest.engine import PredictionMarketBacktester\nfrom src.backtest.metrics import BacktestMetrics\nfrom src.backtest.sizing import KellySizer\nfrom src.models.inplay_xgb import XGBInPlayModel\n\ndb = PrismDatabase()\nedges = db.query_df('SELECT * FROM edge_signals')\nprices = db.query_df('SELECT * FROM market_prices')\nstates = db.query_df('SELECT * FROM game_states')"),
        code("final = XGBInPlayModel.final_game_states(states)\noutcomes = pd.DataFrame({'game_id': final['game_id'], 'home_won': XGBInPlayModel.home_win_outcomes(final).astype(bool)})\nmeta = states[['game_id','game_date','sport','seconds_remaining']].drop_duplicates()\nedges = edges.merge(meta, on=['game_id','seconds_remaining'], how='left')\n\nbt = PredictionMarketBacktester(edge_threshold=0.05)\nresults = bt.run(edges, prices, outcomes, KellySizer())\nmetrics = BacktestMetrics().compute_all(results)\nmetrics_obj = BacktestMetrics(); metrics_obj.save_plots(results)"),
        code("for k in ['total_return','sharpe_ratio','max_drawdown','hit_rate','n_trades','bootstrap_ci_sharpe','deflated_sharpe_ratio']:\n    print(f'{k}: {metrics[k]}')"),
        md("## Honest Reporting\n\nIf 95% CI Sharpe includes zero, the strategy is **not** statistically significant. Report it anyway."),
    ],
    "06_findings_summary.ipynb": [
        md("# PRISM Notebook 06: Findings Summary\n\n**2-minute interview walkthrough.** Run after full pipeline."),
        code("from src.data.database import PrismDatabase\nfrom src.backtest.metrics import BacktestMetrics\n\n# Run: python scripts/phase1_checkpoint.py through phase5_checkpoint.py first\ndb = PrismDatabase()\nprint('=== PRISM Summary ===')\nprint(db.phase1_checkpoint())\nprint(db.phase4_checkpoint())"),
        code("print('''\nRESEARCH QUESTION:\n  Do in-play prediction market prices lag after scoring events?\n\nMETHODOLOGY:\n  Calibrated models (BT → Elo-MOV → XGBoost → Bayesian online)\n  vs Kalshi/Polymarket mid-prices, backtested at touch with Kalshi fees.\n\nKEY DESIGN CHOICES:\n  - 2023 held out as sacred test set\n  - Block-bootstrap Sharpe CI by game\n  - Deflated Sharpe from experiment_log.jsonl\n  - Diebold-Mariano vs market-as-forecast\n\nNEXT: Fill X.XX from phase5_checkpoint output into README resume bullets.\n''')"),
        md("## Resume Bullets (fill after backtest)\n\n```\n- Built sport-agnostic prediction market pricing engine ... ECE of X.XX vs XGBoost X.XX\n- Detected price staleness of ~X seconds; Sharpe X.XX (95% CI: [X, X])\n- Engineered pipeline: nfl_data_py + nba_api + Kalshi + Polymarket → DuckDB\n```"),
    ],
}


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        path = NOTEBOOKS_DIR / name
        path.write_text(json.dumps(nb(cells), indent=1))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
