"""Tests for backtest performance metrics."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestResults, Trade
from src.backtest.metrics import BacktestMetrics


@pytest.fixture
def metrics() -> BacktestMetrics:
    return BacktestMetrics()


@pytest.fixture
def sample_results() -> BacktestResults:
    trades = [
        Trade(
            game_id="g1",
            contract_id="c1",
            signal_time=pd.Timestamp("2023-01-15 19:00", tz="UTC"),
            execution_time=pd.Timestamp("2023-01-15 19:00:05", tz="UTC"),
            side="buy_yes",
            n_contracts=10,
            signal_price=0.55,
            fill_price=0.56,
            fee=0.10,
            model_prob=0.65,
            edge=0.10,
            edge_type="inplay_staleness",
            home_won=True,
            pnl=4.0,
        ),
        Trade(
            game_id="g2",
            contract_id="c2",
            signal_time=pd.Timestamp("2023-01-16 19:00", tz="UTC"),
            execution_time=pd.Timestamp("2023-01-16 19:00:05", tz="UTC"),
            side="sell_yes",
            n_contracts=10,
            signal_price=0.50,
            fill_price=0.49,
            fee=0.10,
            model_prob=0.40,
            edge=-0.10,
            edge_type="inplay_drift",
            home_won=False,
            pnl=3.0,
        ),
        Trade(
            game_id="g1",
            contract_id="c1",
            signal_time=pd.Timestamp("2023-01-15 20:00", tz="UTC"),
            execution_time=pd.Timestamp("2023-01-15 20:00:05", tz="UTC"),
            side="buy_yes",
            n_contracts=5,
            signal_price=0.60,
            fill_price=0.61,
            fee=0.05,
            model_prob=0.70,
            edge=0.10,
            edge_type="inplay_staleness",
            home_won=True,
            pnl=1.5,
        ),
    ]
    equity = pd.DataFrame(
        {
            "timestamp": [t.execution_time for t in trades],
            "capital": [10_004.0, 10_007.0, 10_008.5],
            "pnl": [t.pnl for t in trades],
            "game_id": [t.game_id for t in trades],
        }
    )
    return BacktestResults(trades=trades, equity_curve=equity, initial_capital=10_000.0)


def test_compute_all(metrics: BacktestMetrics, sample_results: BacktestResults):
    m = metrics.compute_all(sample_results)
    assert "sharpe_ratio" in m
    assert "max_drawdown" in m
    assert m["n_trades"] == 3
    assert m["hit_rate"] == 1.0


def test_bootstrap_ci_computed(metrics: BacktestMetrics, sample_results: BacktestResults):
    trades_df = metrics._trades_df(sample_results)
    ci = metrics.block_bootstrap_sharpe_ci(trades_df, n_bootstrap=200, seed=42)
    assert len(ci) == 2
    assert ci[0] <= ci[1]


def test_deflated_sharpe(metrics: BacktestMetrics):
    dsr = metrics.deflated_sharpe(sharpe=1.5, n_trials=10, n_obs=100, skew=0.0, kurt=3.0)
    assert 0 <= dsr <= 1


def test_diebold_mariano(metrics: BacktestMetrics):
    outcomes = np.array([1, 0, 1, 1, 0])
    model_probs = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
    market_probs = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    dm = metrics.diebold_mariano_vs_market(model_probs, market_probs, outcomes)
    assert "dm_statistic" in dm
    assert "p_value" in dm
    assert dm["p_value"] < 1.0


def test_brier_decomposition(metrics: BacktestMetrics):
    probs = np.array([0.2, 0.8, 0.6, 0.4])
    outcomes = np.array([0, 1, 1, 0])
    decomp = metrics.brier_decomposition(probs, outcomes)
    assert abs(decomp["brier"] - (decomp["reliability"] - decomp["resolution"] + decomp["uncertainty"])) < 0.05


def test_empty_results(metrics: BacktestMetrics):
    m = metrics.compute_all(BacktestResults())
    assert m["n_trades"] == 0
    assert m["total_return"] == 0.0


def test_save_plots(metrics: BacktestMetrics, sample_results: BacktestResults, tmp_path):
    metrics.save_plots(sample_results, output_dir=tmp_path)
    assert (tmp_path / "equity_curve.png").exists()
    assert (tmp_path / "drawdown.png").exists()
    assert (tmp_path / "trade_pnl_hist.png").exists()


def test_pbo_proxy(metrics: BacktestMetrics):
    returns = np.array([0.1, -0.2, 0.05, -0.1, 0.3, -0.4, 0.2, -0.3])
    pbo = metrics._prob_backtest_overfitting(returns, n_partitions=4)
    assert 0 <= pbo <= 1
