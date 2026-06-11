"""Tests for event-driven backtester."""

from datetime import timedelta

import pandas as pd
import pytest

from src.backtest.engine import PredictionMarketBacktester
from src.backtest.sizing import KellySizer


def test_kalshi_fee_maximal_at_half(backtester: PredictionMarketBacktester):
    fee_half = backtester.kalshi_fee(100, 0.50)
    fee_low = backtester.kalshi_fee(100, 0.10)
    fee_high = backtester.kalshi_fee(100, 0.90)
    assert fee_half > fee_low
    assert fee_half > fee_high


def test_kalshi_fee_ceil_to_cent(backtester: PredictionMarketBacktester):
    fee = backtester.kalshi_fee(1, 0.50, fee_rate=0.07)
    assert fee == 0.02  # ceil(0.0175 * 100) / 100


def test_no_lookahead_bias(
    backtester: PredictionMarketBacktester,
    sample_edge_signals: pd.DataFrame,
    sample_market_prices_bt: pd.DataFrame,
    sample_game_outcomes: pd.DataFrame,
    sizer: KellySizer,
):
    results = backtester.run(
        sample_edge_signals, sample_market_prices_bt, sample_game_outcomes, sizer
    )
    delay = timedelta(seconds=backtester.entry_delay_seconds)
    for trade in results.trades:
        assert trade.execution_time >= trade.signal_time + delay


def test_fees_applied(
    backtester: PredictionMarketBacktester,
    sample_edge_signals: pd.DataFrame,
    sample_market_prices_bt: pd.DataFrame,
    sample_game_outcomes: pd.DataFrame,
    sizer: KellySizer,
):
    with_fees = backtester.run(
        sample_edge_signals, sample_market_prices_bt, sample_game_outcomes, sizer, fee_rate=0.07
    )
    no_fees = backtester.run(
        sample_edge_signals, sample_market_prices_bt, sample_game_outcomes, sizer, fee_rate=0.0
    )
    assert with_fees.total_pnl < no_fees.total_pnl


def test_fills_at_touch_not_mid(
    backtester: PredictionMarketBacktester,
    sample_edge_signals: pd.DataFrame,
    sample_market_prices_bt: pd.DataFrame,
    sample_game_outcomes: pd.DataFrame,
    sizer: KellySizer,
):
    tight = backtester.run(
        sample_edge_signals,
        sample_market_prices_bt,
        sample_game_outcomes,
        sizer,
        spread_override=0.01,
    )
    wide = backtester.run(
        sample_edge_signals,
        sample_market_prices_bt,
        sample_game_outcomes,
        sizer,
        spread_override=0.10,
    )
    assert wide.total_pnl < tight.total_pnl


def test_volume_participation_cap(backtester: PredictionMarketBacktester, sizer: KellySizer):
    n, _ = backtester.simulate_trade(
        capital=100_000,
        model_prob=0.70,
        fill_price=0.50,
        side="buy_yes",
        sizer=sizer,
        touch_volume=200,
    )
    assert n <= int(0.10 * 200)


def test_empty_signals(backtester: PredictionMarketBacktester, sizer: KellySizer):
    results = backtester.run(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame({"game_id": [], "home_won": []}),
        sizer,
    )
    assert results.total_pnl == 0.0
    assert len(results.trades) == 0


def test_walk_forward_no_overlap(backtester: PredictionMarketBacktester):
    dates = pd.Series(pd.date_range("2020-01-01", periods=100, freq="W"))
    folds = backtester.build_walk_forward_folds(dates, n_folds=3)
    for fold in folds:
        assert fold.train_end < fold.test_start


def test_settle_buy_yes_win(backtester: PredictionMarketBacktester):
    pnl = backtester._settle_trade("buy_yes", 10, 0.50, fee=0.10, home_won=True)
    assert pnl > 0


def test_settle_buy_yes_loss(backtester: PredictionMarketBacktester):
    pnl = backtester._settle_trade("buy_yes", 10, 0.50, fee=0.10, home_won=False)
    assert pnl < 0


def test_settle_sell_yes(backtester: PredictionMarketBacktester):
    pnl_win = backtester._settle_trade("sell_yes", 10, 0.50, fee=0.10, home_won=False)
    pnl_loss = backtester._settle_trade("sell_yes", 10, 0.50, fee=0.10, home_won=True)
    assert pnl_win > 0
    assert pnl_loss < 0


def test_experiment_log_written(
    backtester: PredictionMarketBacktester,
    sample_edge_signals: pd.DataFrame,
    sample_market_prices_bt: pd.DataFrame,
    sample_game_outcomes: pd.DataFrame,
    sizer: KellySizer,
    tmp_path,
):
    backtester.experiment_log_path = tmp_path / "experiment_log.jsonl"
    backtester.run(sample_edge_signals, sample_market_prices_bt, sample_game_outcomes, sizer)
    assert backtester.experiment_log_path.exists()
