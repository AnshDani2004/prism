"""Performance analytics and statistical significance tests."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest.engine import BacktestResults, EXPERIMENT_LOG
from src.models.calibration import CalibrationAnalyzer

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.05  # annualized


class BacktestMetrics:
    """Rigorous backtest performance and significance metrics."""

    def __init__(
        self,
        experiment_log_path: Path | str = EXPERIMENT_LOG,
        risk_free_rate: float = RISK_FREE_RATE,
    ) -> None:
        self.experiment_log_path = Path(experiment_log_path)
        self.risk_free_rate = risk_free_rate

    def compute_all(self, results: BacktestResults) -> dict[str, object]:
        """Compute full metrics dictionary."""
        trades_df = self._trades_df(results)
        returns = trades_df["return_pct"].to_numpy() if not trades_df.empty else np.array([])

        sharpe = self._sharpe_ratio(returns)
        block_ci = self.block_bootstrap_sharpe_ci(trades_df) if not trades_df.empty else (0.0, 0.0)

        metrics: dict[str, object] = {
            "total_return": results.total_pnl / results.initial_capital,
            "annualized_return": self._annualized_return(results),
            "sharpe_ratio": sharpe,
            "sortino_ratio": self._sortino_ratio(returns),
            "max_drawdown": self._max_drawdown(results),
            "max_drawdown_duration_days": self._max_drawdown_duration(results),
            "value_at_risk_95": self._var_95(returns),
            "n_trades": len(results.trades),
            "hit_rate": self._hit_rate(results),
            "avg_profit_per_trade": self._avg_win(results),
            "avg_loss_per_trade": self._avg_loss(results),
            "profit_factor": self._profit_factor(results),
            "p_value_returns": self._p_value_returns(returns),
            "bootstrap_ci_sharpe": block_ci,
            "deflated_sharpe_ratio": self.deflated_sharpe(
                sharpe,
                n_trials=self._count_experiments(),
                n_obs=max(len(returns), 1),
                skew=float(stats.skew(returns)) if len(returns) > 2 else 0.0,
                kurt=float(stats.kurtosis(returns)) if len(returns) > 3 else 3.0,
            ),
            "prob_backtest_overfitting": self._prob_backtest_overfitting(returns),
            "brier_decomposition": None,
            "sharpe_by_edge_type": self._sharpe_by_edge_type(trades_df),
            "sharpe_by_time_of_game": {},
        }
        return metrics

    @staticmethod
    def _trades_df(results: BacktestResults) -> pd.DataFrame:
        if not results.trades:
            return pd.DataFrame()
        rows = [
            {
                "game_id": t.game_id,
                "pnl": t.pnl,
                "edge_type": t.edge_type,
                "return_pct": t.pnl / max(t.n_contracts * t.fill_price, 1e-6),
                "timestamp": t.execution_time,
            }
            for t in results.trades
        ]
        return pd.DataFrame(rows)

    def _sharpe_ratio(self, returns: np.ndarray, periods_per_year: float = 252.0) -> float:
        if len(returns) < 2:
            return 0.0
        excess = returns - self.risk_free_rate / periods_per_year
        std = float(np.std(excess, ddof=1))
        if std < 1e-12:
            return 0.0
        return float(np.mean(excess) / std * math.sqrt(periods_per_year))

    def _sortino_ratio(self, returns: np.ndarray, periods_per_year: float = 252.0) -> float:
        if len(returns) < 2:
            return 0.0
        excess = returns - self.risk_free_rate / periods_per_year
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float("inf")
        dd_std = float(np.std(downside, ddof=1))
        if dd_std < 1e-12:
            return 0.0
        return float(np.mean(excess) / dd_std * math.sqrt(periods_per_year))

    def _annualized_return(self, results: BacktestResults) -> float:
        if results.equity_curve.empty:
            return 0.0
        days = (
            results.equity_curve["timestamp"].max() - results.equity_curve["timestamp"].min()
        ).days
        if days <= 0:
            days = 1
        total_ret = results.total_pnl / results.initial_capital
        return float((1 + total_ret) ** (365.0 / days) - 1)

    def _max_drawdown(self, results: BacktestResults) -> float:
        if results.equity_curve.empty:
            return 0.0
        capital = results.equity_curve["capital"].to_numpy()
        peak = np.maximum.accumulate(capital)
        dd = (capital - peak) / peak
        return float(abs(np.min(dd)))

    def _max_drawdown_duration(self, results: BacktestResults) -> float:
        if results.equity_curve.empty:
            return 0.0
        capital = results.equity_curve["capital"].to_numpy()
        peak = np.maximum.accumulate(capital)
        underwater = capital < peak
        max_dur = 0
        cur = 0
        for u in underwater:
            if u:
                cur += 1
                max_dur = max(max_dur, cur)
            else:
                cur = 0
        return float(max_dur)

    @staticmethod
    def _var_95(returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        return float(np.percentile(returns, 5))

    @staticmethod
    def _hit_rate(results: BacktestResults) -> float:
        if not results.trades:
            return 0.0
        wins = sum(1 for t in results.trades if t.pnl > 0)
        return wins / len(results.trades)

    @staticmethod
    def _avg_win(results: BacktestResults) -> float:
        wins = [t.pnl for t in results.trades if t.pnl > 0]
        return float(np.mean(wins)) if wins else 0.0

    @staticmethod
    def _avg_loss(results: BacktestResults) -> float:
        losses = [t.pnl for t in results.trades if t.pnl < 0]
        return float(np.mean(losses)) if losses else 0.0

    @staticmethod
    def _profit_factor(results: BacktestResults) -> float:
        gross_profit = sum(t.pnl for t in results.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in results.trades if t.pnl < 0))
        if gross_loss < 1e-12:
            return float("inf") if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    @staticmethod
    def _p_value_returns(returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 1.0
        _, p_value = stats.ttest_1samp(returns, 0.0)
        return float(p_value)

    def block_bootstrap_sharpe_ci(
        self,
        trades_df: pd.DataFrame,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> tuple[float, float]:
        """
        Block bootstrap CI on Sharpe, resampling games as blocks.

        Trades within the same game are correlated (shared outcome).
        """
        if trades_df.empty or "game_id" not in trades_df.columns:
            return (0.0, 0.0)

        rng = np.random.default_rng(seed)
        games = trades_df["game_id"].unique()
        sharpes: list[float] = []

        for _ in range(n_bootstrap):
            sampled_games = rng.choice(games, size=len(games), replace=True)
            boot = pd.concat(
                [trades_df[trades_df["game_id"] == g] for g in sampled_games],
                ignore_index=True,
            )
            rets = boot["return_pct"].to_numpy()
            sharpes.append(self._sharpe_ratio(rets))

        alpha = (1 - ci) / 2
        return (float(np.percentile(sharpes, 100 * alpha)), float(np.percentile(sharpes, 100 * (1 - alpha))))

    def deflated_sharpe(
        self,
        sharpe: float,
        n_trials: int,
        n_obs: int,
        skew: float = 0.0,
        kurt: float = 3.0,
    ) -> float:
        """
        Deflated Sharpe Ratio per Bailey & Lopez de Prado (2014).

        Adjusts for multiple testing and non-normality of returns.
        """
        if n_obs <= 1 or n_trials <= 0:
            return sharpe

        euler_gamma = 0.5772156649
        # Expected maximum Sharpe under null from n_trials independent tests
        e_max_sharpe = (
            (1 - euler_gamma) * stats.norm.ppf(1 - 1.0 / n_trials)
            + euler_gamma * stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
        )
        # Sharpe variance adjustment for skew/kurtosis
        sharpe_var = (
            1
            - skew * sharpe
            + (kurt - 1) / 4 * sharpe**2
        ) / max(n_obs - 1, 1)

        if sharpe_var <= 0:
            return 0.0

        z = (sharpe - e_max_sharpe) / math.sqrt(sharpe_var)
        return float(stats.norm.cdf(z))

    def _count_experiments(self) -> int:
        if not self.experiment_log_path.exists():
            return 1
        lines = [
            line for line in self.experiment_log_path.read_text().splitlines() if line.strip()
        ]
        return max(1, len(lines))

    @staticmethod
    def _prob_backtest_overfitting(returns: np.ndarray, n_partitions: int = 4) -> float:
        """
        Simplified PBO proxy: fraction of partitions with negative Sharpe.

        Full CSCV requires combinatorial symmetric CV; this is a lightweight proxy.
        """
        if len(returns) < n_partitions * 2:
            return 0.5
        parts = np.array_split(returns, n_partitions)
        negative = sum(1 for p in parts if len(p) > 1 and np.mean(p) < 0)
        return negative / n_partitions

    def _sharpe_by_edge_type(self, trades_df: pd.DataFrame) -> dict[str, float]:
        if trades_df.empty:
            return {}
        result: dict[str, float] = {}
        for edge_type, group in trades_df.groupby("edge_type"):
            result[str(edge_type)] = self._sharpe_ratio(group["return_pct"].to_numpy())
        return result

    def diebold_mariano(
        self,
        model_errors: np.ndarray,
        benchmark_errors: np.ndarray,
        h: int = 1,
    ) -> dict[str, float]:
        """
        Diebold-Mariano test: is model forecast significantly better than benchmark?

        Uses squared errors (MSE loss). Negative DM stat favors model.
        """
        d = model_errors**2 - benchmark_errors**2
        n = len(d)
        if n < 2:
            return {"dm_statistic": 0.0, "p_value": 1.0}

        d_mean = float(np.mean(d))
        # Newey-West variance with h-1 lags
        gamma0 = float(np.var(d, ddof=1))
        nw_var = gamma0
        for lag in range(1, h):
            if lag >= n:
                break
            cov = float(np.cov(d[lag:], d[:-lag], ddof=1)[0, 1])
            nw_var += 2 * (1 - lag / h) * cov

        if nw_var <= 0:
            return {"dm_statistic": 0.0, "p_value": 1.0}

        dm_stat = d_mean / math.sqrt(nw_var / n)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        return {"dm_statistic": float(dm_stat), "p_value": float(p_value)}

    def diebold_mariano_vs_market(
        self,
        model_probs: np.ndarray,
        market_probs: np.ndarray,
        outcomes: np.ndarray,
    ) -> dict[str, float]:
        """DM test comparing model Brier errors to market-as-forecast."""
        model_errors = model_probs - outcomes
        market_errors = market_probs - outcomes
        return self.diebold_mariano(model_errors, market_errors)

    def brier_decomposition(
        self,
        model_probs: np.ndarray,
        outcomes: np.ndarray,
    ) -> dict[str, float]:
        """Murphy decomposition for model forecasts."""
        return CalibrationAnalyzer().brier_decomposition(model_probs, outcomes)

    def save_plots(self, results: BacktestResults, output_dir: Path | str = "outputs/backtest") -> None:
        """Save equity curve, drawdown, and PnL distribution plots."""
        import matplotlib.pyplot as plt

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if results.equity_curve.empty:
            return

        # Equity curve
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(results.equity_curve["timestamp"], results.equity_curve["capital"])
        ax.set_title("Equity Curve")
        ax.set_ylabel("Capital ($)")
        fig.savefig(out / "equity_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Drawdown
        capital = results.equity_curve["capital"].to_numpy()
        peak = np.maximum.accumulate(capital)
        dd = (capital - peak) / peak
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.fill_between(range(len(dd)), dd, 0, alpha=0.5, color="red")
        ax.set_title("Drawdown")
        fig.savefig(out / "drawdown.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # PnL histogram
        pnls = [t.pnl for t in results.trades]
        if pnls:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(pnls, bins=20, edgecolor="black")
            ax.set_title("Trade PnL Distribution")
            ax.set_xlabel("PnL ($)")
            fig.savefig(out / "trade_pnl_hist.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        logger.info("Backtest plots saved to %s", out)
