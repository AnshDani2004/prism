"""Calibration analysis: ECE, reliability diagrams, Brier decomposition."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.models.base import WinProbabilityModel

logger = logging.getLogger(__name__)


class CalibrationAnalyzer:
    """Rigorous calibration diagnostics and model comparison."""

    def __init__(self, output_dir: Path | str = "outputs/calibration") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error via equal-frequency binning."""
        from src.models.base import WinProbabilityModel

        return WinProbabilityModel.calibration_error(probs, outcomes, n_bins=n_bins)

    def brier_decomposition(
        self, probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
    ) -> dict[str, float]:
        """
        Murphy (1973) Brier score decomposition.

        BS = reliability - resolution + uncertainty
        """
        probs = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        o_bar = float(np.mean(outcomes))
        uncertainty = o_bar * (1 - o_bar)

        order = np.argsort(probs)
        probs_sorted = probs[order]
        outcomes_sorted = outcomes[order]
        bins = np.array_split(np.arange(len(probs)), n_bins)

        reliability = 0.0
        resolution = 0.0
        n = len(probs)
        for indices in bins:
            if len(indices) == 0:
                continue
            f_k = float(np.mean(probs_sorted[indices]))
            o_k = float(np.mean(outcomes_sorted[indices]))
            n_k = len(indices)
            reliability += (n_k / n) * (f_k - o_k) ** 2
            resolution += (n_k / n) * (o_k - o_bar) ** 2

        brier = float(np.mean((probs - outcomes) ** 2))
        return {
            "brier": brier,
            "reliability": reliability,
            "resolution": resolution,
            "uncertainty": uncertainty,
        }

    def compare_models(
        self,
        models: dict[str, WinProbabilityModel],
        game_states: pd.DataFrame,
        outcomes: np.ndarray,
    ) -> pd.DataFrame:
        """Comparison table: ECE, Brier, log loss, AUC per model."""
        from sklearn.metrics import log_loss, roc_auc_score

        rows: list[dict[str, float | str]] = []
        for name, model in models.items():
            probs = model.predict(game_states)
            probs_clipped = np.clip(probs, 1e-6, 1 - 1e-6)
            rows.append(
                {
                    "model": name,
                    "ece": self.ece(probs, outcomes),
                    "brier": float(np.mean((probs - outcomes) ** 2)),
                    "log_loss": float(log_loss(outcomes, probs_clipped)),
                    "auc": float(roc_auc_score(outcomes, probs))
                    if len(np.unique(outcomes)) > 1
                    else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    def plot_reliability_diagram(
        self,
        probs: np.ndarray,
        outcomes: np.ndarray,
        model_name: str = "model",
        n_bins: int = 10,
        save: bool = True,
    ) -> Path | None:
        """Reliability diagram: predicted probability vs observed win rate."""
        order = np.argsort(probs)
        probs_sorted = probs[order]
        outcomes_sorted = outcomes[order]
        bins = np.array_split(np.arange(len(probs)), n_bins)

        mean_preds: list[float] = []
        mean_obs: list[float] = []
        for indices in bins:
            if len(indices) == 0:
                continue
            mean_preds.append(float(np.mean(probs_sorted[indices])))
            mean_obs.append(float(np.mean(outcomes_sorted[indices])))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.scatter(mean_preds, mean_obs, s=80, zorder=3)
        ax.plot(mean_preds, mean_obs, alpha=0.7)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed win rate")
        ax.set_title(f"Reliability Diagram — {model_name}")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        path = None
        if save:
            path = self.output_dir / f"reliability_{model_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved reliability diagram to %s", path)
        plt.close(fig)
        return path

    def plot_ece_by_time(
        self,
        game_states: pd.DataFrame,
        probs: np.ndarray,
        outcomes: np.ndarray,
        model_name: str = "model",
        n_time_bins: int = 10,
        save: bool = True,
    ) -> Path | None:
        """ECE as a function of seconds remaining (binned)."""
        df = game_states.copy()
        df["prob"] = probs
        df["outcome"] = outcomes
        df["time_bin"] = pd.qcut(df["seconds_remaining"], n_time_bins, duplicates="drop")

        eces: list[float] = []
        labels: list[str] = []
        for label, group in df.groupby("time_bin", observed=True):
            eces.append(self.ece(group["prob"].to_numpy(), group["outcome"].to_numpy()))
            labels.append(str(label))

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=labels, y=eces, ax=ax, color="steelblue")
        ax.set_xlabel("Seconds remaining (bin)")
        ax.set_ylabel("ECE")
        ax.set_title(f"Calibration Error by Game Clock — {model_name}")
        plt.xticks(rotation=45, ha="right")

        path = None
        if save:
            path = self.output_dir / f"ece_by_time_{model_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_sharpness(
        self,
        probs: np.ndarray,
        model_name: str = "model",
        save: bool = True,
    ) -> Path | None:
        """Histogram of predicted probabilities (sharpness)."""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(probs, bins=20, range=(0, 1), edgecolor="black", alpha=0.7)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Count")
        ax.set_title(f"Sharpness — {model_name}")

        path = None
        if save:
            path = self.output_dir / f"sharpness_{model_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
