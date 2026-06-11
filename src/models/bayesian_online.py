"""
Bayesian online state-space model for in-play win probability.

State: latent relative team strength theta_t (+ favors home scoring rate)
Transition: theta_t ~ N(theta_{t-1}, process_noise^2 * dt)
Observation: signed scoring events inform strength via Poisson-rate likelihood
Inference: Extended Kalman Filter (default) or Sequential Monte Carlo particle filter
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.base import WinProbabilityModel
from src.utils.pandas_typing import as_float, as_int

logger = logging.getLogger(__name__)

SPORT_TOTAL_SECONDS = {"NFL": 3600, "NBA": 2880}


@dataclass
class PosteriorState:
    """Current belief over latent strength."""

    mean: float
    variance: float
    score_diff: int = 0
    seconds_remaining: float = 3600.0
    time_elapsed: float = 0.0


@dataclass
class ScoringEvent:
    """Normalized scoring event for online updates."""

    team: str  # 'home' or 'away'
    points: int
    time_elapsed: float
    score_diff: int
    seconds_remaining: float


class BayesianOnlineWinProb(WinProbabilityModel):
    """
    State-space model for in-play win probability with online Bayesian updates.

    Example:
        model = BayesianOnlineWinProb(inference='ekf')
        model.fit_hyperparams(historical_states)
        prob = model.update({'team': 'home', 'points': 7}, time_elapsed=1800)
        win_p = model.win_probability(seconds_remaining=1800)
    """

    model_name = "bayesian_online"
    model_version = "1.0.0"

    def __init__(
        self,
        prior_mean: float = 0.0,
        prior_var: float = 1.0,
        process_noise: float = 0.01,
        observation_noise: float = 3.0,
        base_rate: float = 0.05,
        n_particles: int = 1000,
        inference: str = "ekf",
        mc_samples: int = 2000,
        seed: int = 42,
    ) -> None:
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.base_rate = base_rate
        self.n_particles = n_particles
        self.inference = inference
        self.mc_samples = mc_samples
        self._rng = np.random.default_rng(seed)

        self._posterior = PosteriorState(mean=prior_mean, variance=prior_var)
        self._particles: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._default_params = {
            "process_noise": process_noise,
            "observation_noise": observation_noise,
            "base_rate": base_rate,
        }
        self._fitted_params = dict(self._default_params)
        self._is_fitted = False

    def reset(self, prior_mean: float | None = None) -> None:
        """Reset filter state for a new game."""
        mu = self.prior_mean if prior_mean is None else prior_mean
        self._posterior = PosteriorState(mean=mu, variance=self.prior_var)
        self._particles = None
        self._weights = None
        if self.inference == "particle":
            self._init_particles(mu)

    def _init_particles(self, mean: float) -> None:
        self._particles = self._rng.normal(
            mean, np.sqrt(self.prior_var), size=self.n_particles
        )
        self._weights = np.ones(self.n_particles) / self.n_particles

    def _total_seconds(self, sport: str = "NFL") -> int:
        return SPORT_TOTAL_SECONDS.get(sport.upper(), 3600)

    def _expected_rate(self, theta: float) -> float:
        """Home minus away scoring rate differential."""
        return float(self.base_rate * (np.exp(theta / 2.0) - np.exp(-theta / 2.0)))

    def _h(self, theta: float) -> float:
        """Observation model: expected signed points per unit strength."""
        return self._expected_rate(theta)

    def _H(self, theta: float) -> float:
        """Analytic Jacobian dh/dtheta (no numerical differentiation)."""
        return float(0.5 * self.base_rate * (np.exp(theta / 2.0) + np.exp(-theta / 2.0)))

    def predict_step(self, dt: float) -> None:
        """Time update: strength drifts with process noise over dt seconds."""
        if dt <= 0:
            return
        q = (self.process_noise**2) * dt
        self._posterior.variance += q

        if self.inference == "particle" and self._particles is not None:
            self._particles += self._rng.normal(0, np.sqrt(q), size=self.n_particles)

    def _ekf_update(self, z: float, dt: float) -> None:
        """EKF measurement update for signed scoring observation."""
        mu = self._posterior.mean
        p = self._posterior.variance
        h = self._h(mu) * dt
        H = self._H(mu) * dt
        r = self.observation_noise**2
        innovation = z - h
        s = H * p * H + r
        kalman_gain = (p * H) / s if s > 1e-12 else 0.0
        self._posterior.mean = mu + kalman_gain * innovation
        self._posterior.variance = max(1e-9, (1 - kalman_gain * H) * p)

    def _particle_update(self, z: float, dt: float) -> None:
        """Particle filter measurement update."""
        if self._particles is None or self._weights is None:
            self._init_particles(self._posterior.mean)

        assert self._particles is not None and self._weights is not None
        r = self.observation_noise**2
        predicted = np.array([self._h(th) * dt for th in self._particles])
        log_lik = -0.5 * ((z - predicted) ** 2) / r
        log_w = np.log(self._weights + 1e-300) + log_lik
        log_w -= np.max(log_w)
        self._weights = np.exp(log_w)
        self._weights /= self._weights.sum()

        ess = 1.0 / np.sum(self._weights**2)
        if ess < self.n_particles / 2:
            idx = self._rng.choice(self.n_particles, size=self.n_particles, p=self._weights)
            self._particles = self._particles[idx]
            self._weights = np.ones(self.n_particles) / self.n_particles

        self._posterior.mean = float(np.average(self._particles, weights=self._weights))
        self._posterior.variance = float(
            np.average((self._particles - self._posterior.mean) ** 2, weights=self._weights)
        )

    def update(self, scoring_event: dict[str, object], time_elapsed: float) -> float:
        """
        Online update from a scoring event.

        scoring_event keys: team ('home'|'away'), points (int)
        Returns updated home win probability.
        """
        team = str(scoring_event["team"])
        points = as_int(scoring_event["points"])
        dt = max(time_elapsed - self._posterior.time_elapsed, 1e-6)

        self.predict_step(dt)
        z = float(points if team == "home" else -points)

        if self.inference == "ekf":
            self._ekf_update(z, dt)
        else:
            self._particle_update(z, dt)

        if team == "home":
            self._posterior.score_diff += points
        else:
            self._posterior.score_diff -= points

        self._posterior.time_elapsed = time_elapsed
        return self.win_probability(self._posterior.seconds_remaining)

    def win_probability(
        self,
        seconds_remaining: float,
        score_diff: int | None = None,
        sport: str = "NFL",
    ) -> float:
        """
        Compute home win probability from posterior over theta.

        Uses Monte Carlo integration over remaining game time.
        """
        self._posterior.seconds_remaining = seconds_remaining
        s = self._posterior.score_diff if score_diff is None else score_diff
        total = self._total_seconds(sport)
        t_frac = seconds_remaining / total

        mu = self._posterior.mean
        var = self._posterior.variance
        thetas = self._rng.normal(mu, np.sqrt(max(var, 1e-9)), size=self.mc_samples)

        wins = 0.0
        for theta in thetas:
            lam_h = self.base_rate * np.exp(theta / 2.0) * t_frac
            lam_a = self.base_rate * np.exp(-theta / 2.0) * t_frac
            add_h = self._rng.poisson(lam_h * total)
            add_a = self._rng.poisson(lam_a * total)
            if s + add_h > add_a:
                wins += 1
            elif s + add_h == add_a:
                wins += 0.5

        return float(wins / self.mc_samples)

    @staticmethod
    def extract_scoring_events(game_df: pd.DataFrame, sport: str = "NFL") -> list[ScoringEvent]:
        """Convert game state rows to ordered scoring events."""
        total = SPORT_TOTAL_SECONDS.get(sport.upper(), 3600)
        scoring = game_df[game_df["is_scoring_event"]].sort_values(
            "seconds_remaining", ascending=False
        )
        events: list[ScoringEvent] = []
        prev_diff = 0
        prev_home, prev_away = 0, 0

        for row in scoring.itertuples(index=False):
            diff = as_int(row.score_differential)
            home = as_int(row.home_score)
            away = as_int(row.away_score)
            delta_diff = diff - prev_diff
            delta_home = home - prev_home
            delta_away = away - prev_away

            if delta_home > 0:
                team, points = "home", delta_home
            elif delta_away > 0:
                team, points = "away", delta_away
            elif delta_diff > 0:
                team, points = "home", abs(delta_diff)
            elif delta_diff < 0:
                team, points = "away", abs(delta_diff)
            else:
                prev_diff, prev_home, prev_away = diff, home, away
                continue

            secs = as_int(row.seconds_remaining)
            events.append(
                ScoringEvent(
                    team=team,
                    points=points,
                    time_elapsed=total - secs,
                    score_diff=diff,
                    seconds_remaining=secs,
                )
            )
            prev_diff, prev_home, prev_away = diff, home, away
        return events

    def replay_game(self, game_df: pd.DataFrame) -> list[float]:
        """Replay all scoring events; return win prob after each update."""
        sport = str(game_df["sport"].iloc[0]) if "sport" in game_df.columns else "NFL"
        self.reset()
        probs: list[float] = []
        for event in self.extract_scoring_events(game_df, sport=sport):
            self._posterior.score_diff = event.score_diff - (
                event.points if event.team == "home" else -event.points
            )
            p = self.update(
                {"team": event.team, "points": event.points},
                time_elapsed=event.time_elapsed,
            )
            self._posterior.seconds_remaining = event.seconds_remaining
            probs.append(p)
        return probs

    def _sequence_nll(
        self,
        events: list[ScoringEvent],
        process_noise: float,
        observation_noise: float,
        base_rate: float,
    ) -> float:
        """Negative log-likelihood for a scoring event sequence."""
        old = (self.process_noise, self.observation_noise, self.base_rate)
        self.process_noise, self.observation_noise, self.base_rate = (
            process_noise,
            observation_noise,
            base_rate,
        )
        self.reset()
        nll = 0.0
        r = observation_noise**2

        for event in events:
            dt = max(event.time_elapsed - self._posterior.time_elapsed, 1e-6)
            self.predict_step(dt)
            z = float(event.points if event.team == "home" else -event.points)
            mu = self._posterior.mean
            pred = self._h(mu) * dt
            nll += 0.5 * ((z - pred) ** 2) / r + np.log(2 * np.pi * r)
            self._ekf_update(z, dt)
            self._posterior.time_elapsed = event.time_elapsed
            if event.team == "home":
                self._posterior.score_diff += event.points
            else:
                self._posterior.score_diff -= event.points

        self.process_noise, self.observation_noise, self.base_rate = old
        return float(nll)

    def fit_hyperparams(self, historical_games: pd.DataFrame) -> dict[str, float]:
        """
        Fit process_noise, observation_noise, and base_rate via MLE.

        Uses EM-style iterative refinement: outer loop optimizes hyperparameters
        on pooled scoring sequences from 2018-2021 training seasons.
        """
        train = historical_games[historical_games["season"].isin([2018, 2019, 2020, 2021])]
        if train.empty:
            train = historical_games

        sequences: list[list[ScoringEvent]] = []
        for _, game_df in train.groupby("game_id"):
            sport = str(game_df["sport"].iloc[0]) if "sport" in game_df.columns else "NFL"
            seq = self.extract_scoring_events(game_df, sport=sport)
            if seq:
                sequences.append(seq)

        if not sequences:
            logger.warning("No scoring sequences for hyperparameter fitting")
            return self._default_params

        def objective(x: np.ndarray) -> float:
            pn, on, br = float(x[0]), float(x[1]), float(x[2])
            if pn <= 0 or on <= 0 or br <= 0:
                return 1e12
            return sum(self._sequence_nll(seq, pn, on, br) for seq in sequences)

        x0 = np.array([self.process_noise, self.observation_noise, self.base_rate])
        bounds = [(1e-4, 1.0), (0.1, 20.0), (1e-4, 1.0)]
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

        if result.success:
            self.process_noise = float(result.x[0])
            self.observation_noise = float(result.x[1])
            self.base_rate = float(result.x[2])
            self._fitted_params = {
                "process_noise": self.process_noise,
                "observation_noise": self.observation_noise,
                "base_rate": self.base_rate,
            }
            self._is_fitted = True
            logger.info("Bayesian hyperparams fit: %s", self._fitted_params)
        else:
            logger.warning("Hyperparameter optimization did not converge")

        return self._fitted_params

    def fit(self, game_states: pd.DataFrame, outcomes: pd.Series) -> BayesianOnlineWinProb:
        """Fit hyperparameters on training seasons."""
        self.fit_hyperparams(game_states)
        return self

    def predict(self, game_states: pd.DataFrame) -> np.ndarray:
        """Predict win probability at each game state row."""
        probs = np.empty(len(game_states))
        idx = 0
        for _, game_df in game_states.groupby("game_id", sort=False):
            sport = str(game_df["sport"].iloc[0]) if "sport" in game_df.columns else "NFL"
            game_df = game_df.sort_values("seconds_remaining", ascending=False)
            self.reset()

            events = self.extract_scoring_events(game_df, sport=sport)
            event_idx = 0

            for row in game_df.itertuples(index=False):
                secs = as_float(row.seconds_remaining)
                diff = as_int(row.score_differential)

                while event_idx < len(events) and events[event_idx].seconds_remaining >= secs:
                    ev = events[event_idx]
                    self._posterior.score_diff = ev.score_diff - (
                        ev.points if ev.team == "home" else -ev.points
                    )
                    self.update(
                        {"team": ev.team, "points": ev.points},
                        time_elapsed=ev.time_elapsed,
                    )
                    event_idx += 1

                self._posterior.score_diff = diff
                self._posterior.seconds_remaining = secs
                probs[idx] = self.win_probability(secs, score_diff=diff, sport=sport)
                idx += 1

        self.validate_predictions(probs)
        return probs

    def evaluate_calibration_error(
        self,
        probs: np.ndarray | None = None,
        outcomes: np.ndarray | None = None,
        n_bins: int = 10,
        use_fitted_params: bool = True,
        game_states: pd.DataFrame | None = None,
        test_outcomes: pd.Series | None = None,
    ) -> float:
        """ECE with optional comparison of fitted vs default hyperparameters."""
        if game_states is not None and test_outcomes is not None:
            if not use_fitted_params:
                self.process_noise = self._default_params["process_noise"]
                self.observation_noise = self._default_params["observation_noise"]
                self.base_rate = self._default_params["base_rate"]
            else:
                self.process_noise = self._fitted_params["process_noise"]
                self.observation_noise = self._fitted_params["observation_noise"]
                self.base_rate = self._fitted_params["base_rate"]
            probs = self.predict(game_states)
            outcomes = test_outcomes.to_numpy()

        if probs is None or outcomes is None:
            raise ValueError("Provide (probs, outcomes) or (game_states, test_outcomes)")
        return WinProbabilityModel.calibration_error(probs, outcomes, n_bins=n_bins)

    def posterior_impact(self, event: dict[str, object], time_elapsed: float) -> float:
        """Measure how much a scoring event shifts the posterior mean."""
        mu_before = self._posterior.mean
        self.update(event, time_elapsed)
        return self._posterior.mean - mu_before
