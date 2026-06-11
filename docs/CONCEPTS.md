# PRISM Technical Concept Compendium

Interview-prep reference for every nontrivial concept used in PRISM.

---

## DuckDB Columnar Analytics

**Used in:** `src/data/database.py` — local analytical storage for game states, contracts, and prices

**The idea:** DuckDB is an embedded columnar SQL database optimized for analytical queries (aggregations, joins over large tables) without a server process. Data is stored column-wise, so queries that touch few columns scan less I/O.

**Why it matters in PRISM:** The research workflow is batch-oriented: ingest once, query many times for modeling and backtesting. DuckDB handles ~10M rows on a laptop with sub-second joins between `game_states` and `market_prices`.

**Pitfalls:** DuckDB is not designed for concurrent writers or low-latency serving. Using it as a production trading database would be wrong.

**Study resources:**
- DuckDB documentation: https://duckdb.org/docs/
- Mark Raasveldt, "DuckDB: an embeddable analytical database"

---

## Play-by-Play Game State Extraction

**Used in:** `src/data/nfl.py`, `src/data/nba.py` — reduce raw PBP to modeling-relevant snapshots

**The idea:** Raw play-by-play has ~150–250 events per game. Win probability models need state at scoring events and periodic clock snapshots, not every snap. NFL extracts quarter-starts + scoring plays (~20–30 states/game). NBA additionally samples every 2 minutes due to faster pace.

**Why it matters in PRISM:** State reduction cuts compute by 10× without losing information relevant to in-play probability jumps. The `seconds_remaining` index becomes the join key to market prices.

**Pitfalls:** Over-sampling adds noise; under-sampling misses rapid score swings. NBA's 2-minute floor is a tunable hyperparameter.

**Study resources:**
- nfl_data_py documentation
- nba_api PlayByPlay endpoint docs

---

## Prediction Market Microstructure (Bid-Ask Spread)

**Used in:** `src/data/kalshi.py`, `src/data/polymarket.py` — price storage semantics

**The idea:** A quoted price in a prediction market is not a single number — it is a bid-ask pair. The mid-price \((bid + ask) / 2\) is a fair estimate of implied probability for *analysis*, but you cannot trade at the mid. Buys fill at the ask; sells fill at the bid. The half-spread is an implicit transaction cost.

**Why it matters in PRISM:** Edge = model_prob − market_mid overstates profitability. Phase 5 backtests must fill at the touch.

**Pitfalls:** Using last-trade price instead of bid/ask when the book is wide. Ignoring that spread widens during volatile in-play moments.

**Study resources:**
- Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
- Kalshi fee schedule documentation

---

## Fuzzy String Matching for Entity Resolution

**Used in:** `src/data/mapping.py` — link market contracts to game_ids

**The idea:** Given inconsistent entity names ("Chiefs", "Kansas City", "KC"), resolve to a canonical identifier using alias dictionaries and string similarity (RapidFuzz ratios). Combine with date proximity for disambiguation.

**Why it matters in PRISM:** Without reliable game↔contract links, edge analysis compares model probabilities to prices from the wrong game — a silent catastrophic error.

**Pitfalls:** High-confidence false matches (same team names, different weeks). Always enforce one-contract-per-game-per-source and log ambiguous band.

**Study resources:**
- RapidFuzz documentation
- Record linkage literature (Christen, 2012)

---

## Expected Calibration Error (ECE)

**Used in:** `src/models/base.py` (Phase 2) — model validation

**The idea:** Partition predictions into bins (equal-frequency). ECE = \(\sum_b \frac{n_b}{N} |acc(b) - conf(b)|\), where \(acc(b)\) is the empirical win rate in bin \(b\) and \(conf(b)\) is the mean predicted probability. ECE = 0 means perfect calibration.

**Why it matters in PRISM:** A model with high AUC but ECE = 0.15 is useless for trading — it ranks well but its probability numbers are wrong. Prediction market edge requires calibrated probabilities.

**Pitfalls:** Too few bins → noisy ECE. Too many bins → empty bins. Use equal-frequency binning and report ECE by game-clock segment.

**Study resources:**
- Niculescu-Mizil & Caruana (2005), "Predicting Good Probabilities with Supervised Learning"
- Guo et al. (2017), "On Calibration of Modern Neural Networks"

---

## Vig Removal (Sportsbook Implied Probabilities)

**Used in:** `src/market/interface.py` (Phase 4) — when comparing to sportsbook odds

**The idea:** Raw implied probabilities from odds sum to > 1 (the overround/vig). Normalize: \(p_i = q_i / \sum_j q_j\) where \(q_i = 1/\text{odds}_i\). Alternative: Shin's method for favorite-longshot bias.

**Why it matters in PRISM:** Comparing a model to vigged probabilities manufactures fake edge equal to roughly half the overround.

**Pitfalls:** Proportional normalization assumes equal vig on both sides; Shin's method is better for asymmetric markets.

**Study resources:**
- Shin (1991), "How to read the odds"
- Clarke (2016), "Adjusting bookmaker odds to allow for overround"

---

## Bradley-Terry Model

**Used in:** `src/models/bradley_terry.py` — pre-game baseline win probability

**The idea:** Each team \(i\) has latent strength \(\lambda_i\). P(team A beats B) = \(\lambda_A / (\lambda_A + \lambda_B)\). With home advantage \(h\): P(home wins) = \(\exp(s_h + h) / (\exp(s_h + h) + \exp(s_a))\). Parameters fit by maximum likelihood.

**Why it matters in PRISM:** Simplest defensible baseline. Any fancier model must beat this on calibration, not just accuracy.

**Pitfalls:** No margin information. Treats all games as independent. Cold-start teams need a league-average prior.

**Study resources:**
- Bradley & Terry (1952), Rank analysis of incomplete block designs
- Agresti, *Categorical Data Analysis*, Ch. 9

---

## Elo with Margin-of-Victory Adjustment

**Used in:** `src/models/margin_model.py` — team ratings and pre-game probability

**The idea:** Elo updates: \(R' = R + K \cdot M(m, p) \cdot (S - p)\) where \(S\) is actual result, \(p\) is pregame win prob, and \(M\) is a margin multiplier with autocorrelation correction so 30-point favorite blowouts don't over-update.

**Why it matters in PRISM:** Captures team strength dynamics and score margin — appropriate for NFL/NBA unlike soccer-specific models.

**Pitfalls:** MOV multipliers can overfit. Season-boundary ratings need shrinkage toward mean.

**Study resources:**
- Elo (1978), *The Rating of Chessplayers*
- FiveThirtyEight NFL Elo methodology post

---

## Skellam Distribution

**Used in:** `src/models/margin_model.py` — NFL score differential PMF

**The idea:** If home and away scoring are independent Poissons with rates \(\mu_1, \mu_2\), the difference \(D = X_1 - X_2\) follows a Skellam distribution: \(P(D=k) = e^{-(\mu_1+\mu_2)}(\mu_1/\mu_2)^{k/2} I_{|k|}(2\sqrt{\mu_1\mu_2})\).

**Why it matters in PRISM:** Natural model for discrete scoring-event sports. Win probability = \(P(D > 0)\).

**Pitfalls:** Assumes independent Poisson scoring; ignores correlation structure and key numbers (3, 7 in NFL).

**Study resources:**
- Karlis & Ntzoufras (2003), "Analysis of sports data by using bivariate Poisson models"
- `scipy.stats.skellam`

---

## Isotonic Regression

**Used in:** `src/models/inplay_xgb.py` — post-hoc probability calibration

**The idea:** Fit a monotonically non-decreasing piecewise-constant function \(\hat{f}\) such that \(\hat{f}(p_i) \approx y_i\), minimizing \(\sum (y_i - \hat{f}(p_i))^2\) subject to monotonicity.

**Why it matters in PRISM:** XGBoost probabilities are poorly calibrated; isotonic regression fixes the scale without destroying ranking.

**Pitfalls:** Needs enough validation samples. Can overfit with very few bins. Must never fit on test data.

**Study resources:**
- scikit-learn `IsotonicRegression`
- Zadrozny & Elkan (2002), "Transforming classifier scores into accurate multiclass probability estimates"

---

## Brier Score and Murphy Decomposition

**Used in:** `src/models/calibration.py` — model comparison

**The idea:** Brier score BS = \(\frac{1}{N}\sum(f_i - o_i)^2\). Murphy (1973) decomposes: BS = Reliability − Resolution + Uncertainty. Reliability measures calibration; resolution measures how much forecasts vary from base rate.

**Why it matters in PRISM:** A model can have low Brier via high resolution but poor reliability — exactly the failure mode for trading.

**Pitfalls:** Brier score alone doesn't separate calibration from sharpness. Always report ECE alongside.

**Study resources:**
- Murphy (1973), "A new vector partition of the probability score"
- Wilks, *Statistical Methods in the Atmospheric Sciences*, Ch. 8

---

## State-Space Models

**Used in:** `src/models/bayesian_online.py` — in-play win probability as a stochastic process

**The idea:** Hidden state \(\theta_t\) evolves over time (transition model); observations (scoring events) are noisy measurements of \(\theta_t\). Win probability is a functional of the posterior \(p(\theta_t \mid \text{data})\) integrated over remaining game time.

**Why it matters in PRISM:** Unlike XGBoost, this treats a game as a continuous process where beliefs update sequentially. Each scoring event shifts the posterior — the "which events matter most" question is answerable via posterior impact analysis.

**Pitfalls:** State-space spec is always wrong in practice. Linear-Gaussian assumptions fail for discrete scoring. Must validate EKF against particle filter.

**Study resources:**
- Särkkä & Svensson (2023), *Bayesian Filtering and Smoothing*
- Simon (2006), *Optimal State Estimation*

---

## Extended Kalman Filter (EKF)

**Used in:** `src/models/bayesian_online.py` — default online inference

**The idea:** For nonlinear models \(x_{t+1} = f(x_t) + w\), \(z_t = h(x_t) + v\), linearize via Jacobians \(F = \partial f/\partial x\), \(H = \partial h/\partial x\) and apply Kalman update equations on the linearized system.

**Why it matters in PRISM:** Fast online updates at each scoring event. Analytic Jacobian of the Poisson-rate observation model avoids numerical differentiation.

**Pitfalls:** Linearization error when posterior is highly non-Gaussian. Can diverge if \(H P H^T + R\) is near-singular.

**Study resources:**
- Anderson & Moore (1979), *Optimal Filtering*, Ch. 10
- Särkkä (2013), *Bayesian Filtering and Smoothing*, Ch. 9

---

## Particle Filter / Sequential Monte Carlo

**Used in:** `src/models/bayesian_online.py` — alternative inference (`inference='particle'`)

**The idea:** Represent posterior as weighted particles \(\{(\theta^{(i)}, w^{(i)})\}_{i=1}^N\). Predict: propagate particles through transition. Update: reweight by observation likelihood. Resample when effective sample size drops.

**Why it matters in PRISM:** Non-parametric posterior; no linearization error. Serves as ground-truth check on EKF approximations. Same algorithm family as bayesian-market-filters.

**Pitfalls:** Particle degeneracy without resampling. Needs enough particles for high-dimensional states (not an issue for scalar \(\theta\)).

**Study resources:**
- Doucet & Johansen (2009), "A Tutorial on Particle Filtering"
- Gordon, Salmond & Smith (1993), "Novel approach to nonlinear/non-Gaussian Bayesian state estimation"

---

## EM Algorithm for Hyperparameter Estimation

**Used in:** `src/models/bayesian_online.py` — `fit_hyperparams()` via MLE on scoring sequences

**The idea:** Expectation-Maximization alternates: (E) compute expected log-likelihood of complete data given current parameters; (M) maximize to update parameters. Here approximated by direct minimization of negative log-likelihood over process_noise, observation_noise, and base_rate.

**Why it matters in PRISM:** Default hyperparameters are guesses. Fitted values on 2018–2021 training data must beat defaults on 2022 validation or the model adds no value.

**Pitfalls:** Local optima. EM assumes correct model structure — if the state-space is misspecified, fitted params just overfit noise.

**Study resources:**
- Dempster, Laird & Rubin (1977), "Maximum Likelihood from Incomplete Data via the EM Algorithm"
- Bishop, *Pattern Recognition and Machine Learning*, Ch. 9.3

---

## Bid-Ask Spread and Microstructure Noise

**Used in:** `src/market/interface.py`, `src/data/kalshi.py` — price semantics

**The idea:** Quoted prices are pairs \((b, a)\). Mid \(m = (b+a)/2\) estimates fair value for analysis; executable prices are \(a\) (buy) and \(b\) (sell). Half-spread \((a-b)/2\) is an implicit transaction cost.

**Why it matters in PRISM:** Edge is computed against mid; Phase 5 backtests fill at the touch. Ignoring spread inflates Sharpe.

**Pitfalls:** Using last trade in illiquid markets. Assuming constant spread in-play (spreads widen during volatility).

**Study resources:**
- Roll (1984), "A Simple Implicit Measure of the Effective Bid-Ask Spread"
- Avellaneda & Stoikov (2008)

---

## Adverse Selection

**Used in:** `src/market/adverse_selection.py` — edge realizability

**The idea:** Informed traders pick off stale quotes. If your model disagrees with the market, you may be the "dumb money" unless the market is genuinely slow to update. Adverse selection tests whether post-event price moves validate your signal direction.

**Why it matters in PRISM:** Stale prices are only alpha if they haven't already been arbitraged. Correction latency measures the exploitable window.

**Pitfalls:** Confusing causation (your signal triggered the move). Not modeling entry delay.

**Study resources:**
- Glosten & Milgrom (1985), "Bid, ask and transaction prices in a specialist market"
- Easley, O'Hara & Paperman (1996), "Liquidity, information, and infrequently traded stocks"

---

## Bootstrap Confidence Intervals

**Used in:** Phase 5 `src/backtest/metrics.py` (planned) — Sharpe significance

**The idea:** Resample data with replacement many times; compute statistic on each replicate; take percentiles for CI. Block bootstrap resamples correlated units (games) intact to preserve dependence.

**Why it matters in PRISM:** Trades within a game share outcome — iid bootstrap overstates significance.

**Pitfalls:** Too few blocks. Assuming stationarity across seasons.

**Study resources:**
- Efron & Tibshirani (1993), *An Introduction to the Bootstrap*
- Politis & Romano (1994), "The stationary bootstrap"

---

## Kelly Criterion

**Used in:** `src/backtest/sizing.py` — position sizing

**The idea:** Kelly optimal fraction \(f^* = (bp - q) / b\) where \(b\) is net odds, \(p\) is win probability, \(q = 1-p\). Maximizes long-run geometric growth rate.

**Why it matters in PRISM:** Converts edge magnitude into capital allocation. Fractional Kelly (0.25×) reduces drawdown variance at the cost of growth.

**Pitfalls:** Full Kelly assumes known probabilities and iid bets. Correlated bets (same-game trades) violate assumptions — use fractional Kelly and position caps.

**Study resources:**
- Kelly (1956), "A New Interpretation of Information Rate"
- Thorp (2006), "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"

---

## Deflated Sharpe Ratio

**Used in:** `src/backtest/metrics.py` — multiple testing adjustment

**The idea:** Bailey & Lopez de Prado (2014) adjust observed Sharpe for the number of strategy variants tried and return non-normality. DSR = P(SR* < SR_obs) where SR* is the expected maximum Sharpe under null from n trials.

**Why it matters in PRISM:** A Sharpe of 2.0 after trying 50 configurations is not the same as 2.0 on the first try.

**Pitfalls:** Requires honest trial count. Does not replace out-of-sample validation.

**Study resources:**
- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio"
- Harvey, Liu & Zhu (2016), "...and the Cross-Section of Expected Returns"

---

## Diebold-Mariano Test

**Used in:** `src/backtest/metrics.py` — forecast comparison vs market

**The idea:** Tests whether loss differential \(d_t = L(e^\text{model}_t) - L(e^\text{market}_t)\) is significantly different from zero, with HAC variance for serial correlation.

**Why it matters in PRISM:** If the model's Brier score is not significantly better than using the market price itself, there is no informational edge — only noise around an efficient price.

**Pitfalls:** Rejects only on average improvement; model could be better in some regimes and worse in others.

**Study resources:**
- Diebold & Mariano (1995), "Comparing Predictive Accuracy"
- Harvey, Leybourne & Newbold (1997), "Testing the equality of prediction mean squared errors"

---

## Probability of Backtest Overfitting (PBO)

**Used in:** `src/backtest/metrics.py` — simplified proxy via partition negative-Sharpe rate

**The idea:** Full CSCV (combinatorially symmetric cross-validation) measures the probability that the best in-sample strategy underperforms out-of-sample. PRISM implements a lightweight proxy: fraction of return partitions with negative mean.

**Why it matters in PRISM:** Signals when backtest performance is fragile across time splits.

**Pitfalls:** Full PBO requires CSCV machinery; the proxy is diagnostic only, not a substitute.

**Study resources:**
- Bailey et al. (2014), "The Probability of Backtest Overfitting"
