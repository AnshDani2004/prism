# PRISM Decision Log

Plain-language record of design choices. Append-only — never rewrite history.

---

## 2026-06-11 — Project scope: NFL and NBA only

**What was decided:** PRISM focuses on NFL and NBA in-play prediction markets, not a broader multi-sport platform.

**Why:** These two sports have the best free play-by-play data (`nfl_data_py`, `nba_api`) and the most liquid prediction market contracts on Kalshi and Polymarket. Starting narrow allows depth on microstructure and calibration rather than shallow coverage of many sports.

**What we explicitly did NOT do:** Soccer, MLB, NHL, or college sports. Adding them later is straightforward once the pipeline is proven, but they would dilute Phase 1 effort without improving the core research question.

**Revisit if:** We secure reliable free PBP data for another sport with active prediction market listings.

---

## 2026-06-11 — Use DuckDB instead of PostgreSQL

**What was decided:** All data lives in a single local DuckDB file at `data/prism.duckdb`.

**Why:** This is an analytical research project, not a serving system. DuckDB gives columnar query speed with zero server setup. Postgres would add operational overhead with no benefit at this scale (~10M rows).

**What we explicitly did NOT do:** No cloud database, no ORM layer. Both add complexity that signals over-engineering for a research project.

**Revisit if:** The project ever needs concurrent writers or live serving.

---

## 2026-06-11 — Store bid/ask separately; use mid for edge, touch for fills

**What was decided:** `market_prices` stores `yes_bid` and `yes_ask` where available. Edge computation uses the mid-price; the backtest engine (Phase 5) will fill at the relevant side of the book.

**Why:** Conflating mid with fill price silently inflates backtest PnL. Interviewers at prop shops will ask about this immediately. Separating the two makes the assumption explicit and auditable.

**What we explicitly did NOT do:** Store only a single price column. That would hide spread costs.

**Revisit if:** We get full order book snapshots and want queue-position modeling.

---

## 2026-06-11 — Kalshi RSA-PSS API key authentication

**What was decided:** Kalshi integration uses API key ID + RSA private key PEM file with RSA-PSS/SHA256 signing per current docs at https://docs.kalshi.com.

**Why:** Kalshi deprecated email/password login. The signed-request scheme is the only supported auth path for the trade API.

**What we explicitly did NOT do:** Hardcode credentials or use deprecated auth flows.

**Revisit if:** Kalshi changes signing algorithm or base URL (verify against live docs before each release).

---

## 2026-06-11 — Polymarket Gamma + CLOB instead of The Graph subgraph

**What was decided:** Market discovery uses Gamma API (`gamma-api.polymarket.com`); price history uses CLOB API (`clob.polymarket.com/prices-history`).

**Why:** The Graph hosted subgraph for Polymarket was deprecated. Polymarket's own APIs are free, documented, and sufficient for historical research.

**What we explicitly did NOT do:** Subgraph queries or paid data vendors.

**Revisit if:** Polymarket changes API structure or rate limits materially.

---

## 2026-06-11 — Fuzzy game-contract matching with 0.8 confidence threshold

**What was decided:** Contracts are matched to games via team alias dictionary + date proximity scoring (70% team, 30% date). Only pairs above 0.8 confidence are stored; 0.6–0.8 are flagged as ambiguous and excluded.

**Why:** Contract titles are inconsistently formatted ("Chiefs ML" vs "Kansas City Chiefs to win"). Fuzzy matching with a high threshold reduces false links that would poison edge analysis. Manual review of ambiguous band is logged but not auto-included.

**What we explicitly did NOT do:** Exact string matching only — match rate would be too low. Manual mapping spreadsheets — doesn't scale and isn't reproducible.

**Revisit if:** Match rate falls below 50% after ingestion; consider ML-based entity resolution or Kalshi structured metadata fields.

---

## 2026-06-11 — Reject Dixon-Coles; use Elo-MOV + Skellam/Gaussian margins

**What was decided:** Phase 2 margin model uses Elo ratings with margin-of-victory adjustment (FiveThirtyEight-style autocorrelation correction), Skellam distributions for NFL score differential, and a Gaussian (Stern) margin model for NBA. Dixon-Coles was explicitly rejected.

**Why:** Dixon-Coles is a soccer model whose low-score correlation correction (0-0, 1-0, 1-1 adjustments) is meaningless for NFL/NBA scorelines. Applying it to American sports is cargo-cult modeling that would fail a quant interview question.

**What we explicitly did NOT do:** Poisson goals model, Dixon-Coles τ correction, or any soccer-specific machinery.

**Revisit if:** We add soccer markets; then Dixon-Coles becomes appropriate in its native domain.

---

## 2026-06-11 — Strict temporal split: 2023 is sacred test set

**What was decided:** Models train on 2018–2021, hyperparameters and isotonic calibration tune on 2022, and 2023 is held out until final evaluation. This split is enforced in `XGBInPlayModel`.

**Why:** Tuning on the test season is data snooping. Prop shops will ask how you know results aren't overfit; a frozen test year with documented protocol is the answer.

**What we explicitly did NOT do:** Random k-fold cross-validation across all seasons (violates temporal structure).

**Revisit if:** We accumulate multiple post-2023 seasons and adopt walk-forward retraining.

---

## 2026-06-11 — Isotonic regression as post-hoc calibration layer

**What was decided:** XGBoost outputs are calibrated via isotonic regression fit on the 2022 validation set only, after hyperparameter tuning completes on training data.

**Why:** Tree models rank well but produce poorly calibrated probabilities. Isotonic regression is nonparametric, monotonic, and won't invert the ranking — it only fixes probability scale.

**What we explicitly did NOT do:** Platt scaling (assumes sigmoid distortion; less flexible). Calibrating on training data (overfits).

**Revisit if:** Validation set is too small for stable isotonic fit (<500 samples).

---

## 2026-06-11 — EKF default inference; particle filter as validation cross-check

**What was decided:** The Bayesian online model uses an Extended Kalman Filter with analytic Jacobians as the default inference method. A particle filter (SMC) is implemented as an alternative and cross-check, not the production path.

**Why:** EKF is orders of magnitude faster for backtesting thousands of games. The particle filter confirms EKF approximations are reasonable (they should agree within ~3 percentage points). This mirrors the particle filter work in the bayesian-market-filters project — same inference algorithm, different state space.

**What we explicitly did NOT do:** Numerical Jacobians (slower, less reproducible). Particle filter as default (unnecessary compute for research backtests).

**Revisit if:** EKF and particle filter systematically disagree (>5pp) on validation games — signals linearization failure and particle filter should become default.

---

## 2026-06-11 — Proportional vig removal for sportsbook odds

**What was decided:** When comparing to sportsbook odds, raw implied probabilities are normalized proportionally: \(p_\text{home} = q_\text{home} / (q_\text{home} + q_\text{away})\). Shin's method was not implemented in Phase 4.

**Why:** Proportional normalization is the standard simple approach, easy to defend in interviews, and sufficient when we primarily compare against Kalshi/Polymarket (no vig). Sportsbook comparison is secondary.

**What we explicitly did NOT do:** Shin's method (better for favorite-longshot bias but more complex). Comparing to vigged odds without removal.

**Revisit if:** Sportsbook odds become a primary benchmark; implement Shin and document the difference.

---

## 2026-06-11 — Game-clock alignment via nominal kickoff + scoring-event anchors

**What was decided:** Map game clock to wall clock using game_date + nominal 18:00 UTC kickoff + elapsed game time. Scoring events snap to the nearest market tick as alignment anchors.

**Why:** We lack broadcast-synced wall-clock timestamps in free PBP data. Nominal kickoff is imperfect but reproducible; scoring anchors reduce drift within a game.

**What we explicitly did NOT do:** Manual broadcast sync or paid timestamped feeds. Claiming sub-second alignment accuracy.

**Revisit if:** We obtain reliable kickoff timestamps per game (e.g., from market open times).

---

## 2026-06-11 — Event-driven backtester with touch fills and Kalshi fee formula

**What was decided:** Phase 5 backtester is event-driven (not vectorized). Edge is detected against mid, but fills execute at the touch (buy at ask, sell at bid). Kalshi fees use `ceil(0.07 × n × p × (1-p))` per contract.

**Why:** Vectorized backtests hide execution assumptions. Mid-fill backtests are the most common form of quiet fraud in systematic trading research. The Kalshi fee formula is nonlinear in price — maximal at p=0.50 — which materially changes which edges are tradeable.

**What we explicitly did NOT do:** Vectorized PnL. Flat 2-cent fees. Infinite liquidity.

**Revisit if:** Kalshi changes fee schedule — verify against kalshi.com/docs before any production run.

---

## 2026-06-11 — Honest findings template; no fabricated results

**What was decided:** `research/findings.md` and README resume bullets use placeholders until `phase5_checkpoint.py` is run once on the held-out 2023 test set. Null or negative Sharpe is reported as a valid finding.

**Why:** Fabricating backtest results is the fastest way to fail a quant interview. The framework is the deliverable; the empirical answer is whatever the data shows after one honest test-set evaluation.

**What we explicitly did NOT do:** Hardcode positive Sharpe in docs. Tune edge threshold on 2023.

**Revisit if:** New seasons are added — re-run checkpoint once per major methodology change.

---

## 2026-06-11 — Block bootstrap Sharpe CI clustered by game

**What was decided:** Sharpe ratio confidence intervals use block bootstrap resampling games (not individual trades) as the unit of replication.

**Why:** Multiple trades in the same game share a single binary outcome. IID resampling of trades understates variance and produces falsely significant Sharpes — a common interview trap.

**What we explicitly did NOT do:** IID bootstrap on per-trade returns. Point-estimate Sharpe without confidence intervals.

**Revisit if:** We trade cross-game portfolios where correlation structure differs materially from game-level clustering.

---

## 2026-06-11 — Experiment log for deflated Sharpe ratio

**What was decided:** Every backtest configuration is appended to `config/experiment_log.jsonl`. Deflated Sharpe uses the count of logged trials.

**Why:** Bailey & Lopez de Prado showed that testing many strategies inflates observed Sharpe. Auditable trial count is intellectual honesty infrastructure, not bureaucracy.

**What we explicitly did NOT do:** Report best Sharpe across unlogged experiments. Cherry-pick threshold on test set.

---

## 2026-06-11 — Anchor gitignore `data/` to repo root

**What was decided:** `.gitignore` uses `/data/` (root-anchored) instead of `data/` to exclude only the DuckDB directory at `data/prism.duckdb`, not the `src/data/` Python package.

**Why:** An unanchored `data/` pattern matches any directory named `data` anywhere in the tree. The entire `src/data/` ingestion layer was silently excluded from version control. Local development kept working because files existed on disk, but CI on a fresh checkout failed immediately — the first real test of reproducibility.

**What we explicitly did NOT do:** Rename `src/data/` to avoid the collision. The package name is correct; the ignore rule was wrong.

**Revisit if:** We add another top-level data directory with a different name, or split raw data from the database path.
