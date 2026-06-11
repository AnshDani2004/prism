# PRISM: Staleness in Sports Prediction Markets
## An Empirical Study of In-Play Pricing Efficiency

**Author:** Ansh [Portfolio Research]  
**Date:** June 2026  
**Code:** [github.com/.../prism](https://github.com)

---

## Abstract

We investigate whether in-play prediction market prices on NFL and NBA moneyline contracts exhibit systematic staleness following scoring events. Using a sport-agnostic research framework (PRISM), we build calibrated win-probability models from 2018–2023 play-by-play data, align them to historical Kalshi and Polymarket prices, and backtest a Kelly-sized strategy under realistic transaction costs. Our methodology enforces strict temporal separation (2023 held out as test set), touch-fills with Kalshi's nonlinear fee schedule, and block-bootstrap confidence intervals clustered by game. **Results are populated by running the end-to-end pipeline** (`scripts/phase5_checkpoint.py`); a null or negative Sharpe with rigorous methodology is a valid and publishable finding.

---

## 1. Introduction

Prediction markets aggregate dispersed information into tradable probabilities. In efficient markets, prices should update instantaneously when new information arrives — e.g., a touchdown or three-pointer. Yet market makers face latency, inventory risk, and model uncertainty. This creates a window where prices may lag true win probability.

**Research hypothesis:** In-play prediction market prices exhibit measurable staleness in the 0–30 seconds following scoring events, producing positive expected value for a calibrated model after fees and spread — *or* the market is efficient enough that no such edge survives realistic execution assumptions.

**Economic mechanism:** When a scoring event occurs, the true win probability jumps discontinuously. If market makers update quotes with delay Δt, any trader with a faster model can buy stale offers. The exploitable edge is bounded by: (1) model calibration error, (2) half-spread, (3) fees, (4) entry latency, and (5) adverse selection if the market knows something the model does not.

---

## 2. Data

### 2.1 Sports Play-by-Play
| Source | Coverage | Granularity |
|--------|----------|-------------|
| `nfl_data_py` | NFL 2018–2023 | ~20–30 states/game (scoring + quarter starts) |
| `nba_api` | NBA 2018–2023 | ~2-min samples + scoring events |

All states stored in DuckDB `game_states` with `seconds_remaining` as the primary clock index.

### 2.2 Market Data
| Source | Auth | Price Semantics |
|--------|------|-----------------|
| Kalshi REST API | RSA-PSS API key | `yes_bid`, `yes_ask`, mid for analysis |
| Polymarket Gamma + CLOB | None (read) | Outcome token prices |

### 2.3 Game–Contract Matching
Fuzzy matching via team alias dictionary + date proximity (70/30 weighting). Only pairs with confidence > 0.8 are used. Ambiguous matches (0.6–0.8) are logged and excluded.

### 2.4 Train/Validation/Test Split
| Set | Seasons | Purpose |
|-----|---------|---------|
| Train | 2018–2021 | Model fitting |
| Validation | 2022 | Hyperparameters, isotonic calibration, edge threshold |
| Test | 2023 | **Single final evaluation — never tuned on** |

---

## 3. Probability Models

### 3.1 Bradley-Terry (Baseline)
MLE paired-comparison model with learnable home advantage. Pre-game only.

### 3.2 Margin-of-Victory Rating Model
Elo with FiveThirtyEight-style MOV autocorrelation correction. NFL margins via Skellam distribution; NBA via Gaussian (Stern model). **Dixon-Coles explicitly rejected** — its low-score correction is meaningless for American sports scorelines.

### 3.3 XGBoost In-Play
13 engineered features (score differential, time elapsed, urgency, pre-game prior). Isotonic regression calibration on 2022 validation set.

### 3.4 Bayesian Online State-Space Model
Latent strength θₜ with random-walk transition; scoring events as observations. EKF (default) with analytic Jacobians; particle filter as cross-check. Hyperparameters fit via MLE on 2018–2021 scoring sequences.

### 3.5 Calibration Results
Run `python scripts/phase2_checkpoint.py` after data ingestion to populate ECE, Brier, and reliability diagrams in `outputs/calibration/`.

**Key insight:** High AUC does not imply good calibration. Prediction market edge requires calibrated probabilities, not just rankings.

---

## 4. Market Interface

### 4.1 Clock Alignment
Game clock → wall clock via `game_date` + nominal kickoff (18:00 UTC) + elapsed time. Scoring events snap to nearest market tick.

### 4.2 Edge Definition
\[
\text{edge}_t = \hat{p}^{\text{model}}_t - \hat{p}^{\text{market}}_t
\]
where market probability uses the **mid** of bid/ask.

### 4.3 Edge Classification
| Type | Definition |
|------|------------|
| `pre_game` | Full clock remaining |
| `inplay_staleness` | ≤ 30s since last score |
| `inplay_drift` | > 30s since last score |
| `cross_market` | Kalshi vs Polymarket disagreement |

### 4.4 Adverse Selection
We measure correction latency (seconds until price moves ≥ 1pp after a score) and edge realizability (does edge persist 5s after detection?).

---

## 5. Backtest

### 5.1 Design
- **Event-driven** (not vectorized)
- **Fills at touch:** buy YES at ask, sell YES at bid
- **Kalshi fee:** `ceil(0.07 × n × p × (1-p))` per fill
- **Entry delay:** 5 seconds; edge re-checked at execution
- **Sizing:** 0.25× Kelly, capped at 5% of capital
- **Liquidity:** max 10% of displayed touch volume

### 5.2 Statistical Tests
| Test | Purpose |
|------|---------|
| Block-bootstrap Sharpe CI (by game) | Significance without overstating precision |
| Deflated Sharpe Ratio | Adjust for number of configs tried (`config/experiment_log.jsonl`) |
| Diebold-Mariano | Is model Brier better than market-as-forecast? |
| PBO proxy | Fragility across time partitions |

### 5.3 Results
Run the full pipeline and checkpoint:

```bash
python scripts/ingest_sports.py
python scripts/ingest_markets.py
python scripts/compute_edges.py
python scripts/phase5_checkpoint.py
```

Fill in your empirical results here after a single test-set evaluation:

| Metric | Value |
|--------|-------|
| Total Return | _run checkpoint_ |
| Sharpe Ratio | _run checkpoint_ |
| 95% CI Sharpe (block bootstrap) | _run checkpoint_ |
| Deflated Sharpe Ratio | _run checkpoint_ |
| Hit Rate | _run checkpoint_ |
| N Trades | _run checkpoint_ |
| DM test p-value (model vs market) | _run checkpoint_ |

**Interpretation guide:**
- If 95% CI Sharpe includes zero → not statistically significant
- If DM test fails to reject → model adds no information over the market
- If DSR ≈ 0.5 → observed Sharpe is indistinguishable from multiple-testing luck

---

## 6. Conclusion

PRISM provides a reproducible, interview-defensible pipeline from raw play-by-play and market data to backtested strategy performance. Whether the staleness hypothesis is confirmed or refuted depends on the empirical output above — both outcomes are scientifically interesting:

- **If edge exists:** Quantify correction latency, edge decay, and net Sharpe after costs.
- **If edge does not exist:** The market is efficiently incorporating scoring information within our execution constraints; document why (fast repricing, wide spreads, fees).

### Limitations
1. **Clock alignment:** Nominal kickoff times, not broadcast-synced timestamps.
2. **Historical prices:** May not capture sub-second quote flicker.
3. **Liquidity:** Thin in-play markets; volume caps bind frequently.
4. **Single test season:** 2023 only; regime change risk.
5. **No market impact model:** Assumes our trades do not move prices.

### Future Work
- Real-time feed integration with verified kickoff timestamps
- Shin vig removal for sportsbook benchmarks
- Full CSCV probability of backtest overfitting
- Cross-market arbitrage (Kalshi vs Polymarket) as standalone strategy

---

## References

1. Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs.
2. Stern, H. (1991). On the probability of winning a football game.
3. Karlis, I., & Ntzoufras, I. (2003). Analysis of sports data by using bivariate Poisson models.
4. Murphy, A. H. (1973). A new vector partition of the probability score.
5. Bailey, D. H., & Lopez de Prado, M. (2014). The deflated Sharpe ratio.
6. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
7. Glosten, L. R., & Milgrom, P. R. (1985). Bid, ask and transaction prices.
8. Kelly, J. L. (1956). A new interpretation of information rate.
9. Guo, C., et al. (2017). On calibration of modern neural networks.
10. Särkkä, S. (2013). Bayesian Filtering and Smoothing.
