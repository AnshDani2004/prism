# PRISM: Prediction & Research Infrastructure for Sports Markets

Sport-agnostic prediction market pricing and alpha research framework for NFL and NBA.

## Research Question

Do in-play prediction market prices on sports contracts exhibit systematic staleness following scoring events, and if so, how large is the exploitable edge after realistic transaction costs?

## Methodology

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
│ Play-by-Play│───▶│ Win Prob     │───▶│ Market      │───▶│ Event-   │
│ NFL + NBA   │    │ Models       │    │ Interface   │    │ Driven   │
│ Kalshi/Poly │    │ (calibrated) │    │ (edge)      │    │ Backtest │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────┘
       │                  │                   │                  │
       └──────────────────┴───────────────────┴──────────────────┘
                              DuckDB (prism.duckdb)
```

1. **Data** — `nfl_data_py`, `nba_api`, Kalshi, Polymarket → DuckDB
2. **Models** — Bradley-Terry → Elo-MOV/Skellam → XGBoost + isotonic → Bayesian online (EKF)
3. **Market** — Game-clock alignment, edge = model − market mid, staleness classification
4. **Backtest** — Touch fills, Kalshi fees, 5s delay, fractional Kelly, block-bootstrap Sharpe

**2023 is the sacred test set.** Never tuned on it.

## Key Findings

Run the pipeline once to populate honest results:

```bash
python scripts/reproduce_findings.py
# or: python scripts/phase5_checkpoint.py
```

| Outcome | What it means |
|---------|---------------|
| Sharpe CI includes 0 | Strategy not statistically significant — report it |
| DM test fails to reject | Model adds no info over market price — report it |
| Positive Sharpe + significant CI | Staleness hypothesis supported after costs |

See [`research/findings.md`](research/findings.md) for the full academic writeup.

## Installation

```bash
cd prism
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # Kalshi credentials optional for Polymarket
```

## Quickstart

```bash
# Full pipeline
python scripts/ingest_sports.py
python scripts/ingest_markets.py
python scripts/compute_edges.py
python scripts/phase5_checkpoint.py

# Five-line reproduction
python scripts/reproduce_findings.py
```

## Project Structure

```
prism/
├── src/
│   ├── data/           # PBP + market ingestion, DuckDB, matching
│   ├── models/         # Win probability models + calibration
│   ├── market/         # Price alignment, edge, adverse selection
│   └── backtest/       # Event-driven backtester + metrics
├── config/             # settings, experiment_log.jsonl
├── tests/              # pytest (80%+ coverage)
├── scripts/            # ingestion, checkpoints, reproduce
├── notebooks/          # 01–06 research narrative
├── research/           # findings.md (PDF-ready writeup)
├── docs/               # DECISIONS.md + CONCEPTS.md
└── outputs/            # calibration plots, backtest charts
```

## Results

After running the pipeline, find artifacts at:

- `outputs/calibration/` — reliability diagrams, ECE by game clock
- `outputs/backtest/` — equity curve, drawdown, trade PnL histogram
- `config/experiment_log.jsonl` — auditable trial count for deflated Sharpe

Export notebooks to HTML:

```bash
./scripts/export_notebooks.sh
```

## Testing

```bash
pytest tests/ --cov=src --cov-report=term-missing
mypy src/ config/ --strict
ruff check src/ config/
```

## Documentation

| File | Purpose |
|------|---------|
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Why we made each design choice (interview spine) |
| [`docs/CONCEPTS.md`](docs/CONCEPTS.md) | Technical compendium + study resources |
| [`research/findings.md`](research/findings.md) | Academic-style research paper |

## Limitations

- Nominal kickoff times for game-clock ↔ wall-clock alignment (no broadcast sync)
- Historical price snapshots may miss sub-second quote flicker
- Thin in-play liquidity; volume caps bind frequently
- Single held-out test season (2023)
- No market impact model

## Resume Bullets (fill after backtest)

```
PRISM: Prediction Market Alpha Research Framework
- Built sport-agnostic pricing engine comparing calibrated models (Bradley-Terry,
  Elo-MOV/Skellam, XGBoost, Bayesian state-space) vs Kalshi/Polymarket across NFL/NBA
- Bayesian online EKF achieving ECE X.XX vs XGBoost X.XX on 2023 test set
- Detected price staleness ~X sec post-score; Sharpe X.XX (95% CI: [X, X]) walk-forward
- Pipeline: nfl_data_py + nba_api + Kalshi + Polymarket → DuckDB; 87%+ match rate
```

## References

Stern (1991); Karlis & Ntzoufras (2003); Murphy (1973); Bailey & López de Prado (2014); Diebold & Mariano (1995); Glosten & Milgrom (1985); Guo et al. (2017); Särkkä (2013).

## Open in Cursor

```bash
cursor ~/Desktop/prism/prism.code-workspace
```
