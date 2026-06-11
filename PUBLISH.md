# Publishing PRISM to GitHub

## 1. Initialize and commit (if not already done)

```bash
cd ~/Desktop/prism
git init
git add .
git commit -m "$(cat <<'EOF'
Initial commit: PRISM prediction market research framework.

End-to-end pipeline from NFL/NBA play-by-play and Kalshi/Polymarket prices
through calibrated models, edge detection, and event-driven backtesting.
EOF
)"
```

## 2. Create GitHub repo and push

```bash
gh repo create prism --public --source=. --remote=origin --push
```

Or manually:

```bash
# Create empty repo at github.com/new named "prism"
git remote add origin git@github.com:YOUR_USERNAME/prism.git
git branch -M main
git push -u origin main
```

## 3. Run data pipeline (local only — not committed)

```bash
source venv/bin/activate
pip install -r requirements-dev.txt

# Fast: NFL only (~5 min)
python scripts/ingest_nfl_only.py

# Markets (Polymarket works without API keys)
python scripts/ingest_markets.py

# Full edges + backtest
python scripts/compute_edges.py
python scripts/phase5_checkpoint.py
```

## 4. Portfolio link

After push, verify the repo URL in:

- `~/Desktop/personal-portfolio/content/projects/prism.mdx` → `repo:` field
- Update `https://github.com/anshdani/prism` if your username differs

## 5. CI

GitHub Actions runs on push: `.github/workflows/test.yml` (pytest + mypy + ruff).
