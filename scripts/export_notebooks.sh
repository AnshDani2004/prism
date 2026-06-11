#!/usr/bin/env bash
# Export notebooks to HTML for portfolio display.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/notebooks
for nb in notebooks/*.ipynb; do
  name=$(basename "$nb" .ipynb)
  jupyter nbconvert --to html "$nb" --output-dir outputs/notebooks --output "$name.html"
  echo "Exported outputs/notebooks/$name.html"
done
