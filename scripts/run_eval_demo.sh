#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p outputs

python3 scripts/make_demo_predictions.py

python3 evaluate.py \
  --pred outputs/demo_predictions.json \
  --split equation score semantic faithful \
  --out outputs/demo_eval_results.json

echo "Demo evaluation finished."
