#!/usr/bin/env bash
# Run Chapter 4 evaluation pipeline (after trials + training, or on existing data).
set -euo pipefail

THESIS_DIR="${THESIS_DIR:-$HOME/thesis-sim}"
VENV_PY="$THESIS_DIR/venv/bin/python"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$THESIS_DIR"
export PATH="$THESIS_DIR/bin:$PATH"

cd "$THESIS_DIR"

echo "══ Chapter 4 pipeline ══"

if [ -f "$THESIS_DIR/output/kpi_master_dataset.csv" ]; then
  if [ ! -f "$THESIS_DIR/models/lstm_model.h5" ]; then
    echo "[1/3] Training ML models..."
    "$VENV_PY" preprocess_and_train.py
  else
    echo "[1/3] Models present — skipping training (delete models/ to retrain)"
  fi
else
  echo "ERROR: Missing output/kpi_master_dataset.csv — run: python3 run_all_trials.py"
  exit 1
fi

echo "[2/3] MAPE-K evaluation (Table 4.6)..."
"$VENV_PY" mapek_loop.py --model all

echo "[3/3] Chapter 4 figures..."
"$VENV_PY" scripts/generate_chapter4_figures.py

echo ""
echo "Done. See ~/thesis-sim/reports/"
