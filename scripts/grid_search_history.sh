#!/usr/bin/env bash
# Grid search over diffusion steps for history-mode runs, training + evaluating each run.
# Usage: ./scripts/grid_search_history.sh

set -euo pipefail

EDGE_PATH="${EDGE_PATH:-/path/to/edges.txt}"
LOG_FILE="${LOG_FILE:-runs/gridsearch_history_D1-BA-SIR-batch8-eta40.txt}"
mkdir -p "$(dirname "$LOG_FILE")"

for STEPS in $(seq 100 100 1500); do
  {
    echo "===== $(date --iso-8601=seconds) | STEPS=${STEPS} ====="
    DEVICE=cuda:0 \
    EPOCHS=10 \
    STEPS="${STEPS}" \
    ETA=40 \
    BATCH=8 \
    MODE=history \
    EDGE_PATH="${EDGE_PATH}" \
    SAVE_DIR=runs \
    ./scripts/launch_experiments.sh D1-BA-SIR
    echo
  } 2>&1 | tee -a "$LOG_FILE"
done
