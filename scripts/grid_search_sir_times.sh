#!/usr/bin/env bash
# Grid search for SIR-time diffusion runs.
# Usage: ./scripts/grid_search_sir_times.sh

set -euo pipefail

# Optional overrides
EDGE_PATH="${EDGE_PATH:-/path/to/edges.txt}"    # only needed for external graphs
LOG_FILE="${LOG_FILE:-runs/gridsearch_sir_times_D1-BA-SIR-ETA-50.txt}"
SAVE_DIR="${SAVE_DIR:-runs/sir-times}"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$SAVE_DIR"

for STEPS in $(seq 100 100 500); do
  {
    echo "===== $(date --iso-8601=seconds) | MODE=sir-times | STEPS=${STEPS} ====="
    MODE=sir-times \
    DEVICE=cuda:0 \
    EPOCHS=10 \
    STEPS="${STEPS}" \
    ETA=50 \
    BATCH=8 \
    EDGE_PATH="${EDGE_PATH}" \
    SAVE_DIR="${SAVE_DIR}" \
    ./scripts/launch_experiments.sh D1-BA-SIR
    echo
  } 2>&1 | tee -a "$LOG_FILE"
done
