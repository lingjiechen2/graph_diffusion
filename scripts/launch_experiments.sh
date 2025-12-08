#!/usr/bin/env bash
# Launch training + evaluation on selected benchmarks.
# Usage:
#   ./scripts/launch_experiments.sh D1-BA-SIR D1-ER-SIR
# Env overrides:
#   DEVICE=cpu|cuda:0   EPOCHS=3   STEPS=200   ETA=40   BATCH=8   EDGE_PATH=/path/to/edges.txt
# For D2 benchmarks (Oregon2/Prost), set EDGE_PATH to the edge list file.

set -euo pipefail

if [ "$#" -eq 0 ]; then
  BENCHMARKS=("D1-BA-SIR")
else
  BENCHMARKS=("$@")
fi

DEVICE="${DEVICE:-cpu}"
EPOCHS="${EPOCHS:-3}"
STEPS="${STEPS:-200}"
ETA="${ETA:-40}"
BATCH="${BATCH:-8}"
EVAL_SAMPLES="${EVAL_SAMPLES:-0}"
MODE="${MODE:-history}"
SAVE_DIR="${SAVE_DIR:-runs}"
mkdir -p "$SAVE_DIR"

for BM in "${BENCHMARKS[@]}"; do
  CKPT="${SAVE_DIR}/${BM}.pt"
  echo ">>> Training benchmark ${BM}"
  python -m graph_diffusion.train \
    --benchmark "${BM}" \
    --epochs "${EPOCHS}" \
    --num-steps "${STEPS}" \
    --eta "${ETA}" \
    --batch-size "${BATCH}" \
    --device "${DEVICE}" \
    --mode "${MODE}" \
    --eval-samples "${EVAL_SAMPLES}" \
    ${EDGE_PATH:+--edge-path "${EDGE_PATH}"} \
    --save-path "${CKPT}"

  echo ">>> Evaluating benchmark ${BM}"
  python scripts/run_benchmarks.py \
    --benchmark "${BM}" \
    --num-steps "${STEPS}" \
    --eta "${ETA}" \
    --device "${DEVICE}" \
    --checkpoint "${CKPT}" \
    --mode "${MODE}" \
    ${EDGE_PATH:+--edge-path "${EDGE_PATH}"}
done
