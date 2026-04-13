#!/usr/bin/env bash
# Full D1 benchmark sweep (BA/ER × SI/SIR) × all 3 backbones on GPU 6.
# Uses the fixed codebase (correct prior, infection prob, random seeds).
# Results saved to runs/D1-*-{inpaint,st,digress}-fixed.pt
set -e

GPU=6
DITTO=/home/lingjie7/KDD23-DITTO/input
PY=/home/lingjie7/anaconda3/envs/d1/bin/python
CKPT_DIR=/home/lingjie7/graph_diffusion/runs
LOG_DIR=/home/lingjie7/graph_diffusion/logs
mkdir -p "$CKPT_DIR" "$LOG_DIR"

COMMON="--ditto-dir $DITTO --num-steps 200 --eta 40
        --batch-size 8 --lr 1e-3 --epochs 100
        --time-embed-dim 16 --cond-drop-prob 0.1
        --eval-samples 5 --device cuda"

echo "========================================================"
echo " D1 fixed sweep  GPU=$GPU  $(date)"
echo "========================================================"

run() {
  local bm=$1 backbone=$2 extra="${@:3}"
  echo ""
  echo "-------- $bm  backbone=$backbone --------"
  CUDA_VISIBLE_DEVICES=$GPU $PY -m graph_diffusion.train \
    --benchmark "$bm" $COMMON \
    --backbone "$backbone" $extra \
    --save-path "$CKPT_DIR/${bm}-${backbone}-fixed.pt" \
    2>&1 | tee "$LOG_DIR/${bm}-${backbone}-fixed.log"
}

# --- Inpaint (baseline) ---
run D1-BA-SI    inpaint --model-type si
run D1-BA-SIR   inpaint
run D1-ER-SI    inpaint --model-type si
run D1-ER-SIR   inpaint

# --- ST ---
run D1-BA-SI    st      --model-type si
run D1-BA-SIR   st
run D1-ER-SI    st      --model-type si
run D1-ER-SIR   st

# --- DiGress ---
run D1-BA-SI    digress --model-type si
run D1-BA-SIR   digress
run D1-ER-SI    digress --model-type si
run D1-ER-SIR   digress

echo ""
echo "========================================================"
echo " D1 fixed sweep done  $(date)"
echo "========================================================"
