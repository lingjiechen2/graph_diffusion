#!/usr/bin/env bash
# Full training run for all 12 DITTO benchmarks on GPU3.
# Results are printed at the end of each benchmark.
set -e

DITTO=/home/lingjie7/KDD23-DITTO/input
PY=/home/lingjie7/anaconda3/envs/d1/bin/python
CKPT_DIR=/home/lingjie7/graph_diffusion/runs
mkdir -p "$CKPT_DIR"

BACKBONE=${1:-inpaint}   # pass "st" as first arg for ST backbone
echo "========================================================"
echo " Full training  backbone=$BACKBONE  started $(date)"
echo "========================================================"

run() {
  local bm=$1; local bs=$2; local ns=$3; local epochs=$4; local extra="${@:5}"
  echo ""
  echo "-------- $bm  bs=$bs  ns=$ns  epochs=$epochs --------"
  CUDA_VISIBLE_DEVICES=3 $PY -m graph_diffusion.train \
    --benchmark "$bm" \
    --ditto-dir "$DITTO" \
    --num-steps 200 --eta 40 \
    --epochs "$epochs" \
    --batch-size "$bs" \
    --lr 1e-3 \
    --device cuda \
    --time-embed-dim 16 \
    --cond-drop-prob 0.1 \
    --backbone "$BACKBONE" \
    --save-path "$CKPT_DIR/${bm}-${BACKBONE}.pt" \
    $extra
}

# D1 — synthetic graphs + synthetic diffusion (fast, 1000 nodes)
run D1-BA-SI    8 1024 100 --model-type si --eval-samples 5
run D1-BA-SIR   8 1024 100               --eval-samples 5
run D1-ER-SI    8 1024 100 --model-type si --eval-samples 5
run D1-ER-SIR   8 1024 100               --eval-samples 5

# D2 — synthetic diffusion on real graphs
run D2-Oregon2-SI   2 256 30 --model-type si --eval-samples 3
run D2-Oregon2-SIR  2 256 30               --eval-samples 3
run D2-Prost-SI     1 128 30 --model-type si --eval-samples 2
run D2-Prost-SIR    1 128 30               --eval-samples 2

# D3 — real diffusion on real graphs (eval on single real history)
run D3-BrFarmers 4 256 50 --model-type si --eval-samples 1 --eval-real
run D3-Covid     4 256 50               --eval-samples 1 --eval-real
run D3-Hebrew    2 256 50               --eval-samples 1 --eval-real
run D3-Pol       1  64 20 --model-type si --eval-samples 1 --eval-real

echo ""
echo "========================================================"
echo " All done  backbone=$BACKBONE  $(date)"
echo "========================================================"
