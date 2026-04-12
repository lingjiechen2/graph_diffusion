#!/usr/bin/env bash
# ST backbone — GPU3: D1 (all 4) + D2-Oregon2 (SI/SIR)
set -e
DITTO=/home/lingjie7/KDD23-DITTO/input
PY=/home/lingjie7/anaconda3/envs/d1/bin/python
CKPT=/home/lingjie7/graph_diffusion/runs
mkdir -p "$CKPT"

run() {
  local bm=$1 bs=$2 ns=$3 epochs=$4; shift 4
  echo ""; echo "-------- $bm  bs=$bs  ns=$ns  epochs=$epochs --------"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=3 $PY -m graph_diffusion.train \
    --benchmark "$bm" --ditto-dir "$DITTO" \
    --num-steps 200 --eta 40 --epochs "$epochs" \
    --batch-size "$bs" --lr 1e-3 --device cuda \
    --time-embed-dim 16 --cond-drop-prob 0.1 \
    --backbone st --num-heads 4 --ffn-mult 2 \
    --save-path "$CKPT/${bm}-st.pt" "$@"
}

echo "========================================================"
echo " ST backbone  GPU3  started $(date)"
echo "========================================================"

# D1 — 1000 nodes, fast
run D1-BA-SI   8 1024 100 --model-type si --eval-samples 5
run D1-BA-SIR  8 1024 100               --eval-samples 5
run D1-ER-SI   8 1024 100 --model-type si --eval-samples 5
run D1-ER-SIR  8 1024 100               --eval-samples 5

# D2 Oregon2 — 11461 nodes, bs=1 for ST (OOM at bs=2)
run D2-Oregon2-SI   1 256 30 --model-type si --eval-samples 3
run D2-Oregon2-SIR  1 256 30               --eval-samples 3

echo ""
echo "========================================================"
echo " GPU3 done  $(date)"
echo "========================================================"
