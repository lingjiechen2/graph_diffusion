#!/usr/bin/env bash
# ST backbone — GPU6: D2-Prost (SI/SIR) + D3 all (BrFarmers/Covid/Hebrew/Pol)
set -e
DITTO=/home/lingjie7/KDD23-DITTO/input
PY=/home/lingjie7/anaconda3/envs/d1/bin/python
CKPT=/home/lingjie7/graph_diffusion/runs
mkdir -p "$CKPT"

run() {
  local bm=$1 bs=$2 ns=$3 epochs=$4; shift 4
  echo ""; echo "-------- $bm  bs=$bs  ns=$ns  epochs=$epochs --------"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=6 $PY -m graph_diffusion.train \
    --benchmark "$bm" --ditto-dir "$DITTO" \
    --num-steps 200 --eta 40 --epochs "$epochs" \
    --batch-size "$bs" --lr 1e-3 --device cuda \
    --time-embed-dim 16 --cond-drop-prob 0.1 \
    --backbone st --num-heads 4 --ffn-mult 2 \
    --save-path "$CKPT/${bm}-st.pt" "$@"
}

echo "========================================================"
echo " ST backbone  GPU6  started $(date)"
echo "========================================================"

# D2 Prost — 15810 nodes, bs=1
run D2-Prost-SI   1 128 30 --model-type si --eval-samples 2
run D2-Prost-SIR  1 128 30               --eval-samples 2

# D3 — real history eval
run D3-BrFarmers  4 256 50 --model-type si --eval-samples 1 --eval-real
run D3-Covid      4 256 50               --eval-samples 1 --eval-real
run D3-Hebrew     2 256 50               --eval-samples 1 --eval-real
run D3-Pol        1  64 20 --model-type si --eval-samples 1 --eval-real

echo ""
echo "========================================================"
echo " GPU6 done  $(date)"
echo "========================================================"
