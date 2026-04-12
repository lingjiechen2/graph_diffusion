#!/usr/bin/env bash
# Test: D1-BA-SIR with large epochs to probe convergence (DSCT paper insight).
# Runs 3 backbones sequentially on GPU 2.
# Inpaint: 2000 ep | ST: 1000 ep | DiGress: 1000 ep

set -e
GPU=2
DITTO=/home/lingjie7/KDD23-DITTO/input
PY=/home/lingjie7/anaconda3/envs/d1/bin/python
CKPT_DIR=/home/lingjie7/graph_diffusion/runs
LOG_DIR=/home/lingjie7/graph_diffusion/logs
mkdir -p "$CKPT_DIR" "$LOG_DIR"

BM=D1-BA-SIR
COMMON="--benchmark $BM --ditto-dir $DITTO --num-steps 200 --eta 40
        --batch-size 8 --lr 3e-4 --device cuda
        --time-embed-dim 16 --cond-drop-prob 0.1 --eval-samples 5
        --model-type sir"

echo "========================================================"
echo " Large-epoch convergence test  GPU=$GPU  $(date)"
echo "========================================================"

echo ""
echo "---- [1/3] Inpaint  2000 epochs ----"
CUDA_VISIBLE_DEVICES=$GPU $PY -m graph_diffusion.train \
    $COMMON --backbone inpaint --epochs 2000 \
    --save-path "$CKPT_DIR/${BM}-inpaint-2000ep.pt" \
    2>&1 | tee "$LOG_DIR/${BM}-inpaint-2000ep.log"

echo ""
echo "---- [2/3] ST  1000 epochs ----"
CUDA_VISIBLE_DEVICES=$GPU $PY -m graph_diffusion.train \
    $COMMON --backbone st --epochs 1000 \
    --save-path "$CKPT_DIR/${BM}-st-1000ep.pt" \
    2>&1 | tee "$LOG_DIR/${BM}-st-1000ep.log"

echo ""
echo "---- [3/3] DiGress  1000 epochs ----"
CUDA_VISIBLE_DEVICES=$GPU $PY -m graph_diffusion.train \
    $COMMON --backbone digress --epochs 1000 \
    --save-path "$CKPT_DIR/${BM}-digress-1000ep.pt" \
    2>&1 | tee "$LOG_DIR/${BM}-digress-1000ep.log"

echo ""
echo "========================================================"
echo " All done  $(date)"
echo "========================================================"
