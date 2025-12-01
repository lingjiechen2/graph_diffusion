# Graph Beta Diffusion for History Reconstruction

This project implements the “graph-shaped tensor” formulation of Graph Beta Diffusion (GBD) for reconstructing diffusion histories from a single observed snapshot on a graph. The code follows the specification in the prompt: forward multiplicative Beta diffusion over node–time–state tensors, a GNN-based reverse process, KLUB-style training, and masked conditional sampling that clamps the final snapshot during reverse diffusion.

## What’s here
- `graph_diffusion/beta_diffusion.py` — forward corruption, reverse posterior targets, KLUB loss, and helper math for Beta distributions.
- `graph_diffusion/model.py` — a lightweight graph message-passing network that operates on the space–time history tensor with diffusion step embeddings.
- `graph_diffusion/schedules.py` — utilities to create decreasing Alpha schedules (`alpha_0=1` to `alpha_K≈0`) and register Beta concentrations.
- `graph_diffusion/sampling.py` — conditional reverse sampling with inpainting-style clamping of the observed snapshot.
- `graph_diffusion/data.py` — simple synthetic SIR-style generator to create training histories on toy graphs.
- `graph_diffusion/train.py` — example training loop wiring the pieces together.
- `graph_diffusion/benchmarks.py` — codified benchmark specs from D1–D3 (BA/ER synthetic, Oregon/Prost synthetic diffusion, real diffusion placeholders) plus graph loaders.
- `graph_diffusion/metrics.py` — Macro F1, NRMSE of hitting times, and performance gaps.
- `scripts/run_benchmarks.py` — run evaluation for a benchmark spec (D1/D2) with optional checkpoint.
- `scripts/debug_pipeline.py` — tiny end-to-end sanity check on a small synthetic graph.
- `scripts/launch_experiments.sh` — convenience launcher to train + evaluate selected benchmarks (set EDGE_PATH for D2).

## Quick start (conceptual)
```bash
python -m graph_diffusion.train \
  --num-nodes 32 --timesteps 6 --history-dim 3 \
  --num-steps 200 --eta 40.0 --lr 1e-3 --epochs 10
```
This uses a toy SIR generator and trains the reverse model for a few epochs. At inference, pass an observed final snapshot to `sample_history` to reconstruct plausible histories.

### Benchmarks
- D1 (synthetic graphs + synthetic diffusion): BA/ER graphs with SI/SIR (T=10, β_I=β_R=0.1, 5% seeds). Use `--benchmark D1-BA-SIR` (or D1-BA-SI/D1-ER-SI/D1-ER-SIR) or call `scripts/run_benchmarks.py --benchmark ...`.
- D2 (synthetic diffusion on real graphs): Oregon2/Prost. Provide `--edge-path` to an edge list when using `scripts/run_benchmarks.py` with `--benchmark D2-Oregon2-SIR` (or SI).
- D3 (real diffusion on real graphs): placeholders for BrFarmers, Pol, Covid, Hebrew. Supply preprocessed histories and graphs externally; see URLs in `graph_diffusion/benchmarks.py`.

### Evaluation metrics
`graph_diffusion/metrics.py` implements Macro F1, NRMSE over hitting times, and the performance-gap formula. `scripts/run_benchmarks.py` reports these automatically.

## Notes
- The implementation is self-contained (only PyTorch and NumPy are required).
- The reverse kernel and loss closely mirror the equations in the prompt; KLUB’s correction term is implemented explicitly.
- Everything treats histories as a single tensor `Y ∈ [0,1]^{N×(T+1)×d_s}`; adjacency is used only as conditioning in the GNN.
