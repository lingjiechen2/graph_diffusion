#!/usr/bin/env python3
"""
Run evaluation on a benchmark spec and report Macro F1, NRMSE, and gaps.
For D3 datasets, provide preprocessed histories externally; this script targets D1/D2 (simulated diffusion).
"""

import argparse
from pathlib import Path

import torch

from graph_diffusion.beta_diffusion import HistoryBetaDiffusion
from graph_diffusion.benchmarks import BENCHMARKS, BenchmarkSpec, make_dataset_from_spec
from graph_diffusion.metrics import evaluate_history
from graph_diffusion.model import SpaceTimeGNN
from graph_diffusion.sampling import sample_history
from graph_diffusion.schedules import make_alpha_schedule
from graph_diffusion.sir_times import decode_hitting_times, sample_sir_times


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark evaluation for Graph Beta Diffusion.")
    p.add_argument("--benchmark", required=True, choices=BENCHMARKS.keys())
    p.add_argument("--num-steps", type=int, default=200, help="Diffusion steps K.")
    p.add_argument("--eta", type=float, default=40.0)
    p.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint to load.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num-samples", type=int, default=4, help="Number of histories to sample/evaluate.")
    p.add_argument("--edge-path", type=str, default=None, help="Path to edge list for external graphs (D2).")
    p.add_argument("--mode", type=str, default="history", choices=["history", "sir-times"], help="Evaluation space.")
    return p.parse_args()


def load_model(spec: BenchmarkSpec, timesteps: int, args) -> SpaceTimeGNN:
    if args.mode == "sir-times":
        dim = 2
    else:
        dim = (timesteps + 1) * spec.history_dim
    model = SpaceTimeGNN(history_dim=dim, hidden_dim=128, num_layers=3)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(ckpt)
        print(f"Loaded checkpoint from {args.checkpoint}")
    return model.to(args.device)


def main():
    args = parse_args()
    device = torch.device(args.device)
    spec = BENCHMARKS[args.benchmark]
    dataset = make_dataset_from_spec(spec, num_samples=args.num_samples, device=device, edge_path=args.edge_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    alphas = make_alpha_schedule(args.num_steps)
    diffusion = HistoryBetaDiffusion(eta=args.eta, alphas=alphas.to(device))
    model = load_model(spec, spec.timesteps, args)
    model.eval()

    results = []
    for batch in loader:
        Y_true = batch["Y"].to(device)
        A = batch["A"].to(device)
        if args.mode == "sir-times":
            final_snapshot = Y_true[:, :, -1, :]
            Y_times = sample_sir_times(diffusion, model, A, final_snapshot, spec.timesteps, num_steps=args.num_steps)
            Y_pred = decode_hitting_times(Y_times, spec.timesteps)
        else:
            mask = torch.zeros_like(Y_true)
            mask[:, :, -1, :] = 1.0
            Y_obs = mask * Y_true
            Y_pred = sample_history(diffusion, model, A, Y_obs, mask)
        metrics = evaluate_history(Y_true, Y_pred)
        results.append(metrics)
        print(f"Sample metrics: {metrics}")

    # aggregate
    if results:
        agg = {k: sum(m[k] for m in results) / len(results) for k in results[0].keys()}
        print(f"\nAveraged over {len(results)} samples: {agg}")


if __name__ == "__main__":
    main()
