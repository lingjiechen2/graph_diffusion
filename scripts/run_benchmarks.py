#!/usr/bin/env python3
"""
Run evaluation on a benchmark spec and report Macro F1, NRMSE, and gaps.

D1/D2 specs: evaluate on synthetically generated histories.
D3  specs:   evaluate on the single real history from the DITTO dataset
             (pass --ditto-dir pointing to KDD23-DITTO/input/).
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from graph_diffusion.benchmarks import (
    BENCHMARKS,
    BenchmarkSpec,
    load_real_history_from_spec,
    make_dataset_from_spec,
)
from graph_diffusion.beta_diffusion import HistoryBetaDiffusion
from graph_diffusion.metrics import evaluate_history
from graph_diffusion.model import HistoryInpaintGNN, SpaceTimeGNN
from graph_diffusion.model_st import STGraphTransformer
from graph_diffusion.model_digress import DigressSTTransformer
from graph_diffusion.sampling import sample_history
from graph_diffusion.schedules import make_alpha_schedule
from graph_diffusion.sir_times import decode_hitting_times, sample_sir_times


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark evaluation for Graph Beta Diffusion.")
    p.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()))
    p.add_argument("--num-steps", type=int, default=200, help="Diffusion steps K.")
    p.add_argument("--eta", type=float, default=40.0)
    p.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num-samples", type=int, default=4, help="Synthetic eval samples (D1/D2).")
    p.add_argument("--ditto-dir", type=str, default=None,
                   help="Path to KDD23-DITTO input/ directory (required for D2/D3).")
    p.add_argument("--mode", type=str, default="history", choices=["history", "sir-times"])
    p.add_argument("--backbone", type=str, default="inpaint",
                   choices=["inpaint", "st", "digress"],
                   help="Model backbone matching the trained checkpoint.")
    p.add_argument("--time-embed-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--ffn-mult", type=int, default=2)
    p.add_argument("--edge-dim", type=int, default=16)
    return p.parse_args()


def build_model(spec: BenchmarkSpec, timesteps: int, args, A: torch.Tensor = None):
    device = args.device
    if args.mode == "sir-times":
        model = SpaceTimeGNN(feature_dim=2, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    elif args.backbone == "st":
        model = STGraphTransformer(
            timesteps=timesteps,
            state_dim=spec.history_dim,
            cond_components=2,
            time_embed_dim=args.time_embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_mult=args.ffn_mult,
            dropout=0.1,
        )
        if A is not None:
            A_dev = A.to(device)
            N = A_dev.size(0)
            if N > 2000:
                edge_index_sp = A_dev.nonzero(as_tuple=False).t().contiguous()
                model.precompute_graph(edge_index_sp, N)
                print(f"[sparse graph conv] N={N}, E={edge_index_sp.size(1)}")
    elif args.backbone == "digress":
        model = DigressSTTransformer(
            timesteps=timesteps,
            state_dim=spec.history_dim,
            cond_components=2,
            time_embed_dim=args.time_embed_dim,
            hidden_dim=args.hidden_dim,
            edge_dim=args.edge_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_mult=args.ffn_mult,
            dropout=0.1,
        )
        if A is not None:
            A_dev = A.to(device)
            N = A_dev.size(0)
            edge_index_sp = A_dev.nonzero(as_tuple=False).t().contiguous()
            model.precompute_graph(edge_index_sp, N)
            print(f"[digress sparse GT] N={N}, E={edge_index_sp.size(1)}")
    else:
        model = HistoryInpaintGNN(
            timesteps=timesteps,
            state_dim=spec.history_dim,
            cond_components=2,
            time_embed_dim=args.time_embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        print(f"Loaded checkpoint from {args.checkpoint}")
    return model.to(device)


def run_eval(model, diffusion, batches, device, mode, timesteps):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in batches:
            Y_true = batch["Y"].to(device)
            A = batch["A"].to(device)
            if mode == "sir-times":
                final_snapshot = Y_true[:, :, -1, :]
                Y_times = sample_sir_times(
                    diffusion, model, A, final_snapshot, timesteps, num_steps=diffusion.K
                )
                Y_pred = decode_hitting_times(Y_times, timesteps)
            else:
                mask = torch.zeros_like(Y_true)
                mask[:, :, -1, :] = 1.0
                Y_obs = mask * Y_true
                Y_pred = sample_history(diffusion, model, A, Y_obs, mask)
            metrics = evaluate_history(Y_true, Y_pred)
            results.append(metrics)
            print(f"  sample metrics: {metrics}")
    return results


def main():
    args = parse_args()
    device = torch.device(args.device)
    spec = BENCHMARKS[args.benchmark]

    # Build evaluation batches
    if spec.graph_type == "ditto_real":
        # D3: evaluate on the single real history
        print(f"Loading real history for {spec.name} ...")
        real_ds = load_real_history_from_spec(spec, device=device, ditto_dir=args.ditto_dir)
        loader = DataLoader(real_ds, batch_size=1, shuffle=False)
        timesteps = spec.timesteps
        sample0 = real_ds[0]
        A_graph = sample0["A"]
        num_nodes = A_graph.size(0)
    else:
        # D1/D2: evaluate on synthetic samples
        dataset = make_dataset_from_spec(
            spec, num_samples=args.num_samples, device=device, ditto_dir=args.ditto_dir
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        timesteps = spec.timesteps
        A_graph = dataset.A
        num_nodes = A_graph.size(0)

    batches = list(loader)

    alphas = make_alpha_schedule(args.num_steps)
    diffusion = HistoryBetaDiffusion(eta=args.eta, alphas=alphas.to(device))
    model = build_model(spec, timesteps, args, A=A_graph)

    print(f"\nEvaluating {spec.name} ({len(batches)} sample(s), mode={args.mode}, backbone={args.backbone}) ...")
    results = run_eval(model, diffusion, batches, device, args.mode, timesteps)

    if results:
        agg = {k: sum(m[k] for m in results) / len(results) for k in results[0]}
        print(f"\n[{spec.name}] averaged over {len(results)} sample(s):")
        for k, v in agg.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
