import argparse
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from .beta_diffusion import HistoryBetaDiffusion
from .benchmarks import BENCHMARKS, BenchmarkSpec, make_dataset_from_spec
from .data import SyntheticHistoryDataset, random_graph
from .metrics import evaluate_history
from .model import SpaceTimeGNN
from .sampling import sample_history
from .schedules import make_alpha_schedule


def parse_args():
    p = argparse.ArgumentParser(description="Train Graph Beta Diffusion reverse model on synthetic histories.")
    p.add_argument("--num-nodes", type=int, default=16)
    p.add_argument("--graph-p", type=float, default=0.15)
    p.add_argument("--timesteps", type=int, default=5, help="Epidemic time horizon T.")
    p.add_argument("--history-dim", type=int, default=3, help="State dimension d_s (1 for binary, >1 for one-hot).")
    p.add_argument("--num-steps", type=int, default=200, help="Diffusion steps K.")
    p.add_argument("--eta", type=float, default=40.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--infection-rate", type=float, default=0.3)
    p.add_argument("--recovery-rate", type=float, default=0.1)
    p.add_argument("--source-rate", type=float, default=0.05, help="Initial infected fraction.")
    p.add_argument("--model-type", type=str, default="sir", choices=["si", "sir"])
    p.add_argument("--benchmark", type=str, default=None, help="Benchmark key (e.g., D1-BA-SIR). Overrides graph args.")
    p.add_argument("--edge-path", type=str, default=None, help="Edge list path for external graphs (D2).")
    p.add_argument("--save-path", type=str, default=None, help="Optional path to save model checkpoint.")
    p.add_argument("--seed", type=int, default=42, help="Global random seed.")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip norm (0 to disable).")
    p.add_argument("--eval-samples", type=int, default=0, help="If >0, run sampling/eval on this many batches after training.")
    return p.parse_args()


def make_dataloader(args) -> Tuple[DataLoader, torch.Tensor, int, int]:
    device = torch.device(args.device)
    # DataLoader requires a CPU generator; model/data tensors still placed on `device` manually.
    g = torch.Generator().manual_seed(args.seed)
    if args.benchmark:
        spec: BenchmarkSpec = BENCHMARKS[args.benchmark]
        dataset = make_dataset_from_spec(spec, num_samples=256, device=device, edge_path=args.edge_path)
        A = dataset.A
        history_dim = spec.history_dim
        timesteps = spec.timesteps
    else:
        A = random_graph(args.num_nodes, args.graph_p, device=device)
        dataset = SyntheticHistoryDataset(
            A,
            args.timesteps,
            num_samples=256,
            infection_rate=args.infection_rate,
            recovery_rate=args.recovery_rate,
            source_rate=args.source_rate,
            model=args.model_type,
        )
        history_dim = args.history_dim
        timesteps = args.timesteps
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=g)
    return loader, A, history_dim, timesteps


def train_loop():
    args = parse_args()
    # set global seeds for reproducibility
    import random
    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    loader, A_full, history_dim, timesteps = make_dataloader(args)
    alphas = make_alpha_schedule(args.num_steps)
    diffusion = HistoryBetaDiffusion(eta=args.eta, alphas=alphas.to(device))

    model = SpaceTimeGNN(history_dim=(timesteps + 1) * history_dim, hidden_dim=128, num_layers=3).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            Y = batch["Y"].to(device)  # (B,N,T+1,d_s)
            A = batch["A"].to(device)
            opt.zero_grad()
            loss, metrics = diffusion.klub_loss(model, A, Y)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total_loss += loss.item() * Y.size(0)
        avg = total_loss / len(loader.dataset)
        print(f"epoch {epoch+1} | loss {avg:.4f} | k {metrics['k']} kl {metrics['kl']:.4f}")

    # decide save path if not provided
    if args.save_path is None:
        import os

        os.makedirs("runs", exist_ok=True)
        args.save_path = os.path.join("runs", f"{args.benchmark or 'model'}.pt")
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved checkpoint to {args.save_path}")

    # optional quick evaluation on a few batches from the same generator
    if args.eval_samples > 0:
        model.eval()
        results = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.eval_samples:
                    break
                Y_true = batch["Y"].to(device)
                A = batch["A"].to(device)
                mask = torch.zeros_like(Y_true)
                mask[:, :, -1, :] = 1.0
                Y_obs = mask * Y_true
                Y_pred = sample_history(diffusion, model, A, Y_obs, mask)
                metrics_eval = evaluate_history(Y_true, Y_pred)
                results.append(metrics_eval)
                print(f"[eval sample {i}] {metrics_eval}")
        if results:
            agg = {k: sum(m[k] for m in results) / len(results) for k in results[0]}
            print(f"[eval avg over {len(results)}] {agg}")


if __name__ == "__main__":
    train_loop()
