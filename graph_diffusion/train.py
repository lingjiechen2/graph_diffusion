import argparse
import os
from typing import Tuple

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

import torch
from torch.utils.data import DataLoader

from .beta_diffusion import HistoryBetaDiffusion
from .benchmarks import BENCHMARKS, BenchmarkSpec, load_real_history_from_spec, make_dataset_from_spec
from .data import SyntheticHistoryDataset, random_graph
from .metrics import evaluate_history
from .model import HistoryInpaintGNN, SpaceTimeGNN
from .model_st import STGraphTransformer
from .model_digress import DigressSTTransformer
from .sampling import sample_history
from .schedules import make_alpha_schedule
from .sir_times import (
    decode_hitting_times,
    encode_hitting_times,
    monotonic_penalty,
    project_monotonic_times,
    sample_sir_times,
)
from .sir_history_rules import sir_history_penalty


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
    p.add_argument("--mode", type=str, default="history", choices=["history", "sir-times"], help="Diffusion space: full history or SIR hitting times.")
    p.add_argument("--ditto-dir", type=str, default=None, help="Path to KDD23-DITTO input/ directory (required for D2/D3 benchmarks).")
    p.add_argument("--eval-real", action="store_true", default=False, help="For D3 benchmarks: evaluate on the single real history instead of synthetic.")
    p.add_argument("--backbone", type=str, default="inpaint", choices=["inpaint", "st", "digress"],
                   help="Denoiser backbone: inpaint=HistoryInpaintGNN (baseline), st=STGraphTransformer, digress=DigressSTTransformer.")
    p.add_argument("--edge-dim", type=int, default=16, help="Edge feature dimension (digress backbone only).")
    p.add_argument("--num-heads", type=int, default=4, help="Attention heads (st backbone only).")
    p.add_argument("--ffn-mult", type=int, default=2, help="FFN hidden multiplier (st backbone only).")
    p.add_argument("--mono-weight", type=float, default=0.0, help="Optional weight for monotonicity penalty (sir-times mode).")
    # --- history-mode conditioning + SIR-rule regularization ---
    p.add_argument("--cond-drop-prob", type=float, default=0.0, help="Classifier-free conditioning dropout prob (history mode).")
    p.add_argument("--time-embed-dim", type=int, default=16, help="True-time positional embedding dim (history mode).")
    p.add_argument("--sir-rule-weight", type=float, default=0.0, help="Weight for SIR transition penalties (history mode).")
    p.add_argument("--sir-mono-w", type=float, default=1.0, help="Relative weight for S non-increasing / R non-decreasing.")
    p.add_argument("--sir-dyn-w", type=float, default=1.0, help="Relative weight for mean-field neighbor-driven dynamics.")
    # --- wandb logging ---
    p.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project", type=str, default="graph-diffusion", help="W&B project name.")
    return p.parse_args()


def make_dataloader(args) -> Tuple[DataLoader, torch.Tensor, int, int]:
    device = torch.device(args.device)
    # DataLoader requires a CPU generator; model/data tensors still placed on `device` manually.
    g = torch.Generator().manual_seed(args.seed)
    if args.benchmark:
        spec: BenchmarkSpec = BENCHMARKS[args.benchmark]
        dataset = make_dataset_from_spec(spec, num_samples=256, device=device, ditto_dir=args.ditto_dir)
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

    # Initialise W&B run (opt-in via --wandb flag)
    use_wandb = args.wandb and _WANDB_AVAILABLE
    if use_wandb:
        run_name = f"{args.benchmark or 'custom'}-{args.backbone}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    loader, A_full, history_dim, timesteps = make_dataloader(args)
    alphas = make_alpha_schedule(args.num_steps)
    diffusion = HistoryBetaDiffusion(eta=args.eta, alphas=alphas.to(device))

    state_dim = history_dim
    if args.mode == "sir-times":
        model = SpaceTimeGNN(feature_dim=2, hidden_dim=128, num_layers=3).to(device)
    elif args.backbone == "st":
        model = STGraphTransformer(
            timesteps=timesteps,
            state_dim=state_dim,
            cond_components=2,
            time_embed_dim=args.time_embed_dim,
            hidden_dim=128,
            num_layers=4,
            num_heads=args.num_heads,
            ffn_mult=args.ffn_mult,
            dropout=0.1,
        ).to(device)
        # For large sparse graphs, precompute D^{-1}(A+I) edge weights so that
        # graph conv runs in O(E·H) instead of O(N²·H).
        A_full_dev = A_full.to(device)
        N_nodes = A_full_dev.size(0)
        if N_nodes > 2000:
            edge_index_sp = A_full_dev.nonzero(as_tuple=False).t().contiguous()
            model.precompute_graph(edge_index_sp, N_nodes)
            print(f"[sparse graph conv enabled] N={N_nodes}, E={edge_index_sp.size(1)}")
        del A_full_dev
    elif args.backbone == "digress":
        model = DigressSTTransformer(
            timesteps=timesteps,
            state_dim=state_dim,
            cond_components=2,
            time_embed_dim=args.time_embed_dim,
            hidden_dim=128,
            edge_dim=args.edge_dim,
            num_layers=4,
            num_heads=args.num_heads,
            ffn_mult=args.ffn_mult,
            dropout=0.1,
        ).to(device)
        # DiGress always uses sparse edge-indexed graph conv (requires edge features)
        A_full_dev = A_full.to(device)
        N_nodes = A_full_dev.size(0)
        edge_index_sp = A_full_dev.nonzero(as_tuple=False).t().contiguous()
        model.precompute_graph(edge_index_sp, N_nodes)
        print(f"[digress sparse GT] N={N_nodes}, E={edge_index_sp.size(1)}")
        del A_full_dev
    else:
        # history mode: conditional + time-aware + simplex output
        model = HistoryInpaintGNN(
            timesteps=timesteps,
            state_dim=state_dim,
            cond_components=2,  # [Y_obs, mask] in addition to Z_k
            time_embed_dim=args.time_embed_dim,
            hidden_dim=128,
            num_layers=3,
        ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            Y = batch["Y"].to(device)  # (B,N,T+1,d_s)
            A = batch["A"].to(device)
            opt.zero_grad()
            if args.mode == "sir-times":
                Y_enc = encode_hitting_times(Y)
                loss, metrics = diffusion.klub_loss(
                    model,
                    A,
                    Y_enc,
                    project_fn=project_monotonic_times,
                    penalty_fn=monotonic_penalty,
                    penalty_weight=args.mono_weight,
                )
            else:
                # Condition on the final snapshot DURING TRAINING (not only at sampling time).
                mask = torch.zeros_like(Y)
                mask[:, :, -1, :] = 1.0
                Y_obs = mask * Y

                # Optional: explicitly inject SIR transition knowledge as a regularizer.
                if args.model_type.lower() == "sir" and state_dim == 3 and args.sir_rule_weight > 0:
                    penalty_fn = lambda Y_hat: sir_history_penalty(
                        Y_hat,
                        A,
                        beta=args.infection_rate,
                        gamma=args.recovery_rate,
                        w_mono=args.sir_mono_w,
                        w_dyn=args.sir_dyn_w,
                    )
                    penalty_weight = args.sir_rule_weight
                else:
                    penalty_fn = None
                    penalty_weight = 0.0

                loss, metrics = diffusion.klub_loss(
                    model,
                    A,
                    Y,
                    Y_obs=Y_obs,
                    mask=mask,
                    cond_drop_prob=args.cond_drop_prob,
                    penalty_fn=penalty_fn,
                    penalty_weight=penalty_weight,
                )
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total_loss += loss.item() * Y.size(0)
        avg = total_loss / len(loader.dataset)
        print(f"epoch {epoch+1} | loss {avg:.4f} | k {metrics['k']} kl {metrics['kl']:.4f}", flush=True)
        if use_wandb:
            wandb.log({"train/loss": avg, "train/kl": metrics["kl"], "train/k": metrics["k"]}, step=epoch + 1)

    # decide save path if not provided
    if args.save_path is None:
        import os

        os.makedirs("runs", exist_ok=True)
        args.save_path = os.path.join("runs", f"{args.benchmark or 'model'}.pt")
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved checkpoint to {args.save_path}")

    # optional quick evaluation on a few batches after training
    if args.eval_samples > 0:
        model.eval()
        results = []
        # For D3 with --eval-real: evaluate on the single real history.
        eval_spec = BENCHMARKS.get(args.benchmark) if args.benchmark else None
        if args.eval_real and eval_spec is not None and eval_spec.graph_type == "ditto_real":
            from torch.utils.data import DataLoader as _DL
            real_ds = load_real_history_from_spec(eval_spec, device=device, ditto_dir=args.ditto_dir)
            eval_loader = _DL(real_ds, batch_size=1, shuffle=False)
            eval_batches = list(eval_loader)  # just 1 sample
        else:
            eval_batches = []
            for i, batch in enumerate(loader):
                if i >= args.eval_samples:
                    break
                eval_batches.append({k: v.to(device) for k, v in batch.items()})

        with torch.no_grad():
            for i, batch in enumerate(eval_batches):
                Y_true = batch["Y"].to(device)
                A = batch["A"].to(device)
                if args.mode == "sir-times":
                    final_snapshot = Y_true[:, :, -1, :]
                    Y_times = sample_sir_times(diffusion, model, A, final_snapshot, timesteps, num_steps=args.num_steps)
                    Y_pred = decode_hitting_times(Y_times, timesteps)
                else:
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
            if use_wandb:
                wandb.log({f"eval/{k}": v for k, v in agg.items()})

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train_loop()
