#!/usr/bin/env python3
"""
Experiment: Hitting-time diffusion mode for SI benchmarks.

Instead of predicting full (B,N,T+1,3) state matrix with HistoryInpaintGNN,
predict (B,N,2) hitting times directly with SpaceTimeGNN.

Key advantages:
- Output is 2D instead of 33D (for T=10), much simpler
- No T+1 flatten → GCN bottleneck
- Built-in monotonicity constraints
- Final-state constraints from observed snapshot
"""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader

from graph_diffusion.benchmarks import BENCHMARKS, make_dataset_from_spec
from graph_diffusion.beta_diffusion import HistoryBetaDiffusion
from graph_diffusion.model import SpaceTimeGNN
from graph_diffusion.schedules import make_alpha_schedule
from graph_diffusion.metrics import evaluate_history
from graph_diffusion.sir_times import (
    encode_hitting_times,
    decode_hitting_times,
    sample_sir_times,
    monotonic_penalty,
)

DITTO_DIR = "/home/lingjie7/KDD23-DITTO/input"
RUNS_DIR = "/home/lingjie7/graph_diffusion/runs"
SEED = 42


def train_and_eval(bm_name, device, epochs=200, lr=1e-3, batch_size=8,
                   num_samples=256, hidden_dim=128, num_layers=3,
                   mono_weight=0.1):
    """Train SpaceTimeGNN in hitting-time mode and evaluate."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    spec = BENCHMARKS[bm_name]
    T = spec.timesteps

    ds = make_dataset_from_spec(spec, num_samples=num_samples,
                                 device=device, ditto_dir=DITTO_DIR)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(SEED))

    alphas = make_alpha_schedule(200).to(device)
    diffusion = HistoryBetaDiffusion(eta=40, alphas=alphas)

    model = SpaceTimeGNN(
        feature_dim=2, hidden_dim=hidden_dim, num_layers=num_layers,
    ).to(device)
    print(f"  SpaceTimeGNN params: {sum(p.numel() for p in model.parameters()):,}")
    sys.stdout.flush()

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    t0_train = time.time()
    for epoch in range(epochs):
        total_loss = 0.0
        total_mono = 0.0
        for batch in loader:
            Y = batch["Y"].to(device)
            A = batch["A"].to(device)
            Y_enc = encode_hitting_times(Y)  # (B, N, 2)

            opt.zero_grad()
            loss, metrics = diffusion.klub_loss(model, A, Y_enc)
            mono = monotonic_penalty(
                model(A, Y_enc, torch.zeros(Y.size(0), device=device, dtype=torch.long))
            )
            total = loss + mono_weight * mono
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * Y.size(0)
            total_mono += mono.item() * Y.size(0)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = total_loss / num_samples
            avg_mono = total_mono / num_samples
            print(f"  epoch {epoch+1:>3} | loss {avg_loss:.4f} | mono {avg_mono:.6f}")
            sys.stdout.flush()

    train_time = time.time() - t0_train
    print(f"  Training done in {train_time:.0f}s")
    sys.stdout.flush()

    # Evaluate
    model.eval()
    eval_ds = make_dataset_from_spec(spec, num_samples=40,
                                      device=device, ditto_dir=DITTO_DIR)
    eval_loader = DataLoader(eval_ds, batch_size=8, shuffle=False,
                            generator=torch.Generator().manual_seed(SEED + 1))

    results = []
    t0_preds = []
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= 5:
                break
            Y_true = batch["Y"].to(device)
            A = batch["A"].to(device)
            final_snapshot = Y_true[:, :, -1, :]

            Y_times = sample_sir_times(diffusion, model, A, final_snapshot, T,
                                        num_steps=50)
            Y_pred = decode_hitting_times(Y_times, T)

            results.append(evaluate_history(Y_true, Y_pred))
            for bb in range(Y_true.shape[0]):
                t0_preds.append(
                    (Y_pred[bb, :, 0, :].argmax(-1) == 1).sum().item()
                )

            if i == 0:
                # Print per-time prediction for first sample
                print(f"\n  {'t':>3} | {'true I':>7} | {'pred I':>7}")
                print("  " + "-" * 25)
                for t in range(T + 1):
                    ti = (Y_true[0, :, t, :].argmax(-1) == 1).sum().item()
                    pi = (Y_pred[0, :, t, :].argmax(-1) == 1).sum().item()
                    print(f"  {t:>3} | {ti:>7} | {pi:>7}")

    agg = {k: sum(m[k] for m in results) / len(results) for k in results[0]}
    t0_pred_avg = np.mean(t0_preds)

    print(f"\n  RESULT: F1={agg['macro_f1']:.4f}  NRMSE={agg['nrmse']:.4f}  "
          f"t0_pred={t0_pred_avg:.1f}")
    sys.stdout.flush()

    return model, agg, t0_pred_avg


def main():
    gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    device = torch.device("cuda")

    benchmarks = [
        "D1-BA-SI",
        "D1-ER-SI",
        "D1-BA-SIR",
        "D1-ER-SIR",
    ]

    print("=" * 80)
    print("HITTING-TIME DIFFUSION MODE EXPERIMENT")
    print(f"GPU: {gpu}")
    print("=" * 80)
    sys.stdout.flush()

    all_results = {}
    for bm in benchmarks:
        print(f"\n{'='*60}")
        print(f"  {bm}")
        print(f"{'='*60}")

        model, result, t0_pred = train_and_eval(
            bm, device,
            epochs=200,
            lr=1e-3,
            batch_size=8,
            num_samples=256,
            hidden_dim=128,
            num_layers=3,
            mono_weight=0.1,
        )

        # Save checkpoint
        ckpt_path = f"{RUNS_DIR}/{bm}-hitting-times.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved to {ckpt_path}")
        sys.stdout.flush()

        all_results[bm] = {
            "f1": result["macro_f1"],
            "nrmse": result["nrmse"],
            "t0_pred": t0_pred,
        }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Benchmark':<22} {'F1':>8} {'NRMSE':>8} {'t0_pred':>8}")
    print("-" * 50)
    for bm, r in all_results.items():
        print(f"{bm:<22} {r['f1']:>8.4f} {r['nrmse']:>8.4f} {r['t0_pred']:>8.1f}")

    print("\nDone!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
