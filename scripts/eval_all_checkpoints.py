#!/usr/bin/env python3
"""
Re-evaluate all existing checkpoints with current post-processing (sir_project=True).
Outputs a TSV table with F1 and NRMSE for each (benchmark, backbone) pair.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader

from graph_diffusion.benchmarks import (
    BENCHMARKS, make_dataset_from_spec, load_real_history_from_spec,
)
from graph_diffusion.beta_diffusion import HistoryBetaDiffusion
from graph_diffusion.model import HistoryInpaintGNN
from graph_diffusion.model_st import STGraphTransformer
from graph_diffusion.model_digress import DigressSTTransformer
from graph_diffusion.sampling import sample_history
from graph_diffusion.schedules import make_alpha_schedule
from graph_diffusion.metrics import evaluate_history

DITTO_DIR = "/home/lingjie7/KDD23-DITTO/input"
RUNS_DIR = "/home/lingjie7/graph_diffusion/runs"
DEVICE = torch.device("cuda")

# Evaluation settings
EVAL_SAMPLES = 5       # number of synthetic eval batches (D1/D2)
NUM_STEPS = 200        # diffusion sampling steps
ETA = 40.0
SEED = 42

# Checkpoint mapping: (benchmark, backbone) -> checkpoint filename
# Use the best available checkpoint for each pair
CHECKPOINTS = {}

# D1: prefer "-fixed" checkpoints
for graph in ["BA", "ER"]:
    for model in ["SI", "SIR"]:
        bm = f"D1-{graph}-{model}"
        for bb in ["inpaint", "st", "digress"]:
            ckpt_name = f"{bm}-{bb}-fixed.pt"
            ckpt_path = os.path.join(RUNS_DIR, ckpt_name)
            if os.path.exists(ckpt_path):
                CHECKPOINTS[(bm, bb)] = ckpt_path
            else:
                # fallback to non-fixed
                for fallback in [f"{bm}-{bb}.pt", f"{bm}.pt"]:
                    fp = os.path.join(RUNS_DIR, fallback)
                    if os.path.exists(fp):
                        CHECKPOINTS[(bm, bb)] = fp
                        break

# D2
for graph in ["Oregon2", "Prost"]:
    for model in ["SI", "SIR"]:
        bm = f"D2-{graph}-{model}"
        for bb in ["inpaint", "st", "digress"]:
            for suffix in [f"{bm}-{bb}.pt", f"{bm}.pt"]:
                fp = os.path.join(RUNS_DIR, suffix)
                if os.path.exists(fp):
                    CHECKPOINTS[(bm, bb)] = fp
                    break

# D3
for ds in ["BrFarmers", "Covid", "Hebrew", "Pol"]:
    bm = f"D3-{ds}"
    for bb in ["inpaint", "st", "digress"]:
        for suffix in [f"{bm}-{bb}.pt", f"{bm}.pt"]:
            fp = os.path.join(RUNS_DIR, suffix)
            if os.path.exists(fp):
                CHECKPOINTS[(bm, bb)] = fp
                break


def make_model(spec, backbone, state_dict):
    """Create model matching the checkpoint architecture."""
    T = spec.timesteps
    d_s = 3  # S, I, R
    cond_comp = 2  # Y_obs + mask

    if backbone == "inpaint":
        # Infer num_layers from state_dict
        layer_keys = [k for k in state_dict if k.startswith("layers.") and k.endswith(".weight")]
        num_layers = max(int(k.split(".")[1]) for k in layer_keys) + 1
        model = HistoryInpaintGNN(
            timesteps=T, state_dim=d_s, cond_components=cond_comp,
            time_embed_dim=16, hidden_dim=128, num_layers=num_layers,
        )
    elif backbone == "st":
        model = STGraphTransformer(
            timesteps=T, state_dim=d_s, cond_components=cond_comp,
            time_embed_dim=16, hidden_dim=128, num_layers=4,
            num_heads=4, ffn_mult=2, dropout=0.1,
        )
    elif backbone == "digress":
        model = DigressSTTransformer(
            timesteps=T, state_dim=d_s, cond_components=cond_comp,
            time_embed_dim=16, hidden_dim=128, num_layers=4,
            num_heads=4, ffn_mult=2, dropout=0.1, edge_dim=16,
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    model.load_state_dict(state_dict, strict=False)
    return model


def evaluate_checkpoint(bm_name, backbone, ckpt_path):
    """Evaluate a single checkpoint with sir_project=True."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    spec = BENCHMARKS[bm_name]
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model = make_model(spec, backbone, state_dict).to(DEVICE)
    model.eval()

    alphas = make_alpha_schedule(NUM_STEPS).to(DEVICE)
    diffusion = HistoryBetaDiffusion(eta=ETA, alphas=alphas)

    # For D3, evaluate on real history
    is_d3 = spec.graph_type == "ditto_real"

    results_pp = []    # with post-processing
    results_nopp = []  # without post-processing

    if is_d3:
        ds = load_real_history_from_spec(spec, device=DEVICE, ditto_dir=DITTO_DIR)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        max_batches = len(loader)
    else:
        ds = make_dataset_from_spec(spec, num_samples=EVAL_SAMPLES * 8,
                                     device=DEVICE, ditto_dir=DITTO_DIR)
        loader = DataLoader(ds, batch_size=8, shuffle=False,
                           generator=torch.Generator().manual_seed(SEED))
        max_batches = EVAL_SAMPLES

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            Y_true = batch["Y"].to(DEVICE)
            A = batch["A"].to(DEVICE)
            mask = torch.zeros_like(Y_true)
            mask[:, :, -1, :] = 1.0
            Y_obs = mask * Y_true

            # With post-processing
            Y_pred_pp = sample_history(diffusion, model, A, Y_obs, mask,
                                       num_steps=50, sir_project=True)
            results_pp.append(evaluate_history(Y_true, Y_pred_pp))

            # Without post-processing
            Y_pred_nopp = sample_history(diffusion, model, A, Y_obs, mask,
                                          num_steps=50, sir_project=False)
            results_nopp.append(evaluate_history(Y_true, Y_pred_nopp))

    def avg_results(results):
        return {k: sum(m[k] for m in results) / len(results) for k in results[0]}

    return avg_results(results_pp), avg_results(results_nopp)


def main():
    print(f"Evaluating {len(CHECKPOINTS)} checkpoints...")
    print(f"{'Benchmark':<22} {'Backbone':<10} {'F1(pp)':>8} {'NRMSE(pp)':>10} {'F1(no)':>8} {'NRMSE(no)':>10}  Checkpoint")
    print("-" * 100)

    all_results = {}

    # Sort by benchmark name for nice output
    for (bm, bb), ckpt in sorted(CHECKPOINTS.items()):
        try:
            t0 = time.time()
            res_pp, res_nopp = evaluate_checkpoint(bm, bb, ckpt)
            elapsed = time.time() - t0
            print(f"{bm:<22} {bb:<10} {res_pp['macro_f1']:>8.4f} {res_pp['nrmse']:>10.4f} "
                  f"{res_nopp['macro_f1']:>8.4f} {res_nopp['nrmse']:>10.4f}  "
                  f"{os.path.basename(ckpt)} ({elapsed:.0f}s)")
            all_results[(bm, bb)] = {"pp": res_pp, "nopp": res_nopp, "ckpt": ckpt}
            sys.stdout.flush()
        except Exception as e:
            print(f"{bm:<22} {bb:<10} ERROR: {e}")
            sys.stdout.flush()

    # Save results as JSON
    out = {}
    for (bm, bb), v in all_results.items():
        out[f"{bm}_{bb}"] = {
            "benchmark": bm, "backbone": bb,
            "f1_pp": v["pp"]["macro_f1"], "nrmse_pp": v["pp"]["nrmse"],
            "f1_nopp": v["nopp"]["macro_f1"], "nrmse_nopp": v["nopp"]["nrmse"],
            "checkpoint": v["ckpt"],
        }
    with open("runs/eval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to runs/eval_results.json")


if __name__ == "__main__":
    main()
