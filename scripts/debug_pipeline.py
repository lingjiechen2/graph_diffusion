#!/usr/bin/env python3
"""
Lightweight debugging script: runs a tiny pipeline end-to-end on a small ER graph.
Useful for sanity checking training/sampling/metrics without heavy benchmarks.
"""

import torch

from graph_diffusion.beta_diffusion import HistoryBetaDiffusion
from graph_diffusion.data import SyntheticHistoryDataset, random_graph
from graph_diffusion.metrics import evaluate_history
from graph_diffusion.model import SpaceTimeGNN
from graph_diffusion.sampling import sample_history
from graph_diffusion.schedules import make_alpha_schedule


def main():
    device = torch.device("cpu")
    N, T, d_s = 12, 4, 3
    A = random_graph(N, p=0.2, device=device)
    ds = SyntheticHistoryDataset(A, timesteps=T, num_samples=2, infection_rate=0.3, recovery_rate=0.1, source_rate=0.2)
    batch = ds[0]
    Y_true = batch["Y"].unsqueeze(0)

    alphas = make_alpha_schedule(10)
    diffusion = HistoryBetaDiffusion(eta=20.0, alphas=alphas)
    model = SpaceTimeGNN(history_dim=(T + 1) * d_s, hidden_dim=64, num_layers=2).to(device)

    mask = torch.zeros_like(Y_true)
    mask[:, :, -1, :] = 1.0
    Y_obs = mask * Y_true
    Y_pred = sample_history(diffusion, model, A, Y_obs, mask, num_steps=5)
    metrics = evaluate_history(Y_true, Y_pred)
    print("Debug metrics:", metrics)
    snapshot_match = torch.allclose(Y_pred[:, :, -1, :], Y_obs[:, :, -1, :], atol=1e-5)
    print("Observed snapshot match:", bool(snapshot_match))


if __name__ == "__main__":
    main()
