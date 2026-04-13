#!/usr/bin/env python3
"""
Experiment: Improved SI post-processing using BFS from S nodes.

Instead of I-density (weak when ~95% nodes are I), use BFS from the sparse
remaining S nodes at T to rank I nodes by infection order:
  - Far from S = deep inside infection cluster = infected early
  - Close to S = near frontier = infected late

Evaluates existing Inpaint checkpoints on all D1+D2 SI benchmarks.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader

from graph_diffusion.benchmarks import BENCHMARKS, make_dataset_from_spec
from graph_diffusion.beta_diffusion import HistoryBetaDiffusion
from graph_diffusion.sampling import sample_history
from graph_diffusion.schedules import make_alpha_schedule
from graph_diffusion.metrics import evaluate_history
from graph_diffusion import sir_postprocess


# Old (flattened) HistoryInpaintGNN compatible with existing checkpoints.
class OldHistoryInpaintGNN(nn.Module):
    def __init__(self, timesteps, state_dim, cond_components=2,
                 time_embed_dim=16, hidden_dim=128, num_layers=3,
                 dropout=0.0, max_steps=1024):
        super().__init__()
        self.timesteps = timesteps
        self.T_plus_1 = timesteps + 1
        self.state_dim = state_dim
        self.in_state_dim = (1 + cond_components) * state_dim
        self.time_embed = nn.Embedding(max_steps, time_embed_dim)
        per_time_dim = self.in_state_dim + time_embed_dim
        flat_in_dim = self.T_plus_1 * per_time_dim
        flat_out_dim = self.T_plus_1 * state_dim
        self.encoder = nn.Linear(flat_in_dim, hidden_dim)
        self.step_embed = nn.Embedding(max_steps, hidden_dim)
        self.layers = nn.ModuleList(
            nn.ModuleList([nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)])
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, flat_out_dim),
        )

    def _normalize_adj(self, A):
        if A.dim() == 2: A = A.unsqueeze(0)
        I = torch.eye(A.size(-1), device=A.device).expand_as(A)
        A_hat = A + I
        deg = A_hat.sum(-1, keepdim=True).clamp(min=1.0)
        return A_hat / deg

    def forward(self, A, X_k, k_index):
        B, N, T_plus_1, Cin = X_k.shape
        t_idx = torch.arange(T_plus_1, device=X_k.device, dtype=torch.long)
        t_emb = self.time_embed(t_idx).view(1, 1, T_plus_1, -1).expand(B, N, T_plus_1, -1)
        X = torch.cat([X_k, t_emb], dim=-1)
        flat = X.reshape(B, N, -1)
        h = self.encoder(flat)
        h_skip = h
        A_norm = self._normalize_adj(A)
        if A_norm.size(0) == 1 and B > 1: A_norm = A_norm.expand(B, -1, -1)
        step_emb = self.step_embed(k_index)[:, None, :]
        for self_w, neigh_w in self.layers:
            m = torch.bmm(A_norm, h)
            h = self_w(h) + neigh_w(m) + step_emb
            h = torch.relu(h)
            h = self.dropout(h)
        h = h + h_skip
        logits = self.decoder(h).view(B, N, T_plus_1, self.state_dim)
        return torch.softmax(logits, dim=-1)

DITTO_DIR = "/home/lingjie7/KDD23-DITTO/input"
RUNS_DIR = "/home/lingjie7/graph_diffusion/runs"
SEED = 42


def sir_bfs_project_v2(Y_pred, Y_obs, A):
    """
    Improved post-processing for SI: BFS from S nodes instead of I-density.
    For SIR: same as original (R-density + BFS from R).
    """
    B, N, T1, ds = Y_pred.shape
    T = T1 - 1
    device = Y_pred.device
    A_mat = (A[0] if A.dim() == 3 else A).cpu()

    Y_proj = Y_pred.clone()

    for b in range(B):
        final = Y_obs[b, :, -1, :].argmax(-1)
        s_nodes = (final == 0).nonzero(as_tuple=True)[0].tolist()
        i_nodes = (final == 1).nonzero(as_tuple=True)[0].tolist()
        r_nodes = (final == 2).nonzero(as_tuple=True)[0].tolist()

        # S-at-T: always S
        if s_nodes:
            s_idx = torch.tensor(s_nodes, device=device)
            Y_proj[b, s_idx] = 0.0
            Y_proj[b, s_idx, :, 0] = 1.0

        # --- R nodes: same as original (R-density heuristic) ---
        dist_r = [T + 2] * N
        for n in r_nodes:
            dist_r[n] = 0
        frontier = list(r_nodes)
        d = 1
        while frontier and d <= T:
            nxt = []
            for fn in frontier:
                for nb in A_mat[fn].nonzero(as_tuple=True)[0].tolist():
                    if dist_r[nb] > d:
                        dist_r[nb] = d
                        nxt.append(nb)
            frontier = nxt
            d += 1

        if r_nodes:
            r_set = set(r_nodes)
            r_density = {}
            for n in r_nodes:
                nbrs = A_mat[n].nonzero(as_tuple=True)[0].tolist()
                r_density[n] = (sum(1 for nb in nbrs if nb in r_set) / len(nbrs)
                                if nbrs else 0.0)
            sorted_r = sorted(r_nodes, key=lambda n: r_density[n], reverse=True)
            num_r = len(sorted_r)
            max_t_inf_r = max(1, T // 2)
            for rank, n in enumerate(sorted_r):
                t_inf = int(round(rank * max_t_inf_r / num_r))
                t_inf = max(0, min(t_inf, T - 2))
                p_r_slice = Y_pred[b, n, t_inf + 1:T, 2]
                if p_r_slice.numel() > 0 and float(p_r_slice.max()) > 1e-3:
                    t_rec = int(p_r_slice.argmax().item()) + t_inf + 1
                else:
                    gap = T - t_inf
                    t_rec = t_inf + max(1, int(gap * 0.5))
                t_rec = max(t_rec, t_inf + 1)
                t_rec = min(t_rec, T - 1)
                Y_proj[b, n] = 0.0
                if t_inf > 0:
                    Y_proj[b, n, :t_inf, 0] = 1.0
                Y_proj[b, n, t_inf:t_rec, 1] = 1.0
                Y_proj[b, n, t_rec:T, 2] = 1.0

        # --- I nodes ---
        if i_nodes:
            if r_nodes:
                # SIR: same as original
                sorted_i = sorted(i_nodes, key=lambda n: dist_r[n])
                t_min_i = max(1, T // 2)
                t_max_i = T - 1
            else:
                # *** SI: BFS from S nodes (NEW) ***
                # Remaining S nodes at T mark the infection frontier.
                # BFS distance from S: far = infected early, close = infected late.
                dist_s = [N + 1] * N  # large default
                bfs_frontier = list(s_nodes)
                for sn in s_nodes:
                    dist_s[sn] = 0
                dd = 1
                while bfs_frontier:
                    nxt = []
                    for fn in bfs_frontier:
                        for nb in A_mat[fn].nonzero(as_tuple=True)[0].tolist():
                            if dist_s[nb] > dd:
                                dist_s[nb] = dd
                                nxt.append(nb)
                    bfs_frontier = nxt
                    dd += 1

                # Sort: farthest from S first (= infected earliest)
                sorted_i = sorted(i_nodes, key=lambda n: dist_s[n], reverse=True)
                t_min_i = 0
                t_max_i = T - 1

            num_i = len(sorted_i)
            for rank, n in enumerate(sorted_i):
                if num_i == 1:
                    t_inf = t_min_i
                else:
                    t_inf = t_min_i + int(round(rank * (t_max_i - t_min_i) / (num_i - 1)))
                t_inf = max(0, min(t_inf, T - 1))
                Y_proj[b, n] = 0.0
                if t_inf > 0:
                    Y_proj[b, n, :t_inf, 0] = 1.0
                Y_proj[b, n, t_inf:T, 1] = 1.0

        Y_proj[b, :, T, :] = Y_obs[b, :, T, :]

    return Y_proj


def evaluate_with_postprocess(bm_name, ckpt_path, pp_fn, device):
    """Evaluate a checkpoint with a given post-processing function."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    spec = BENCHMARKS[bm_name]
    state_dict = torch.load(ckpt_path, map_location="cpu")

    # Infer num_layers from checkpoint
    layer_keys = [k for k in state_dict if k.startswith("layers.") and k.endswith(".weight")]
    num_layers = max(int(k.split(".")[1]) for k in layer_keys) + 1

    model = OldHistoryInpaintGNN(
        timesteps=spec.timesteps, state_dim=3, cond_components=2,
        time_embed_dim=16, hidden_dim=128, num_layers=num_layers,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    alphas = make_alpha_schedule(200).to(device)
    diffusion = HistoryBetaDiffusion(eta=40, alphas=alphas)

    ds = make_dataset_from_spec(spec, num_samples=40, device=device, ditto_dir=DITTO_DIR)
    loader = DataLoader(ds, batch_size=8, shuffle=False,
                       generator=torch.Generator().manual_seed(SEED))

    results = []
    t0_preds = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 5:
                break
            Y_true = batch["Y"].to(device)
            A = batch["A"].to(device)
            mask = torch.zeros_like(Y_true)
            mask[:, :, -1, :] = 1.0
            Y_obs = mask * Y_true

            # Sample WITHOUT post-processing
            Y_raw = sample_history(diffusion, model, A, Y_obs, mask,
                                   num_steps=50, sir_project=False)
            # Apply custom post-processing
            Y_pred = pp_fn(Y_raw, Y_obs, A)

            results.append(evaluate_history(Y_true, Y_pred))
            for bb in range(Y_true.shape[0]):
                t0_preds.append((Y_pred[bb, :, 0, :].argmax(-1) == 1).sum().item())

    agg = {k: sum(m[k] for m in results) / len(results) for k in results[0]}
    agg["t0_pred_avg"] = np.mean(t0_preds)
    return agg


def main():
    gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    device = torch.device("cuda")

    # SI benchmarks to evaluate
    si_benchmarks = [
        ("D1-BA-SI", f"{RUNS_DIR}/D1-BA-SI-inpaint-fixed.pt"),
        ("D1-ER-SI", f"{RUNS_DIR}/D1-ER-SI-inpaint-fixed.pt"),
        ("D2-Oregon2-SI", f"{RUNS_DIR}/D2-Oregon2-SI-inpaint.pt"),
        ("D2-Prost-SI", f"{RUNS_DIR}/D2-Prost-SI-inpaint.pt"),
    ]

    print("=" * 80)
    print("SI POST-PROCESSING COMPARISON")
    print(f"GPU: {gpu}")
    print("=" * 80)
    print(f"\n{'Benchmark':<22} {'Method':<16} {'F1':>8} {'NRMSE':>8} {'t0_pred':>8}")
    print("-" * 70)
    sys.stdout.flush()

    for bm, ckpt in si_benchmarks:
        if not os.path.exists(ckpt):
            print(f"{bm:<22} SKIP (no checkpoint)")
            continue

        # 1. No post-processing
        t0 = time.time()
        res_none = evaluate_with_postprocess(
            bm, ckpt,
            pp_fn=lambda y, yo, a: y,  # identity
            device=device,
        )
        t1 = time.time()
        print(f"{bm:<22} {'none':<16} {res_none['macro_f1']:>8.4f} {res_none['nrmse']:>8.4f} {res_none['t0_pred_avg']:>8.1f}  ({t1-t0:.0f}s)")
        sys.stdout.flush()

        # 2. Original I-density
        t0 = time.time()
        res_orig = evaluate_with_postprocess(
            bm, ckpt,
            pp_fn=sir_postprocess.sir_bfs_project,
            device=device,
        )
        t1 = time.time()
        print(f"{'':<22} {'I-density':<16} {res_orig['macro_f1']:>8.4f} {res_orig['nrmse']:>8.4f} {res_orig['t0_pred_avg']:>8.1f}  ({t1-t0:.0f}s)")
        sys.stdout.flush()

        # 3. New BFS-from-S
        t0 = time.time()
        res_new = evaluate_with_postprocess(
            bm, ckpt,
            pp_fn=sir_bfs_project_v2,
            device=device,
        )
        t1 = time.time()
        print(f"{'':<22} {'BFS-from-S':<16} {res_new['macro_f1']:>8.4f} {res_new['nrmse']:>8.4f} {res_new['t0_pred_avg']:>8.1f}  ({t1-t0:.0f}s)")
        sys.stdout.flush()

        # Delta
        delta_f1 = res_new['macro_f1'] - max(res_none['macro_f1'], res_orig['macro_f1'])
        best_prev = "none" if res_none['macro_f1'] > res_orig['macro_f1'] else "I-density"
        print(f"{'':<22} {'delta vs '+best_prev:<16} {delta_f1:>+8.4f}")
        print()
        sys.stdout.flush()

    print("Done!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
