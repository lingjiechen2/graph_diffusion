"""
SIR/SI-aware post-processing for graph diffusion predictions.

The raw model output often lacks per-node spatial discriminability due to
GCN over-smoothing.  This module applies a graph-topology-based projection:
  1. Forces every trajectory to be a valid S→I(→R) sequence.
  2. Estimates per-node infection times using a cluster-density heuristic:

     SIR mode (R cluster exists):
     - R nodes: sorted by local R-density (fraction of R neighbours).
       High density = central in recovered cluster = infected early (t ∈ [0, T//2]).
     - I nodes: sorted by BFS distance to R cluster.
       Closer = currently spreading from R frontier (t ∈ [T//2, T-1]).

     SI mode (no R cluster):
     - I nodes: sorted by local I-density (fraction of I neighbours).
       High density = deeply embedded in infected cluster = infected early (t ∈ [0, T-1]).
       Low density = peripheral frontier node = infected late.

Both density heuristics exploit the same epidemic geometry principle:
nodes at the *core* of an infection cluster were exposed first, while
nodes at the *periphery* were reached last by the spreading front.
"""

import torch


@torch.no_grad()
def sir_bfs_project(
    Y_pred: torch.Tensor,
    Y_obs: torch.Tensor,
    A: torch.Tensor,
) -> torch.Tensor:
    """
    Project model predictions to valid SIR trajectories.

    Args:
        Y_pred: (B, N, T+1, d_s)  raw model output (continuous probabilities).
        Y_obs:  (B, N, T+1, d_s)  conditioning observation (non-zero only at t=T).
        A:      (N, N) or (B, N, N)  adjacency matrix (values in {0,1}).

    Returns:
        Y_proj: (B, N, T+1, d_s)  valid SIR history (one-hot), t=T exact.
    """
    B, N, T1, ds = Y_pred.shape
    T = T1 - 1
    device = Y_pred.device

    A_mat = (A[0] if A.dim() == 3 else A).cpu()  # (N, N) on CPU for fast BFS

    Y_proj = Y_pred.clone()

    for b in range(B):
        final = Y_obs[b, :, -1, :].argmax(-1)   # (N,) final labels

        s_nodes = (final == 0).nonzero(as_tuple=True)[0].tolist()
        i_nodes = (final == 1).nonzero(as_tuple=True)[0].tolist()
        r_nodes = (final == 2).nonzero(as_tuple=True)[0].tolist()

        # ------------------------------------------------------------------ #
        # S-at-T nodes: always S (SIR is monotone, so S at T ⇒ S at all t).  #
        # ------------------------------------------------------------------ #
        if s_nodes:
            s_idx = torch.tensor(s_nodes, device=device)
            Y_proj[b, s_idx] = 0.0
            Y_proj[b, s_idx, :, 0] = 1.0

        # ------------------------------------------------------------------ #
        # BFS distances from R cluster (needed for I-node timing).            #
        # ------------------------------------------------------------------ #
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

        # ------------------------------------------------------------------ #
        # R nodes: rank by local R-density to differentiate infection timing. #
        #   r_density[n] = fraction of n's neighbours that are also R-at-T.  #
        #   High density ⇒ central in epidemic ⇒ infected early.             #
        #   The top ~source_rate fraction gets t_inf=0 (seeds + first wave).  #
        # ------------------------------------------------------------------ #
        if r_nodes:
            r_set = set(r_nodes)
            r_density = {}
            for n in r_nodes:
                nbrs = A_mat[n].nonzero(as_tuple=True)[0].tolist()
                r_density[n] = (sum(1 for nb in nbrs if nb in r_set) / len(nbrs)
                                if nbrs else 0.0)
            # Sort: highest density first (earliest infected).
            sorted_r = sorted(r_nodes, key=lambda n: r_density[n], reverse=True)
            num_r = len(sorted_r)
            # Infection times span [0, T//2] — R nodes must recover before T.
            max_t_inf_r = max(1, T // 2)
            for rank, n in enumerate(sorted_r):
                t_inf = int(round(rank * max_t_inf_r / num_r))
                t_inf = max(0, min(t_inf, T - 2))
                # Recovery time: argmax P(R) after t_inf, fall back to midpoint.
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

        # ------------------------------------------------------------------ #
        # I nodes: sort by BFS distance to R cluster (closer ⇒ earlier).     #
        # If no R cluster (SI model), fall back to raw P(I,t=0) scores.       #
        # When R cluster exists: span [T//2, T-1] (SIR; haven't recovered).  #
        # When no R cluster: span [0, T-1] (SI; seeds can be at t=0).        #
        # ------------------------------------------------------------------ #
        if i_nodes:
            if r_nodes:
                # SIR: anchor to R cluster via BFS distance.
                sorted_i = sorted(i_nodes, key=lambda n: dist_r[n])
                t_min_i = max(1, T // 2)
                t_max_i = T - 1
            else:
                # SI: no R cluster — use local I-density heuristic (mirrors R-density).
                #   i_density[n] = fraction of n's neighbours that are I-at-T.
                #   High density ⇒ deeply embedded in I cluster ⇒ infected early.
                #   Low  density ⇒ peripheral frontier node  ⇒ infected late.
                i_set_local = set(i_nodes)
                i_density = {}
                for n in i_nodes:
                    nbrs = A_mat[n].nonzero(as_tuple=True)[0].tolist()
                    i_density[n] = (sum(1 for nb in nbrs if nb in i_set_local) / len(nbrs)
                                    if nbrs else 0.0)
                sorted_i = sorted(i_nodes, key=lambda n: i_density[n], reverse=True)
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

        # t=T is always the clamped observation — restore it exactly.
        Y_proj[b, :, T, :] = Y_obs[b, :, T, :]

    return Y_proj
