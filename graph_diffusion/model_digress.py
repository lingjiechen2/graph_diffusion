"""
DiGress-style Spatiotemporal Graph Transformer for graph diffusion history denoising.

Extends STGraphTransformer by replacing the simple GCN graph block with a
sparse, FiLM edge-conditioned attention block inspired by DiGress
(Vignac et al., ICLR 2023).

Key improvements over STGraphTransformer:
  1. Edge features: structural priors [1/deg_dst, deg_src_norm, deg_dst_norm]
     initialised from graph topology, updated per layer via an MLP.
  2. Graph block: FiLM-modulated sparse attention replaces degree-normalised GCN.
     score(dst←src) = Q[dst]·K[src]/√d · (e_mul(e_ij)+1) + e_add(e_ij)
  3. Time re-injection: epidemic-time positional embedding re-added before every
     graph block so each spatial attention call knows which time step it handles.

Architecture per layer:
  (1) Temporal MHA — unchanged from STGraphTransformer
  (2) Temporal FFN — unchanged
  (3) Time re-injection — broadcast (T+1, H) → h
  (4) DiGress GT graph block (sparse):
        node update: h[dst] ← LayerNorm(h[dst] + out_proj(Σ attn·V[src]))
        edge update: e_ij   ← LayerNorm(e_ij   + MLP([h_mean[dst], h_mean[src], e_ij]))

precompute_graph() MUST be called before forward() for every graph.
No dense fallback — sparse is always used (works for any graph size).
"""

import math

import torch
from torch import nn


class DigressSTTransformer(nn.Module):
    """
    Drop-in replacement for STGraphTransformer with DiGress GT graph blocks.

    Input contract (same as STGraphTransformer / HistoryInpaintGNN):
        A       : (B, N, N) or (N, N)          adjacency  (used only to detect device; graph
                                                            structure comes from precompute_graph)
        X_k     : (B, N, T+1, in_state_dim)    noisy history + conditioning
        k_index : (B,)                          diffusion step indices

    Output: (B, N, T+1, state_dim) softmax probabilities.
    """

    def __init__(
        self,
        timesteps: int,
        state_dim: int,
        cond_components: int = 2,
        time_embed_dim: int = 16,
        hidden_dim: int = 128,
        edge_dim: int = 16,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_mult: int = 2,
        dropout: float = 0.1,
        max_steps: int = 1024,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.T_plus_1   = timesteps + 1
        self.state_dim  = state_dim
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.edge_dim   = edge_dim
        self.in_state_dim = (1 + cond_components) * state_dim

        H, de, nh = hidden_dim, edge_dim, num_heads

        # ------ Embedding layers ------
        # Epidemic-time positional embedding (t = 0 … T)
        self.time_embed    = nn.Embedding(self.T_plus_1, time_embed_dim)
        # Diffusion-step embedding (k = 0 … K-1)
        self.step_embed    = nn.Embedding(max_steps, H)
        # Re-injection: maps time_embed_dim → H, added to h before every graph block
        self.time_reproject = nn.Linear(time_embed_dim, H)

        # ------ Input projection ------
        self.input_proj = nn.Linear(self.in_state_dim + time_embed_dim, H)
        self.input_norm = nn.LayerNorm(H)

        # ------ Edge feature initialisation ------
        # Raw edge features: [1/deg_dst, src_deg_norm, dst_deg_norm]  →  edge_dim
        self.edge_proj = nn.Linear(3, de)

        # ------ Layers ------
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                # --- Temporal block (identical to STGraphTransformer) ---
                "time_attn": nn.MultiheadAttention(
                    H, nh, dropout=dropout, batch_first=True
                ),
                "time_norm": nn.LayerNorm(H),
                "time_ffn": nn.Sequential(
                    nn.Linear(H, H * ffn_mult), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(H * ffn_mult, H), nn.Dropout(dropout),
                ),
                "ffn_norm": nn.LayerNorm(H),

                # --- DiGress GT graph block ---
                # Node projections (no bias in Q/K following common practice)
                "Q":        nn.Linear(H, H, bias=False),
                "K":        nn.Linear(H, H, bias=False),
                "V":        nn.Linear(H, H, bias=False),
                "out_proj": nn.Linear(H, H),
                # FiLM edge conditioning: e_ij → per-head multiplicative & additive bias
                "e_mul":    nn.Linear(de, nh),
                "e_add":    nn.Linear(de, nh),
                # Edge update MLP: [h_dst, h_src, e_ij] → new e_ij
                "edge_mlp": nn.Sequential(
                    nn.Linear(2 * H + de, de * 4), nn.GELU(),
                    nn.Linear(de * 4, de), nn.Dropout(dropout),
                ),
                "graph_norm": nn.LayerNorm(H),
                "edge_norm":  nn.LayerNorm(de),
            }))

        self.dropout    = nn.Dropout(dropout)
        self.output_proj = nn.Linear(H, state_dim)

    # ------------------------------------------------------------------
    def precompute_graph(self, edge_index: torch.Tensor, num_nodes: int) -> None:
        """
        Precompute normalised edge list and structural edge features.
        Must be called before forward() for every new graph.

        Args:
            edge_index : (2, E) int64 on target device.  No self-loops needed.
            num_nodes  : N
        """
        device = edge_index.device
        N = num_nodes

        # Add self-loops so every node can attend to itself
        loop = torch.arange(N, device=device, dtype=torch.long)
        ei   = torch.cat([edge_index, torch.stack([loop, loop])], dim=1)  # (2, E+N)
        dst, src = ei[0], ei[1]

        # In-degree (counts self-loops too)
        deg = torch.zeros(N, device=device)
        deg.scatter_add_(0, dst, torch.ones(ei.size(1), device=device))
        deg_max = deg.max().clamp(min=1.0)

        # D^{-1} edge weight (kept for compatibility; not directly used in GT attention)
        edge_w = (1.0 / deg[dst].clamp(min=1.0)).float()

        # Initial structural edge features: [1/deg_dst, src_deg_norm, dst_deg_norm]
        feat = torch.stack([
            edge_w,
            (deg[src] / deg_max).float(),
            (deg[dst] / deg_max).float(),
        ], dim=1)   # (E+N, 3)

        self.register_buffer("_ei",        ei,     persistent=False)
        self.register_buffer("_edge_w",    edge_w, persistent=False)
        self.register_buffer("_edge_feat", feat,   persistent=False)
        self._sparse_N = N

    # ------------------------------------------------------------------
    def _sparse_softmax(
        self,
        scores: torch.Tensor,  # (BT, E, nh)
        dst:    torch.Tensor,  # (E,) int64 — destination node per edge
        N:      int,
    ) -> torch.Tensor:         # (BT, E, nh)
        """Numerically-stable softmax over incoming edges per destination node."""
        BT, E, nh = scores.shape

        # Subtract global max for numerical stability (valid: constant cancels in softmax)
        with torch.no_grad():
            s_max = scores.amax(dim=1, keepdim=True)   # (BT, 1, nh)
        exp_s = (scores - s_max).exp()

        # Sum per (bt, dst_node, head)
        idx   = dst[None, :, None].expand(BT, E, nh)
        denom = torch.zeros(BT, N, nh, device=scores.device, dtype=scores.dtype)
        denom.scatter_add_(1, idx, exp_s)

        return exp_s / (denom[:, dst, :] + 1e-9)

    # ------------------------------------------------------------------
    def _digress_gt_block(
        self,
        layer: nn.ModuleDict,
        h_BT:  torch.Tensor,   # (BT, N, H)   BT = B*(T+1)
        e:     torch.Tensor,   # (E, edge_dim)
    ):
        """One DiGress GT graph block (sparse, FiLM edge-conditioned)."""
        BT, N, H = h_BT.shape
        nh, df   = self.num_heads, self.head_dim
        dst, src = self._ei[0], self._ei[1]
        E        = self._ei.size(1)

        # ---------- Node attention with FiLM edge conditioning ----------
        Q = layer["Q"](h_BT).reshape(BT, N, nh, df)   # (BT, N, nh, df)
        K = layer["K"](h_BT).reshape(BT, N, nh, df)
        V = layer["V"](h_BT).reshape(BT, N, nh, df)

        # Dot-product score for every edge: Q[dst] · K[src] / √df  →  (BT, E, nh)
        scores = (Q[:, dst] * K[:, src]).sum(-1) / math.sqrt(df)

        # FiLM: multiply by (e_mul+1) and add e_add  — broadcast BT dimension
        scores = scores * (layer["e_mul"](e) + 1.0) + layer["e_add"](e)  # (BT, E, nh)

        # Sparse softmax over incoming edges per destination node
        attn = self._sparse_softmax(scores, dst, N)    # (BT, E, nh)

        # Weighted V[src] aggregation: (BT, E, nh, df) → flatten → (BT, E, H)
        agg = (attn.unsqueeze(-1) * V[:, src]).reshape(BT, E, H)

        # Scatter-add to destination nodes (loop avoids a large BT×E×H index tensor)
        row_H = dst.unsqueeze(1).expand(E, H)          # (E, H) — reused each iter
        out   = torch.zeros_like(h_BT)
        for bt in range(BT):
            out[bt].scatter_add_(0, row_H, agg[bt])

        h_out = h_BT + layer["out_proj"](out)          # residual (norm applied in forward)

        # ---------- Edge feature update ----------
        # Use mean node features over all B*(T+1) slices (edges are time-invariant)
        h_mean = h_BT.detach().mean(0)                  # (N, H)
        e_in   = torch.cat([h_mean[dst], h_mean[src], e], dim=-1)  # (E, 2H+de)
        e_new  = layer["edge_norm"](e + layer["edge_mlp"](e_in))

        return h_out, e_new

    # ------------------------------------------------------------------
    def forward(
        self,
        A:       torch.Tensor,  # (B, N, N) or (N, N) — only used for device check
        X_k:     torch.Tensor,  # (B, N, T+1, in_state_dim)
        k_index: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        assert hasattr(self, "_ei"), \
            "precompute_graph() must be called before forward()"
        B, N, T_plus_1, Cin = X_k.shape
        assert T_plus_1 == self.T_plus_1, \
            f"timestep mismatch: {T_plus_1} vs {self.T_plus_1}"
        assert Cin == self.in_state_dim, \
            f"channel mismatch: {Cin} vs {self.in_state_dim}"
        H = self.hidden_dim

        # 1. Epidemic-time positional embedding
        t_idx = torch.arange(T_plus_1, device=X_k.device, dtype=torch.long)
        t_emb = self.time_embed(t_idx)                           # (T+1, te)
        X     = torch.cat(
            [X_k, t_emb.view(1, 1, T_plus_1, -1).expand(B, N, T_plus_1, -1)],
            dim=-1
        )   # (B, N, T+1, in_state_dim + te)

        # 2. Input projection
        h = self.input_norm(self.input_proj(X))                  # (B, N, T+1, H)

        # 3. Diffusion-step embedding broadcast over (N, T+1)
        h = h + self.step_embed(k_index)[:, None, None, :]
        h = self.dropout(h)

        # 4. Initialise edge features from structural priors
        e = self.edge_proj(self._edge_feat)                      # (E, edge_dim)

        # Pre-compute time re-injection signal (same for all layers)
        t_reinject = self.time_reproject(t_emb)                  # (T+1, H)

        # 5. Alternating temporal-attention + DiGress GT layers
        for layer in self.layers:

            # --- Temporal self-attention (per node, over T+1 time steps) ---
            h_flat   = h.reshape(B * N, T_plus_1, H)
            attn_out, _ = layer["time_attn"](h_flat, h_flat, h_flat)
            h = layer["time_norm"](h + attn_out.reshape(B, N, T_plus_1, H))

            # --- Temporal FFN ---
            h = layer["ffn_norm"](h + layer["time_ffn"](h))

            # --- Re-inject time positional info before graph block ---
            # Ensures the spatial attention always knows which epidemic time step
            # it is processing, even in deep layers where the initial embedding
            # may have been diluted.
            h = h + t_reinject[None, None, :, :]                 # (B, N, T+1, H)

            # --- DiGress GT graph block ---
            # Fold B and T+1 into a single batch dimension so the GT block
            # sees shape (BT, N, H).  Each (batch, time) slice is processed
            # independently in space; temporal mixing was done by MHA above.
            h_BT = h.permute(0, 2, 1, 3).reshape(B * T_plus_1, N, H)
            h_BT, e = self._digress_gt_block(layer, h_BT, e)
            h = layer["graph_norm"](
                h_BT.reshape(B, T_plus_1, N, H).permute(0, 2, 1, 3)
            )

        # 6. Project to state dimension and normalise as probability simplex
        return torch.softmax(self.output_proj(h), dim=-1)
