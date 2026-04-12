"""
Spatiotemporal Graph Transformer backbone for graph diffusion history denoising.

Architecture per layer:
  (1) Temporal multi-head self-attention over T+1 epidemic time steps (per node)
  (2) Temporal FFN
  (3) Graph convolution over N nodes (per time step, all T+1 in parallel via matmul broadcast)

Compared to the baseline HistoryInpaintGNN:
  - Time axis has explicit self-attention instead of being flattened into a Linear layer.
    This lets the model learn that t=2 follows t=1 (causal structure of S→I→R).
  - Residual connections + LayerNorm after every sub-layer (standard Transformer hygiene).
  - Step embedding is broadcast across all (node, time) positions simultaneously.
  - Graph conv uses matmul broadcasting: A (B,1,N,N) × h (B,T+1,N,H) → avoids
    materialising an expanded adjacency tensor across time.
"""

import torch
from torch import nn


class STGraphTransformer(nn.Module):
    """
    Drop-in replacement for HistoryInpaintGNN with richer temporal modeling.

    Input contract (same as HistoryInpaintGNN):
        A       : (B, N, N) or (N, N)          adjacency
        X_k     : (B, N, T+1, in_state_dim)    where in_state_dim = (1+cond_components)*state_dim
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
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_mult: int = 2,
        dropout: float = 0.1,
        max_steps: int = 1024,
    ):
        super().__init__()
        self.T_plus_1 = timesteps + 1
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # in_state_dim: channels per (node, time) = [Z_k | Y_obs | mask] each of size state_dim
        self.in_state_dim = (1 + cond_components) * state_dim

        # Learned positional embedding for epidemic time axis (0 … T)
        self.time_embed = nn.Embedding(self.T_plus_1, time_embed_dim)
        # Diffusion step embedding (0 … K-1)
        self.step_embed = nn.Embedding(max_steps, hidden_dim)

        flat_in = self.in_state_dim + time_embed_dim
        self.input_proj = nn.Linear(flat_in, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Build alternating temporal-attention + graph-conv layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                # --- Temporal block ---
                "time_attn": nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True
                ),
                "time_norm": nn.LayerNorm(hidden_dim),
                "time_ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * ffn_mult),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * ffn_mult, hidden_dim),
                    nn.Dropout(dropout),
                ),
                "ffn_norm": nn.LayerNorm(hidden_dim),
                # --- Graph block ---
                "graph_self":  nn.Linear(hidden_dim, hidden_dim),
                "graph_neigh": nn.Linear(hidden_dim, hidden_dim),
                "graph_norm":  nn.LayerNorm(hidden_dim),
            }))

        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, state_dim)

    # ------------------------------------------------------------------
    def _normalize_adj(self, A: torch.Tensor) -> torch.Tensor:
        """Degree-normalised adjacency with self-loops: D^{-1}(A + I)."""
        if A.dim() == 2:
            A = A.unsqueeze(0)
        I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype).unsqueeze(0)
        A_hat = A + I
        deg = A_hat.sum(-1, keepdim=True).clamp(min=1.0)
        return A_hat / deg  # (B, N, N)

    # ------------------------------------------------------------------
    def precompute_graph(self, edge_index: torch.Tensor, num_nodes: int) -> None:
        """
        Pre-normalise an edge list for sparse graph conv.

        After calling this, forward() will use O(E·H) sparse scatter instead
        of O(N²·H) dense matmul — critical when N is large and the graph is
        sparse (e.g. Oregon2 N=11461 density 0.05%, Prost N=15810 density 0.03%).

        Args:
            edge_index: (2, E) int64 on the target device.  No self-loops needed.
            num_nodes:  N — required to add self-loops correctly.
        """
        device = edge_index.device
        N = num_nodes

        # Add self-loops
        loop_idx = torch.arange(N, device=device, dtype=torch.long)
        loop_ei = torch.stack([loop_idx, loop_idx], dim=0)        # (2, N)
        ei = torch.cat([edge_index, loop_ei], dim=1)               # (2, E+N)

        # D^{-1} normalisation: weight = 1 / degree[dst]
        row = ei[0]  # destination nodes
        deg = torch.zeros(N, device=device, dtype=torch.float)
        deg.scatter_add_(0, row, torch.ones(ei.size(1), device=device))
        edge_weight = (1.0 / deg[row].clamp(min=1.0)).to(torch.float32)

        # Register as non-parameter buffers so they move with .to(device)
        self.register_buffer("_ei_norm", ei, persistent=False)
        self.register_buffer("_ew_norm", edge_weight, persistent=False)
        self._sparse_N = N

    # ------------------------------------------------------------------
    def _sparse_graph_conv(self, h: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient sparse neighbourhood aggregation.

        Processes one time step at a time to avoid materialising an
        (B, E', T+1, H) index tensor (which would be ~1.5 GB for Prost).
        Peak extra memory per call: O(B · E' · H) ≈ 48 MB for Prost.

        h: (B, N, T+1, H)
        Returns: (B, N, T+1, H)  — aggregated neighbour features.
        """
        B, N, T_plus_1, H = h.shape
        ei = self._ei_norm          # (2, E')
        ew = self._ew_norm          # (E',)
        E = ei.size(1)
        row, col = ei[0], ei[1]     # dst, src

        # Pre-expand row index once for (E', H) scatter — reused each time step.
        row_H = row.unsqueeze(1).expand(E, H)                # (E', H)

        neigh = torch.zeros(B, N, T_plus_1, H, device=h.device, dtype=h.dtype)
        for t in range(T_plus_1):
            # Gather + weight:  (B, E', H)
            src_t = h[:, col, t, :] * ew.unsqueeze(1)        # (B, E', H)
            # Scatter-add for each batch item (usually B=1 for large graphs)
            for b in range(B):
                neigh[b, :, t, :].scatter_add_(0, row_H, src_t[b])
        return neigh

    # ------------------------------------------------------------------
    def forward(
        self,
        A: torch.Tensor,
        X_k: torch.Tensor,
        k_index: torch.Tensor,
    ) -> torch.Tensor:
        B, N, T_plus_1, Cin = X_k.shape
        assert T_plus_1 == self.T_plus_1, f"timestep mismatch: {T_plus_1} vs {self.T_plus_1}"
        assert Cin == self.in_state_dim, f"channel mismatch: {Cin} vs {self.in_state_dim}"
        H = self.hidden_dim

        # 1. Epidemic-time positional embedding
        t_idx = torch.arange(T_plus_1, device=X_k.device, dtype=torch.long)
        t_emb = self.time_embed(t_idx)             # (T+1, E)
        t_emb = t_emb.view(1, 1, T_plus_1, -1).expand(B, N, T_plus_1, -1)
        X = torch.cat([X_k, t_emb], dim=-1)        # (B, N, T+1, Cin+E)

        # 2. Project to hidden dim
        h = self.input_proj(X)                      # (B, N, T+1, H)
        h = self.input_norm(h)

        # 3. Diffusion-step embedding: broadcast over (N, T+1)
        step_emb = self.step_embed(k_index)         # (B, H)
        h = h + step_emb[:, None, None, :]
        h = self.dropout(h)

        # 4. Decide graph conv mode: sparse (precomputed) vs dense fallback
        use_sparse = hasattr(self, "_ei_norm") and self._ei_norm is not None

        if not use_sparse:
            # Dense fallback: O(N²·H) — fine for small graphs (D1, N≤1000)
            A_norm = self._normalize_adj(A)             # (B or 1, N, N)
            if A_norm.size(0) == 1 and B > 1:
                A_norm = A_norm.expand(B, -1, -1)
            A_norm_t = A_norm.unsqueeze(1)              # (B, 1, N, N)

        # 5. Alternating temporal-attention + graph-conv layers
        for layer in self.layers:

            # --- Temporal self-attention ---
            # Flatten (B, N) → (B*N) so that each node's T+1 time steps form one sequence.
            h_flat = h.reshape(B * N, T_plus_1, H)          # (B*N, T+1, H)
            attn_out, _ = layer["time_attn"](h_flat, h_flat, h_flat)
            h = layer["time_norm"](
                h + attn_out.reshape(B, N, T_plus_1, H)
            )

            # --- Temporal FFN ---
            h = layer["ffn_norm"](h + layer["time_ffn"](h))

            # --- Graph convolution ---
            if use_sparse:
                # Sparse: O(E'·H) per layer — E' << N² for real-world graphs
                neigh = self._sparse_graph_conv(h)           # (B, N, T+1, H)
            else:
                # Dense: O(N²·H) per layer — kept for D1 synthetic graphs
                h_perm = h.permute(0, 2, 1, 3)              # (B, T+1, N, H)
                neigh = torch.matmul(A_norm_t, h_perm)       # (B, T+1, N, H)
                neigh = neigh.permute(0, 2, 1, 3)            # (B, N, T+1, H)

            graph_out = torch.relu(
                layer["graph_self"](h) + layer["graph_neigh"](neigh)
            )
            h = layer["graph_norm"](h + graph_out)

        # 6. Project to state dimension and normalise as probability simplex
        logits = self.output_proj(h)                # (B, N, T+1, state_dim)
        return torch.softmax(logits, dim=-1)
