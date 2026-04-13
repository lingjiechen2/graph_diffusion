import torch
from torch import nn


class SpaceTimeGNN(nn.Module):
    """Baseline message-passing network.

    This module treats its input as node features of shape (B,N,F) (or
    (B,N,T+1,d) which will be flattened). It is kept for simple low-dimensional
    diffusion spaces (e.g., SIR hitting times with shape (B,N,2)).
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
        max_steps: int = 1024,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.Linear(feature_dim, hidden_dim)
        self.step_embed = nn.Embedding(max_steps, hidden_dim)
        self.layers = nn.ModuleList(
            nn.ModuleList([nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)])
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )

    def _normalize_adj(self, A: torch.Tensor) -> torch.Tensor:
        if A.dim() == 2:
            A = A.unsqueeze(0)
        I = torch.eye(A.size(-1), device=A.device).expand_as(A)
        A_hat = A + I
        deg = A_hat.sum(-1, keepdim=True).clamp(min=1.0)
        return A_hat / deg

    def forward(self, A: torch.Tensor, X_k: torch.Tensor, k_index: torch.Tensor) -> torch.Tensor:
        # Accept either (B,N,F) or (B,N,T+1,d) and flatten the latter.
        if X_k.dim() == 4:
            B, N, T_plus_1, d = X_k.shape
            flat = X_k.view(B, N, -1)
            out_shape = (B, N, T_plus_1, d)
        else:
            B, N, _ = X_k.shape
            flat = X_k
            out_shape = X_k.shape

        h = self.encoder(flat)
        A_norm = self._normalize_adj(A)
        if A_norm.size(0) == 1 and B > 1:
            A_norm = A_norm.expand(B, -1, -1)

        step_emb = self.step_embed(k_index)[:, None, :]  # (B,1,H)
        for self_w, neigh_w in self.layers:
            m = torch.bmm(A_norm, h)
            h = self_w(h) + neigh_w(m) + step_emb
            h = torch.relu(h)
            h = self.dropout(h)

        out = self.decoder(h)
        return out.view(*out_shape)


class HistoryInpaintGNN(nn.Module):
    """History-mode denoiser that is *explicitly conditional* and time-aware.

    Key differences vs the baseline SpaceTimeGNN:
      - Expects a 4D history tensor (B,N,T+1,C_in) as input.
      - Adds a learned time positional embedding per true epidemic time t.
      - Outputs (B,N,T+1,d_s) as *probabilities on a simplex* via softmax.

    Intended use: conditional inpainting diffusion where the model input is
        X_k = concat([Z_k_inpaint, Y_obs, mask], dim=-1)
    with Z_k_inpaint clamped on observed entries.
    """

    def __init__(
        self,
        timesteps: int,
        state_dim: int,
        cond_components: int = 2,
        time_embed_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
        max_steps: int = 1024,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.T_plus_1 = timesteps + 1
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Input channels per (node,time): Z_k plus extra conditional channels.
        # If cond_components=2, input channels = (1+2)*d_s = 3*d_s for [Z, Y_obs, mask].
        self.in_state_dim = (1 + cond_components) * state_dim
        self.time_embed = nn.Embedding(max_steps, time_embed_dim)

        # Per-timestep encoder (not flattened): preserves temporal structure.
        per_time_dim = self.in_state_dim + time_embed_dim
        self.encoder = nn.Linear(per_time_dim, hidden_dim)
        self.step_embed = nn.Embedding(max_steps, hidden_dim)
        self.layers = nn.ModuleList(
            nn.ModuleList([nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)])
            for _ in range(num_layers)
        )
        # LayerNorm per GCN layer to prevent over-smoothing.
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        )
        # Gated aggregation: learn when to ignore neighbours.
        self.gates = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        # Temporal mixer: per-node mixing ACROSS timesteps.
        # A Linear(T+1, T+1) applied to the time dimension lets the model
        # learn temporal patterns like monotone S→I ordering.
        self.temporal_mix = nn.Linear(self.T_plus_1, self.T_plus_1)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        # Per-timestep decoder (not flattened).
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def _normalize_adj(self, A: torch.Tensor) -> torch.Tensor:
        if A.dim() == 2:
            A = A.unsqueeze(0)
        I = torch.eye(A.size(-1), device=A.device).expand_as(A)
        A_hat = A + I
        deg = A_hat.sum(-1, keepdim=True).clamp(min=1.0)
        return A_hat / deg

    def forward(self, A: torch.Tensor, X_k: torch.Tensor, k_index: torch.Tensor) -> torch.Tensor:
        # X_k: (B,N,T+1,in_state_dim)
        B, N, T_plus_1, Cin = X_k.shape
        assert T_plus_1 == self.T_plus_1, "timesteps mismatch"
        assert Cin == self.in_state_dim, "input channel mismatch"

        # True-time positional embedding (not diffusion step).
        t_idx = torch.arange(T_plus_1, device=X_k.device, dtype=torch.long)
        t_emb = self.time_embed(t_idx)  # (T+1, E)
        t_emb = t_emb.view(1, 1, T_plus_1, -1).expand(B, N, T_plus_1, -1)

        X = torch.cat([X_k, t_emb], dim=-1)  # (B,N,T+1,Cin+E)

        # Per-timestep encoder: (B,N,T+1,per_time_dim) -> (B,N,T+1,H)
        h = self.encoder(X)
        h_skip = h  # skip connection: preserve per-node features before GCN smoothing

        A_norm = self._normalize_adj(A)
        if A_norm.size(0) == 1 and B > 1:
            A_norm = A_norm.expand(B, -1, -1)

        # Diffusion step embedding: (B,1,1,H) broadcast over N and T+1.
        step_emb = self.step_embed(k_index)[:, None, None, :]

        # Per-timestep GCN with gating and LayerNorm.
        # Aggregate over spatial neighbours independently for each timestep.
        for (self_w, neigh_w), ln, gate_w in zip(self.layers, self.layer_norms, self.gates):
            # h: (B, N, T+1, H) -> permute to (B, T+1, N, H) for spatial matmul
            h_perm = h.permute(0, 2, 1, 3)                         # (B, T+1, N, H)
            # A_norm: (B, N, N) -> expand to (B, T+1, N, N)
            A_exp = A_norm.unsqueeze(1).expand(B, T_plus_1, N, N)
            m_perm = torch.matmul(A_exp, h_perm)                   # (B, T+1, N, H)
            m = m_perm.permute(0, 2, 1, 3)                         # (B, N, T+1, H)

            # Gated aggregation: learn when to ignore neighbourhood.
            gate = torch.sigmoid(gate_w(h))                         # (B, N, T+1, H)
            m = gate * m

            gcn_out = torch.relu(self_w(h) + neigh_w(m) + step_emb)
            gcn_out = self.dropout(gcn_out)
            h = ln(h + gcn_out)                                     # residual + LayerNorm

        # Global skip connection from pre-GCN encoder output.
        h = h + h_skip

        # Temporal mixing: let each node reason across its own T+1 timesteps.
        # Transpose so Linear operates across time: (B,N,H,T+1) -> mix -> (B,N,H,T+1)
        h_t = h.permute(0, 1, 3, 2)     # (B, N, H, T+1)
        h_t = self.temporal_mix(h_t)     # mix across T+1 dimension
        h_t = h_t.permute(0, 1, 3, 2)   # (B, N, T+1, H)
        h = self.temporal_norm(h + h_t)

        # Per-timestep decoder: (B,N,T+1,H) -> (B,N,T+1,state_dim)
        logits = self.decoder(h)
        probs = torch.softmax(logits, dim=-1)
        return probs
