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

        # Input channels per (node,time): Z_k plus extra conditional channels.
        # If cond_components=2, input channels = (1+2)*d_s = 3*d_s for [Z, Y_obs, mask].
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flat_out_dim),
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
        flat = X.reshape(B, N, -1)

        h = self.encoder(flat)
        h_skip = h  # skip connection: preserve per-node features before GCN smoothing
        A_norm = self._normalize_adj(A)
        if A_norm.size(0) == 1 and B > 1:
            A_norm = A_norm.expand(B, -1, -1)

        step_emb = self.step_embed(k_index)[:, None, :]
        for self_w, neigh_w in self.layers:
            m = torch.bmm(A_norm, h)
            h = self_w(h) + neigh_w(m) + step_emb
            h = torch.relu(h)
            h = self.dropout(h)

        # Residual from pre-GCN representation preserves node-specific
        # final-state information that 3-layer mean aggregation would erase.
        h = h + h_skip

        logits = self.decoder(h).view(B, N, T_plus_1, self.state_dim)
        probs = torch.softmax(logits, dim=-1)
        return probs
