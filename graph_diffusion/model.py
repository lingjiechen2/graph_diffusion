from typing import Optional

import torch
from torch import nn


class SpaceTimeGNN(nn.Module):
    """
    Lightweight message-passing network over node × time histories.

    Inputs:
        A: adjacency matrix (B,N,N) or (N,N)
        Z_k: noisy history tensor (B,N,T+1,d_s)
        k_index: diffusion step indices (B,)
    Output:
        Y_hat: predicted clean history (B,N,T+1,d_s) in [0,1]
    """

    def __init__(
        self,
        history_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.history_dim = history_dim
        self.input_dim = history_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.Linear(self.input_dim, hidden_dim)
        self.step_embed = nn.Embedding(1024, hidden_dim)
        self.layers = nn.ModuleList(
            nn.ModuleList([nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)])
            for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_dim),
            nn.Sigmoid(),
        )

    def _normalize_adj(self, A: torch.Tensor) -> torch.Tensor:
        if A.dim() == 2:
            A = A.unsqueeze(0)
        I = torch.eye(A.size(-1), device=A.device).expand_as(A)
        A_hat = A + I
        deg = A_hat.sum(-1, keepdim=True).clamp(min=1.0)
        return A_hat / deg

    def forward(self, A: torch.Tensor, Z_k: torch.Tensor, k_index: torch.Tensor) -> torch.Tensor:
        B, N, T_plus_1, d_s = Z_k.shape
        flat = Z_k.view(B, N, -1)
        h = self.encoder(flat)

        A_norm = self._normalize_adj(A)
        if A_norm.size(0) == 1 and B > 1:
            A_norm = A_norm.expand(B, -1, -1)

        step_emb = self.step_embed(k_index)[:, None, :]  # (B,1,H)
        for self_w, neigh_w in self.layers:
            m = torch.bmm(A_norm, h)  # neighbor aggregation
            h = self_w(h) + neigh_w(m) + step_emb
            h = torch.relu(h)
            h = self.dropout(h)

        out = self.decoder(h)
        return out.view(B, N, T_plus_1, d_s)
