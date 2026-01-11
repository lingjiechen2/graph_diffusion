"""SIR-specific constraints for *history-mode* diffusion.

These are lightweight, idea-validation utilities to inject explicit SIR
transition knowledge without switching to the hitting-time parameterization.

All functions operate on a *probabilistic* history tensor Y of shape:
    (B, N, T+1, 3) with channels [S, I, R].
"""

import torch
import torch.nn.functional as F


def sir_monotonicity_penalty(Y: torch.Tensor) -> torch.Tensor:
    """Hinge penalty enforcing S non-increasing and R non-decreasing.

    - S_{t+1} <= S_t
    - R_{t+1} >= R_t
    """
    S = Y[..., 0]
    R = Y[..., 2]
    dS = S[:, :, 1:] - S[:, :, :-1]  # should be <= 0
    dR = R[:, :, :-1] - R[:, :, 1:]  # should be <= 0
    return torch.relu(dS).mean() + torch.relu(dR).mean()


def sir_mean_field_dynamics_penalty(
    Y: torch.Tensor,
    A: torch.Tensor,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    """Mean-field SIR one-step consistency penalty.

    Uses a smooth hazard mapping from neighbor infection intensity to infection
    probability:
        p_inf(u,t) = 1 - exp(-beta * sum_v A[u,v] * I(v,t))

    Then enforces expected transitions:
        S_{t+1} ≈ S_t (1 - p_inf)
        I_{t+1} ≈ I_t (1 - gamma) + S_t p_inf
        R_{t+1} ≈ R_t + gamma I_t

    This is not an exact likelihood (the simulator is stochastic), but it
    strongly regularizes histories toward SIR-like trajectories.
    """
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if A.size(0) == 1 and Y.size(0) > 1:
        A = A.expand(Y.size(0), -1, -1)
    A = A.float()

    S = Y[..., 0]
    I = Y[..., 1]
    R = Y[..., 2]

    S_t, I_t, R_t = S[:, :, :-1], I[:, :, :-1], R[:, :, :-1]  # (B,N,T)
    S_next, I_next, R_next = S[:, :, 1:], I[:, :, 1:], R[:, :, 1:]

    # Neighbor infection intensity per time step.
    # einsum: (B,N,N) x (B,N,T) -> (B,N,T)
    neigh_I = torch.einsum("bij,bjt->bit", A, I_t)

    # Smooth hazard -> probability.
    p_inf = 1.0 - torch.exp(-float(beta) * neigh_I)
    p_inf = p_inf.clamp(0.0, 1.0)

    S_hat = S_t * (1.0 - p_inf)
    new_inf = S_t * p_inf
    I_hat = I_t * (1.0 - float(gamma)) + new_inf
    R_hat = R_t + float(gamma) * I_t

    # MSE over all nodes and times.
    loss = F.mse_loss(S_next, S_hat) + F.mse_loss(I_next, I_hat) + F.mse_loss(R_next, R_hat)
    return loss


def sir_history_penalty(
    Y: torch.Tensor,
    A: torch.Tensor,
    beta: float,
    gamma: float,
    w_mono: float = 1.0,
    w_dyn: float = 1.0,
) -> torch.Tensor:
    """Convenience wrapper: monotonicity + mean-field dynamics."""
    return w_mono * sir_monotonicity_penalty(Y) + w_dyn * sir_mean_field_dynamics_penalty(Y, A, beta, gamma)
