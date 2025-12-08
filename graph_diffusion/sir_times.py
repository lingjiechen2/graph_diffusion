import torch

from .sampling import EPS  # reuse numeric stability constant


def encode_hitting_times(history: torch.Tensor) -> torch.Tensor:
    """
    Encode full S/I/R history into normalized infection/recovery hitting times in [0,1].
    history: (B,N,T+1,3) one-hot or probabilities (argmax used)
    Returns: (B,N,2) with y_I, y_R in [0,1], where 1 denotes no event within horizon.
    """
    labels = history.argmax(dim=-1)  # (B,N,T+1)
    B, N, T_plus_1 = labels.shape
    device = labels.device
    # default to T+1 meaning never
    h_I = torch.full((B, N), T_plus_1, device=device, dtype=torch.long)
    h_R = torch.full((B, N), T_plus_1, device=device, dtype=torch.long)
    for t in range(T_plus_1):
        mask_I = labels[:, :, t] == 1
        h_I = torch.where((h_I == T_plus_1) & mask_I, torch.full_like(h_I, t), h_I)
        mask_R = labels[:, :, t] == 2
        h_R = torch.where((h_R == T_plus_1) & mask_R, torch.full_like(h_R, t), h_R)
    # enforce h_I <= h_R
    h_I_clamped = torch.minimum(h_I, h_R)
    h_R_clamped = torch.maximum(h_I, h_R)
    denom = float(T_plus_1)
    y_I = h_I_clamped.float() / denom
    y_R = h_R_clamped.float() / denom
    y = torch.stack([y_I, y_R], dim=-1)
    return y.clamp(min=EPS, max=1 - EPS)


def decode_hitting_times(times: torch.Tensor, T: int) -> torch.Tensor:
    """
    Decode normalized hitting times back to one-hot S/I/R history.
    times: (B,N,2) in [0,1]; T: horizon (integer).
    Returns: (B,N,T+1,3)
    """
    B, N, _ = times.shape
    T_plus_1 = T + 1
    # round to nearest integer time in [0, T+1]
    h = torch.round(times.clamp(0.0, 1.0) * T_plus_1).long()
    h = torch.clamp(h, 0, T_plus_1)
    # enforce monotone ordering after rounding
    h_I = torch.minimum(h[:, :, 0], h[:, :, 1]).unsqueeze(-1)
    h_R = torch.maximum(h[:, :, 0], h[:, :, 1]).unsqueeze(-1)
    ts = torch.arange(T_plus_1, device=times.device).view(1, 1, -1)
    S_mask = ts < h_I
    I_mask = (ts >= h_I) & (ts < h_R)
    R_mask = ts >= h_R
    history = torch.stack([S_mask, I_mask, R_mask], dim=-1).float()
    return history


def project_monotonic_times(times: torch.Tensor) -> torch.Tensor:
    """
    Enforce y_I <= y_R elementwise.
    """
    y_I = times[..., 0]
    y_R = times[..., 1]
    y_I_new = torch.minimum(y_I, y_R)
    y_R_new = torch.maximum(y_I, y_R)
    return torch.stack([y_I_new, y_R_new], dim=-1).clamp(min=EPS, max=1 - EPS)


def apply_final_state_constraints(times: torch.Tensor, final_snapshot: torch.Tensor, T: int) -> torch.Tensor:
    """
    Adjust hitting times to be consistent with the observed final snapshot at time T.
    Rules:
      - final S: y_I=y_R=1
      - final I: y_R=1, y_I <= T/(T+1)
      - final R: y_R <= T/(T+1), y_I <= y_R
    """
    labels = final_snapshot.argmax(dim=-1)  # (B,N)
    y_I = times[..., 0]
    y_R = times[..., 1]
    T_norm = float(T) / float(T + 1)
    # final S
    mask_S = labels == 0
    y_I = torch.where(mask_S, torch.ones_like(y_I), y_I)
    y_R = torch.where(mask_S, torch.ones_like(y_R), y_R)
    # final I
    mask_I = labels == 1
    y_R = torch.where(mask_I, torch.ones_like(y_R), y_R)
    y_I = torch.where(mask_I, torch.minimum(y_I, torch.full_like(y_I, T_norm)), y_I)
    # final R
    mask_R = labels == 2
    y_R = torch.where(mask_R, torch.minimum(y_R, torch.full_like(y_R, T_norm)), y_R)
    y_I = torch.minimum(y_I, y_R)
    return project_monotonic_times(torch.stack([y_I, y_R], dim=-1))


def monotonic_penalty(times: torch.Tensor) -> torch.Tensor:
    """
    Hinge penalty for monotonicity violations y_I > y_R.
    """
    y_I = times[..., 0]
    y_R = times[..., 1]
    return torch.relu(y_I - y_R).mean()


@torch.no_grad()
def sample_sir_times(
    diffusion,
    model,
    A: torch.Tensor,
    final_snapshot: torch.Tensor,
    T: int,
    num_steps: int = None,
    enforce_constraints: bool = True,
) -> torch.Tensor:
    """
    Reverse diffusion in (y_I, y_R) space with monotonic projection and final-state constraints.
    """
    device = final_snapshot.device
    B, N, _ = final_snapshot.shape
    if num_steps is None:
        num_steps = diffusion.K
    Z_k = diffusion.prior((B, N, 2), device=device)
    if enforce_constraints:
        Z_k = apply_final_state_constraints(project_monotonic_times(Z_k), final_snapshot, T)
    for k in reversed(range(1, num_steps + 1)):
        k_tensor = torch.full((B,), k, device=device, dtype=torch.long)
        Y_hat = model(A, Z_k, k_tensor)
        Y_hat = project_monotonic_times(Y_hat)
        if enforce_constraints:
            Y_hat = apply_final_state_constraints(Y_hat, final_snapshot, T)
        Z_prev = diffusion.reverse_sample(Z_k, Y_hat, k)
        Z_prev = project_monotonic_times(Z_prev)
        if enforce_constraints:
            Z_prev = apply_final_state_constraints(Z_prev, final_snapshot, T)
        Z_k = Z_prev
    return Z_k.clamp(min=EPS, max=1 - EPS)
