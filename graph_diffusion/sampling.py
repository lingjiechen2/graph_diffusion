from typing import Optional

import torch

from .beta_diffusion import HistoryBetaDiffusion, EPS


@torch.no_grad()
def sample_history(
    diffusion: HistoryBetaDiffusion,
    model,
    A: torch.Tensor,
    Y_obs: torch.Tensor,
    mask: torch.Tensor,
    num_steps: Optional[int] = None,
) -> torch.Tensor:
    """
    Conditional reverse diffusion with inpainting-style clamping.

    Args:
        diffusion: HistoryBetaDiffusion instance.
        model: trained reverse model.
        A: adjacency matrix (B,N,N) or (N,N).
        Y_obs: tensor containing observed entries (same shape as mask).
        mask: binary tensor where 1 marks observed entries to clamp.
        num_steps: override number of reverse steps (defaults to diffusion.K).
    Returns:
        Reconstructed history tensor Z_0 approximating Y.
    """
    device = Y_obs.device
    if num_steps is None:
        num_steps = diffusion.K
    B = Y_obs.size(0)
    shape = Y_obs.shape
    Z_k = diffusion.prior(shape, device=device)

    for k in reversed(range(1, num_steps + 1)):
        k_tensor = torch.full((B,), k, device=device, dtype=torch.long)
        Y_hat = model(A, Z_k, k_tensor)
        Z_prev = diffusion.reverse_sample(Z_k, Y_hat, k)
        # Clamp observed entries back to the snapshot.
        Z_prev = mask * Y_obs + (1 - mask) * Z_prev
        Z_k = Z_prev.clamp(min=0.0 + EPS, max=1.0 - EPS)
    return Z_k
