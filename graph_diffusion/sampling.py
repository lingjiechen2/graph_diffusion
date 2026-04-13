from typing import Optional

import torch

from .beta_diffusion import HistoryBetaDiffusion, EPS
from .sir_postprocess import sir_bfs_project


@torch.no_grad()
def sample_history(
    diffusion: HistoryBetaDiffusion,
    model,
    A: torch.Tensor,
    Y_obs: torch.Tensor,
    mask: torch.Tensor,
    num_steps: Optional[int] = None,
    clamp_obs_each_step: bool = True,
    sir_project: bool = True,
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
    # Keep observation inside (0,1) for numeric stability.
    Y_obs = Y_obs.clamp(min=EPS, max=1 - EPS)
    Z_k = diffusion.prior(shape, device=device)

    for k in reversed(range(1, num_steps + 1)):
        k_tensor = torch.full((B,), k, device=device, dtype=torch.long)
        # Inpainting-style conditioning: keep observed entries fixed.
        Z_k = (mask * Y_obs + (1 - mask) * Z_k).clamp(min=EPS, max=1 - EPS)

        # Models that declare in_state_dim > state_dim expect explicit condition channels
        # [Z_k_inpaint | Y_obs | mask] concatenated on the last axis.
        # Both HistoryInpaintGNN and STGraphTransformer carry this attribute.
        if hasattr(model, "in_state_dim"):
            X_k = torch.cat([Z_k, Y_obs, mask], dim=-1)
        else:
            X_k = Z_k

        Y_hat = model(A, X_k, k_tensor)
        Z_prev = diffusion.reverse_sample(Z_k, Y_hat, k)
        if clamp_obs_each_step:
            Z_prev = mask * Y_obs + (1 - mask) * Z_prev
        Z_k = Z_prev.clamp(min=EPS, max=1 - EPS)

    if sir_project:
        Z_k = sir_bfs_project(Z_k, Y_obs, A)
    return Z_k
