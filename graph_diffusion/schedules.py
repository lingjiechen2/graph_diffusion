import torch


def make_alpha_schedule(K: int, min_alpha: float = 1e-3, schedule: str = "cosine") -> torch.Tensor:
    """
    Create a decreasing alpha schedule with alpha_0 = 1 and alpha_K ≈ min_alpha.

    Args:
        K: number of diffusion steps.
        min_alpha: smallest alpha_K value.
        schedule: "cosine" or "linear".
    """
    if K < 1:
        raise ValueError("K must be at least 1.")
    if schedule == "cosine":
        steps = torch.linspace(0, 1, K + 1)
        alphas = torch.cos(0.5 * torch.pi * steps) ** 2
        alphas = alphas / alphas[0]
    elif schedule == "linear":
        alphas = torch.linspace(1.0, min_alpha, K + 1)
    else:
        raise ValueError(f"Unknown schedule {schedule}")
    alphas = alphas.clamp(min=min_alpha)
    alphas[0] = 1.0
    return alphas
