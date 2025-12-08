import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

EPS = 1e-4


def beta_kl(a_p: torch.Tensor, b_p: torch.Tensor, a_q: torch.Tensor, b_q: torch.Tensor) -> torch.Tensor:
    """
    KL(P || Q) for elementwise independent Beta distributions.
    Args use P = Beta(a_p, b_p), Q = Beta(a_q, b_q).
    """
    t1 = torch.lgamma(a_p + b_p) - torch.lgamma(a_p) - torch.lgamma(b_p)
    t2 = torch.lgamma(a_q + b_q) - torch.lgamma(a_q) - torch.lgamma(b_q)
    term = (
        (a_p - a_q) * torch.digamma(a_p)
        + (b_p - b_q) * torch.digamma(b_p)
        + (a_q + b_q - a_p - b_p) * torch.digamma(a_p + b_p)
    )
    return t1 - t2 + term


def _beta_sample(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.Beta(a, b)
    return dist.rsample()


@dataclass
class HistoryBetaDiffusion:
    """
    Implements the multiplicative Beta forward process and KLUB-style training objective.

    Attributes:
        eta: concentration parameter for Beta distributions.
        alphas: tensor of length K+1 with alphas[0]=1, alphas decreasing to ~0.
    """

    eta: float
    alphas: torch.Tensor

    def __post_init__(self):
        if self.alphas.dim() != 1:
            raise ValueError("alphas must be 1D.")
        if not torch.isclose(self.alphas[0], torch.tensor(1.0)):
            raise ValueError("alphas[0] must be 1.")

    @property
    def K(self) -> int:
        return self.alphas.numel() - 1

    def marginal_params(self, Y: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_k = self.alphas[k]
        Yc = Y.clamp(min=EPS, max=1 - EPS)
        a = self.eta * alpha_k * Yc
        b = self.eta * (1 - alpha_k * Yc)
        return a.clamp(min=EPS), b.clamp(min=EPS)

    def sample_marginal(self, Y: torch.Tensor, k: int) -> torch.Tensor:
        a, b = self.marginal_params(Y, k)
        z = _beta_sample(a, b)
        return z.clamp(min=EPS, max=1 - EPS)

    def posterior_params(self, Y: torch.Tensor, Z_k: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters of q(Z_{k-1} | Z_k, Y) over the Beta random variable V = (Z_{k-1} - Z_k) / (1 - Z_k).
        """
        if k == 0:
            raise ValueError("posterior_params undefined for k=0.")
        alpha_prev, alpha_cur = self.alphas[k - 1], self.alphas[k]
        Yc = Y.clamp(min=EPS, max=1 - EPS)
        a = self.eta * (alpha_prev - alpha_cur) * Yc
        b = self.eta * (1 - alpha_prev * Yc)
        return a.clamp(min=EPS), b.clamp(min=EPS)

    def reverse_params(self, Y_hat: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_prev, alpha_cur = self.alphas[k - 1], self.alphas[k]
        Yc = Y_hat.clamp(min=EPS, max=1 - EPS)
        a = self.eta * (alpha_prev - alpha_cur) * Yc
        b = self.eta * (1 - alpha_prev * Yc)
        return a.clamp(min=EPS), b.clamp(min=EPS)

    def reverse_sample(self, Z_k: torch.Tensor, Y_hat: torch.Tensor, k: int) -> torch.Tensor:
        """
        Sample Z_{k-1} ~ p_theta(Z_{k-1} | Z_k, A) using predicted Y_hat.
        """
        a, b = self.reverse_params(Y_hat, k)
        v = _beta_sample(a, b)
        return Z_k + (1 - Z_k) * v

    def klub_loss(
        self,
        model: nn.Module,
        A: torch.Tensor,
        Y: torch.Tensor,
        k: Optional[int] = None,
        project_fn=None,
        penalty_fn=None,
        penalty_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute KLUB loss for a random diffusion level k.
        """
        if k is None:
            k = torch.randint(1, self.K + 1, (1,), device=Y.device).item()

        # Forward sample Z_k from the true marginal.
        Z_k = self.sample_marginal(Y, k)
        a_true, b_true = self.posterior_params(Y, Z_k, k)

        # Predict denoised history.
        k_tensor = torch.full((Y.size(0),), k, device=Y.device, dtype=torch.long)
        Y_hat = model(A, Z_k, k_tensor)
        if project_fn is not None:
            Y_hat = project_fn(Y_hat)
        a_pred, b_pred = self.reverse_params(Y_hat, k)

        kl = beta_kl(a_true, b_true, a_pred, b_pred)

        # Correction term from KLUB: change-of-variable from V to Z_{k-1}.
        Zk_clamped = Z_k.clamp(min=EPS, max=1 - EPS)
        correction = -torch.log1p(-Zk_clamped)
        loss = (kl + correction).mean()
        penalty_val = 0.0
        if penalty_fn is not None and penalty_weight > 0:
            penalty = penalty_fn(Y_hat)
            penalty_val = penalty.item()
            loss = loss + penalty_weight * penalty
        metrics = {
            "kl": kl.mean().item(),
            "correction": correction.mean().item(),
            "k": k,
            "penalty": penalty_val,
        }
        return loss, metrics

    def prior(self, shape: torch.Size, device=None) -> torch.Tensor:
        """
        Stationary-like prior for Z_K; simple symmetric Beta fallback.
        """
        a = torch.full(shape, 0.5 * self.eta, device=device)
        b = torch.full(shape, 0.5 * self.eta, device=device)
        return _beta_sample(a.clamp(min=EPS), b.clamp(min=EPS))
