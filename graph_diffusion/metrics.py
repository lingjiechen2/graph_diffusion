from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _to_labels(y: torch.Tensor) -> torch.Tensor:
    """
    Convert history probabilities/one-hots to integer labels via argmax over state dim.
    Expects shape (..., d_s).
    """
    return y.argmax(dim=-1)


def compute_macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Macro F1 across all classes and time steps.
    Args:
        y_true: ground-truth history, shape (B,N,T+1,d_s), one-hot or probabilities.
        y_pred: reconstructed history, same shape.
    """
    true_labels = _to_labels(y_true)
    pred_labels = _to_labels(y_pred)
    num_classes = y_true.size(-1)
    f1_scores = []
    for c in range(num_classes):
        tp = ((pred_labels == c) & (true_labels == c)).sum().item()
        fp = ((pred_labels == c) & (true_labels != c)).sum().item()
        fn = ((pred_labels != c) & (true_labels == c)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
    return float(sum(f1_scores) / len(f1_scores))


def _hitting_time(labels: torch.Tensor, target: int) -> torch.Tensor:
    """
    Compute hitting times for target class per node.
    labels: (B,N,T+1)
    returns: (B,N) hitting time in [0, T+1] where T+1 means never hit.
    """
    B, N, T_plus_1 = labels.shape
    hits = torch.full((B, N), T_plus_1, device=labels.device, dtype=torch.float)
    for t in range(T_plus_1):
        mask = labels[:, :, t] == target
        hits = torch.where(mask & (hits == T_plus_1), torch.full_like(hits, float(t)), hits)
    return hits


def compute_nrmse(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    infected_idx: int = 1,
    recovered_idx: int = 2,
) -> float:
    """
    Normalized RMSE of hitting times for infected and recovered states.
    Implements Eq. (26) from DITTO:
        sqrt( sum_u [(h^I_true - h^I_pred)^2 + (h^R_true - h^R_pred)^2] / [2 n (T+1)]^2 )
    We compute per-graph NRMSE then average over the batch.
    """
    true_labels = _to_labels(y_true)
    pred_labels = _to_labels(y_pred)
    B, N, T_plus_1 = true_labels.shape
    hit_true_I = _hitting_time(true_labels, infected_idx)
    hit_pred_I = _hitting_time(pred_labels, infected_idx)
    hit_true_R = _hitting_time(true_labels, recovered_idx)
    hit_pred_R = _hitting_time(pred_labels, recovered_idx)
    num = (hit_true_I - hit_pred_I).pow(2) + (hit_true_R - hit_pred_R).pow(2)
    denom = (2 * N * T_plus_1) ** 2
    rmse_per_graph = torch.sqrt(num.sum(dim=1) / denom)  # shape (B,)
    return float(rmse_per_graph.mean())


def performance_gap(score: float, ideal: float, higher_is_better: bool = True) -> float:
    """
    Performance gap vs ideal.
    For F1 (higher better): (ideal - score)/ideal.
    For NRMSE (lower better): (score - ideal)/ideal.
    """
    # If ideal is zero (e.g., NRMSE best-case), return the raw score for lower-is-better metrics
    # so the gap reflects distance from perfect instead of collapsing to 0.
    if ideal == 0:
        return score if not higher_is_better else 0.0
    if higher_is_better:
        return (ideal - score) / ideal
    return (score - ideal) / ideal


@dataclass
class EvalResult:
    macro_f1: float
    nrmse: float
    f1_gap: float
    nrmse_gap: float


def evaluate_history(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    ideal_f1: float = 1.0,
    ideal_nrmse: float = 0.0,
) -> Dict[str, float]:
    f1 = compute_macro_f1(y_true, y_pred)
    nrmse = compute_nrmse(y_true, y_pred)
    return {
        "macro_f1": f1,
        "nrmse": nrmse,
        "f1_gap": performance_gap(f1, ideal_f1, higher_is_better=True),
        "nrmse_gap": performance_gap(nrmse, ideal_nrmse, higher_is_better=False),
    }
