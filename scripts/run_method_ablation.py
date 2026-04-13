#!/usr/bin/env python
"""
Ablation study: 4 model-level methods to fix prediction skew, compared
against baseline. All methods are evaluated WITHOUT post-processing
(sir_project=False) to isolate the model's own discriminative power.

Methods:
  0. baseline     — current code, no changes
  1. reweight     — time-weighted + class-weighted KLUB loss
  2. selfcond     — self-conditioning (feed previous x0 estimate to model)
  3. seedhead     — auxiliary binary head predicting t=0 seeds
  4. backward     — backward masking curriculum (teach model to use future context)

Usage:
  python scripts/run_method_ablation.py --method baseline --gpu 1
  python scripts/run_method_ablation.py --method reweight --gpu 7
"""

import argparse
import copy
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_diffusion.benchmarks import BENCHMARKS, make_dataset_from_spec
from graph_diffusion.beta_diffusion import HistoryBetaDiffusion, EPS, beta_kl, _beta_sample
from graph_diffusion.model import HistoryInpaintGNN
from graph_diffusion.sampling import sample_history
from graph_diffusion.schedules import make_alpha_schedule
from graph_diffusion.metrics import evaluate_history
from sklearn.metrics import f1_score


# ============================================================
# Shared config
# ============================================================
BENCHMARK = "D1-BA-SI"
NUM_STEPS = 200
ETA = 40
TIMESTEPS = 10
STATE_DIM = 3
HIDDEN_DIM = 128
NUM_LAYERS = 3
TIME_EMBED_DIM = 16
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-3
SEED = 42
EVAL_SAMPLES = 5
DITTO_DIR = "/home/lingjie7/KDD23-DITTO/input"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_data(device):
    set_seed(SEED)
    g = torch.Generator().manual_seed(SEED)
    spec = BENCHMARKS[BENCHMARK]
    ds = make_dataset_from_spec(spec, num_samples=256, device=device, ditto_dir=DITTO_DIR)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    A = ds.A
    return loader, A


def make_base_model(device, cond_components=2):
    """Build a fresh HistoryInpaintGNN."""
    return HistoryInpaintGNN(
        timesteps=TIMESTEPS,
        state_dim=STATE_DIM,
        cond_components=cond_components,
        time_embed_dim=TIME_EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(device)


def evaluate_model(model, diffusion, loader, device, sir_project=False):
    """Evaluate model on EVAL_SAMPLES batches, return avg metrics + t=0 infected counts."""
    model.eval()
    results = []
    t0_true_all, t0_pred_all = [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= EVAL_SAMPLES:
                break
            Y_true = batch["Y"].to(device)
            A = batch["A"].to(device)
            mask = torch.zeros_like(Y_true)
            mask[:, :, -1, :] = 1.0
            Y_obs = mask * Y_true

            Y_pred = sample_history(diffusion, model, A, Y_obs, mask,
                                    num_steps=50, sir_project=sir_project)
            metrics = evaluate_history(Y_true, Y_pred)
            results.append(metrics)

            # t=0 infected counts (per batch, sum over B)
            for b in range(Y_true.shape[0]):
                t0_true_all.append((Y_true[b, :, 0, :].argmax(-1) == 1).sum().item())
                t0_pred_all.append((Y_pred[b, :, 0, :].argmax(-1) == 1).sum().item())

    agg = {k: sum(m[k] for m in results) / len(results) for k in results[0]}
    agg["t0_true_avg"] = np.mean(t0_true_all)
    agg["t0_pred_avg"] = np.mean(t0_pred_all)
    return agg


# ============================================================
# Method 0: Baseline
# ============================================================
def train_baseline(device):
    print("\n" + "=" * 60)
    print("METHOD 0: BASELINE")
    print("=" * 60)

    set_seed(SEED)
    loader, A_full = make_data(device)
    alphas = make_alpha_schedule(NUM_STEPS)
    diffusion = HistoryBetaDiffusion(eta=ETA, alphas=alphas.to(device))
    model = make_base_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total = 0.0
        for batch in loader:
            Y = batch["Y"].to(device)
            A = batch["A"].to(device)
            opt.zero_grad()
            mask = torch.zeros_like(Y)
            mask[:, :, -1, :] = 1.0
            Y_obs = mask * Y
            loss, metrics = diffusion.klub_loss(model, A, Y, Y_obs=Y_obs, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * Y.size(0)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch+1} | loss {total/256:.4f}", flush=True)

    result = evaluate_model(model, diffusion, loader, device, sir_project=False)
    print(f"  RESULT: F1={result['macro_f1']:.4f}  NRMSE={result['nrmse']:.4f}  "
          f"t0_true={result['t0_true_avg']:.1f}  t0_pred={result['t0_pred_avg']:.1f}")
    return model, result


# ============================================================
# Method 1: Time-weighted + class-weighted loss
# ============================================================
def klub_loss_reweighted(diffusion, model, A, Y, Y_obs, mask, device):
    """KLUB loss with time-weighted + class-weighted reweighting."""
    k = torch.randint(1, diffusion.K + 1, (1,), device=device).item()
    Z_k = diffusion.sample_marginal(Y, k)
    a_true, b_true = diffusion.posterior_params(Y, Z_k, k)

    Y_obs_eff = Y_obs.clamp(min=EPS, max=1 - EPS)
    Z_k_inpaint = (mask * Y_obs_eff + (1 - mask) * Z_k).clamp(min=EPS, max=1 - EPS)
    X_k = torch.cat([Z_k_inpaint, Y_obs_eff, mask], dim=-1)

    k_tensor = torch.full((Y.size(0),), k, device=device, dtype=torch.long)
    Y_hat = model(A, X_k, k_tensor)
    a_pred, b_pred = diffusion.reverse_params(Y_hat, k)

    kl = beta_kl(a_true, b_true, a_pred, b_pred)
    Zk_clamped = Z_k.clamp(min=EPS, max=1 - EPS)
    correction = -torch.log1p(-Zk_clamped)
    loss_terms = kl + correction  # (B, N, T+1, d_s)

    # --- Time weight: earlier timesteps get higher weight ---
    B, N, T1, ds = loss_terms.shape
    T = T1 - 1
    t_idx = torch.arange(T1, device=device, dtype=torch.float)
    time_w = (T + 1 - t_idx) / (T + 1)  # t=0: 1.0, t=T: 1/(T+1)
    time_w = time_w.view(1, 1, T1, 1).expand_as(loss_terms)

    # --- Class weight: upweight I-class (index 1) ---
    # Y has shape (B,N,T+1,d_s) with one-hot labels
    class_w = torch.ones_like(loss_terms)
    is_infected = (Y.argmax(-1) == 1).unsqueeze(-1).expand_as(loss_terms).float()
    class_w = class_w + 4.0 * is_infected  # 5x weight on I-class positions

    combined_w = time_w * class_w
    unobs = (1 - mask).detach()
    loss = (loss_terms * combined_w * unobs).sum() / (unobs.sum() + 1e-8)

    return loss, {"kl": kl.mean().item(), "k": k}


def train_reweight(device):
    print("\n" + "=" * 60)
    print("METHOD 1: LOSS REWEIGHTING")
    print("=" * 60)

    set_seed(SEED)
    loader, A_full = make_data(device)
    alphas = make_alpha_schedule(NUM_STEPS)
    diffusion = HistoryBetaDiffusion(eta=ETA, alphas=alphas.to(device))
    model = make_base_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total = 0.0
        for batch in loader:
            Y = batch["Y"].to(device)
            A = batch["A"].to(device)
            opt.zero_grad()
            mask = torch.zeros_like(Y)
            mask[:, :, -1, :] = 1.0
            Y_obs = mask * Y
            loss, metrics = klub_loss_reweighted(diffusion, model, A, Y, Y_obs, mask, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * Y.size(0)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch+1} | loss {total/256:.4f}", flush=True)

    result = evaluate_model(model, diffusion, loader, device, sir_project=False)
    print(f"  RESULT: F1={result['macro_f1']:.4f}  NRMSE={result['nrmse']:.4f}  "
          f"t0_true={result['t0_true_avg']:.1f}  t0_pred={result['t0_pred_avg']:.1f}")
    return model, result


# ============================================================
# Method 2: Self-conditioning
# ============================================================
def klub_loss_selfcond(diffusion, model, A, Y, Y_obs, mask, device):
    """KLUB loss with self-conditioning: 50% chance to feed previous x0 estimate."""
    k = torch.randint(1, diffusion.K + 1, (1,), device=device).item()
    Z_k = diffusion.sample_marginal(Y, k)
    a_true, b_true = diffusion.posterior_params(Y, Z_k, k)

    B, N, T1, ds = Y.shape
    Y_obs_eff = Y_obs.clamp(min=EPS, max=1 - EPS)
    mask_eff = mask
    Z_k_inpaint = (mask_eff * Y_obs_eff + (1 - mask_eff) * Z_k).clamp(min=EPS, max=1 - EPS)

    k_tensor = torch.full((B,), k, device=device, dtype=torch.long)

    # Self-conditioning: with 50% prob, first run model to get x0 estimate, then
    # feed that estimate as extra conditioning in a second pass.
    if random.random() < 0.5:
        # First pass (no gradient): get x0 estimate
        with torch.no_grad():
            x0_prev = torch.zeros(B, N, T1, ds, device=device)
            X_k_first = torch.cat([Z_k_inpaint, Y_obs_eff, mask_eff, x0_prev], dim=-1)
            x0_est = model(A, X_k_first, k_tensor).detach()
        # Second pass (with gradient): feed x0_est as extra channel
        X_k = torch.cat([Z_k_inpaint, Y_obs_eff, mask_eff, x0_est], dim=-1)
    else:
        x0_prev = torch.zeros(B, N, T1, ds, device=device)
        X_k = torch.cat([Z_k_inpaint, Y_obs_eff, mask_eff, x0_prev], dim=-1)

    Y_hat = model(A, X_k, k_tensor)
    a_pred, b_pred = diffusion.reverse_params(Y_hat, k)

    kl = beta_kl(a_true, b_true, a_pred, b_pred)
    Zk_clamped = Z_k.clamp(min=EPS, max=1 - EPS)
    correction = -torch.log1p(-Zk_clamped)
    loss_terms = kl + correction

    unobs = (1 - mask_eff).detach()
    loss = (loss_terms * unobs).sum() / (unobs.sum() + 1e-8)
    return loss, {"kl": kl.mean().item(), "k": k}


@torch.no_grad()
def sample_selfcond(diffusion, model, A, Y_obs, mask, num_steps=50):
    """Reverse diffusion sampling with self-conditioning."""
    device = Y_obs.device
    B = Y_obs.size(0)
    shape = Y_obs.shape
    Y_obs_c = Y_obs.clamp(min=EPS, max=1 - EPS)
    Z_k = diffusion.prior(shape, device=device)

    _, N, T1, ds = shape
    x0_prev = torch.zeros(B, N, T1, ds, device=device)

    for k in reversed(range(1, num_steps + 1)):
        k_tensor = torch.full((B,), k, device=device, dtype=torch.long)
        Z_k = (mask * Y_obs_c + (1 - mask) * Z_k).clamp(min=EPS, max=1 - EPS)

        X_k = torch.cat([Z_k, Y_obs_c, mask, x0_prev], dim=-1)
        Y_hat = model(A, X_k, k_tensor)
        x0_prev = Y_hat.detach()  # carry forward for self-conditioning

        Z_prev = diffusion.reverse_sample(Z_k, Y_hat, k)
        Z_prev = mask * Y_obs_c + (1 - mask) * Z_prev
        Z_k = Z_prev.clamp(min=EPS, max=1 - EPS)

    return Z_k


def train_selfcond(device):
    print("\n" + "=" * 60)
    print("METHOD 2: SELF-CONDITIONING")
    print("=" * 60)

    set_seed(SEED)
    loader, A_full = make_data(device)
    alphas = make_alpha_schedule(NUM_STEPS)
    diffusion = HistoryBetaDiffusion(eta=ETA, alphas=alphas.to(device))
    # cond_components=3: [Z_k, Y_obs, mask, x0_prev]
    model = make_base_model(device, cond_components=3)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total = 0.0
        for batch in loader:
            Y = batch["Y"].to(device)
            A = batch["A"].to(device)
            opt.zero_grad()
            mask = torch.zeros_like(Y)
            mask[:, :, -1, :] = 1.0
            Y_obs = mask * Y
            loss, metrics = klub_loss_selfcond(diffusion, model, A, Y, Y_obs, mask, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * Y.size(0)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch+1} | loss {total/256:.4f}", flush=True)

    # Evaluate with self-cond sampling
    model.eval()
    results = []
    t0_true_all, t0_pred_all = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= EVAL_SAMPLES:
                break
            Y_true = batch["Y"].to(device)
            A = batch["A"].to(device)
            mask = torch.zeros_like(Y_true)
            mask[:, :, -1, :] = 1.0
            Y_obs = mask * Y_true
            Y_pred = sample_selfcond(diffusion, model, A, Y_obs, mask, num_steps=50)
            metrics = evaluate_history(Y_true, Y_pred)
            results.append(metrics)
            for b in range(Y_true.shape[0]):
                t0_true_all.append((Y_true[b, :, 0, :].argmax(-1) == 1).sum().item())
                t0_pred_all.append((Y_pred[b, :, 0, :].argmax(-1) == 1).sum().item())

    agg = {k: sum(m[k] for m in results) / len(results) for k in results[0]}
    agg["t0_true_avg"] = np.mean(t0_true_all)
    agg["t0_pred_avg"] = np.mean(t0_pred_all)
    print(f"  RESULT: F1={agg['macro_f1']:.4f}  NRMSE={agg['nrmse']:.4f}  "
          f"t0_true={agg['t0_true_avg']:.1f}  t0_pred={agg['t0_pred_avg']:.1f}")
    return model, agg


# ============================================================
# Method 3: Auxiliary seed detection head
# ============================================================
class InpaintGNNWithSeedHead(nn.Module):
    """HistoryInpaintGNN + binary seed detection head for t=0 infected nodes."""

    def __init__(self, base_model: HistoryInpaintGNN, hidden_dim: int = 128):
        super().__init__()
        self.base = base_model
        # Mark this so sampling.py knows to build [Z_k, Y_obs, mask] input
        self.in_state_dim = base_model.in_state_dim

        # Seed head: predict from the pre-decoder hidden state
        self.seed_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, A, X_k, k_index, return_seed_logits=False):
        # Replicate base model forward up to hidden state
        B, N, T_plus_1, Cin = X_k.shape
        t_idx = torch.arange(T_plus_1, device=X_k.device, dtype=torch.long)
        t_emb = self.base.time_embed(t_idx).view(1, 1, T_plus_1, -1).expand(B, N, T_plus_1, -1)
        X = torch.cat([X_k, t_emb], dim=-1)
        flat = X.reshape(B, N, -1)

        h = self.base.encoder(flat)
        h_skip = h
        A_norm = self.base._normalize_adj(A)
        if A_norm.size(0) == 1 and B > 1:
            A_norm = A_norm.expand(B, -1, -1)

        step_emb = self.base.step_embed(k_index)[:, None, :]
        for self_w, neigh_w in self.base.layers:
            m = torch.bmm(A_norm, h)
            h = self_w(h) + neigh_w(m) + step_emb
            h = torch.relu(h)
            h = self.base.dropout(h)

        h = h + h_skip
        logits = self.base.decoder(h).view(B, N, T_plus_1, self.base.state_dim)
        probs = torch.softmax(logits, dim=-1)

        if return_seed_logits:
            seed_logits = self.seed_head(h).squeeze(-1)  # (B, N)
            return probs, seed_logits
        return probs


def klub_loss_seedhead(diffusion, model, A, Y, Y_obs, mask, device, seed_weight=1.0):
    """KLUB loss + auxiliary seed detection BCE loss."""
    k = torch.randint(1, diffusion.K + 1, (1,), device=device).item()
    Z_k = diffusion.sample_marginal(Y, k)
    a_true, b_true = diffusion.posterior_params(Y, Z_k, k)

    Y_obs_eff = Y_obs.clamp(min=EPS, max=1 - EPS)
    Z_k_inpaint = (mask * Y_obs_eff + (1 - mask) * Z_k).clamp(min=EPS, max=1 - EPS)
    X_k = torch.cat([Z_k_inpaint, Y_obs_eff, mask], dim=-1)

    k_tensor = torch.full((Y.size(0),), k, device=device, dtype=torch.long)
    Y_hat, seed_logits = model(A, X_k, k_tensor, return_seed_logits=True)

    # KLUB loss
    a_pred, b_pred = diffusion.reverse_params(Y_hat, k)
    kl = beta_kl(a_true, b_true, a_pred, b_pred)
    Zk_clamped = Z_k.clamp(min=EPS, max=1 - EPS)
    correction = -torch.log1p(-Zk_clamped)
    loss_terms = kl + correction
    unobs = (1 - mask).detach()
    klub = (loss_terms * unobs).sum() / (unobs.sum() + 1e-8)

    # Seed detection loss: binary BCE on t=0 labels
    # seed_target: 1 if node is infected at t=0, 0 otherwise
    seed_target = (Y[:, :, 0, :].argmax(-1) == 1).float()  # (B, N)
    # Class-weighted BCE (seeds are ~5% of nodes)
    pos_weight = torch.tensor([19.0], device=device)  # ~950/50
    seed_loss = nn.functional.binary_cross_entropy_with_logits(
        seed_logits, seed_target, pos_weight=pos_weight
    )

    loss = klub + seed_weight * seed_loss
    return loss, {"kl": kl.mean().item(), "k": k, "seed_loss": seed_loss.item()}


def train_seedhead(device):
    print("\n" + "=" * 60)
    print("METHOD 3: AUXILIARY SEED HEAD")
    print("=" * 60)

    set_seed(SEED)
    loader, A_full = make_data(device)
    alphas = make_alpha_schedule(NUM_STEPS)
    diffusion = HistoryBetaDiffusion(eta=ETA, alphas=alphas.to(device))
    base_model = make_base_model(device)
    model = InpaintGNNWithSeedHead(base_model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total = 0.0
        for batch in loader:
            Y = batch["Y"].to(device)
            A = batch["A"].to(device)
            opt.zero_grad()
            mask = torch.zeros_like(Y)
            mask[:, :, -1, :] = 1.0
            Y_obs = mask * Y
            loss, metrics = klub_loss_seedhead(diffusion, model, A, Y, Y_obs, mask, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * Y.size(0)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch+1} | loss {total/256:.4f} | seed_loss {metrics['seed_loss']:.4f}", flush=True)

    result = evaluate_model(model, diffusion, loader, device, sir_project=False)
    print(f"  RESULT: F1={result['macro_f1']:.4f}  NRMSE={result['nrmse']:.4f}  "
          f"t0_true={result['t0_true_avg']:.1f}  t0_pred={result['t0_pred_avg']:.1f}")
    return model, result


# ============================================================
# Method 4: Backward masking curriculum
# ============================================================
def klub_loss_backward(diffusion, model, A, Y, Y_obs_base, mask_base, device):
    """
    KLUB loss with backward masking curriculum.

    With 50% probability, expand the conditioning mask to include some
    true future timesteps (t=T-1, T-2, ..., T-r for random r). This teaches
    the model to exploit backward context from later timesteps when available.

    At sampling time, we cascade: first sample conditioning only on t=T,
    then re-sample conditioning on t=T + predicted t=T-1, etc.
    """
    k = torch.randint(1, diffusion.K + 1, (1,), device=device).item()
    Z_k = diffusion.sample_marginal(Y, k)
    a_true, b_true = diffusion.posterior_params(Y, Z_k, k)

    B, N, T1, ds = Y.shape
    T = T1 - 1

    # Backward curriculum: randomly reveal extra future timesteps
    if random.random() < 0.5:
        # Reveal r random timesteps from the end (in addition to t=T)
        r = random.randint(1, max(1, T // 2))  # reveal 1 to T//2 extra timesteps
        mask = mask_base.clone()
        for t in range(T - r, T):
            mask[:, :, t, :] = 1.0
        Y_obs = mask * Y
    else:
        mask = mask_base
        Y_obs = Y_obs_base

    Y_obs_eff = Y_obs.clamp(min=EPS, max=1 - EPS)
    Z_k_inpaint = (mask * Y_obs_eff + (1 - mask) * Z_k).clamp(min=EPS, max=1 - EPS)
    X_k = torch.cat([Z_k_inpaint, Y_obs_eff, mask], dim=-1)

    k_tensor = torch.full((B,), k, device=device, dtype=torch.long)
    Y_hat = model(A, X_k, k_tensor)
    a_pred, b_pred = diffusion.reverse_params(Y_hat, k)

    kl = beta_kl(a_true, b_true, a_pred, b_pred)
    Zk_clamped = Z_k.clamp(min=EPS, max=1 - EPS)
    correction = -torch.log1p(-Zk_clamped)
    loss_terms = kl + correction
    unobs = (1 - mask).detach()
    loss = (loss_terms * unobs).sum() / (unobs.sum() + 1e-8)

    return loss, {"kl": kl.mean().item(), "k": k}


@torch.no_grad()
def sample_backward_cascade(diffusion, model, A, Y_obs, mask, num_steps=50, refine_passes=3):
    """
    Cascade sampling: initial diffusion → iteratively expand conditioning mask
    with argmax predictions from previous pass.
    """
    device = Y_obs.device
    B, N, T1, ds = Y_obs.shape
    T = T1 - 1

    # Initial pass: standard sampling
    Z = sample_history(diffusion, model, A, Y_obs, mask, num_steps=num_steps,
                       sir_project=False)

    # Refinement passes: expand mask backward from T
    for r in range(1, min(refine_passes + 1, T)):
        # Expand mask to include t=T-r based on previous predictions
        mask_exp = mask.clone()
        Y_obs_exp = Y_obs.clone()
        for t in range(T - r, T):
            mask_exp[:, :, t, :] = 1.0
            # Use argmax of previous pass as "pseudo-observation"
            pred_onehot = torch.zeros(B, N, ds, device=device)
            pred_onehot.scatter_(-1, Z[:, :, t, :].argmax(-1, keepdim=True), 1.0)
            Y_obs_exp[:, :, t, :] = pred_onehot

        Z = sample_history(diffusion, model, A, Y_obs_exp, mask_exp,
                           num_steps=num_steps, sir_project=False)

    return Z


def train_backward(device):
    print("\n" + "=" * 60)
    print("METHOD 4: BACKWARD MASKING CURRICULUM")
    print("=" * 60)

    set_seed(SEED)
    loader, A_full = make_data(device)
    alphas = make_alpha_schedule(NUM_STEPS)
    diffusion = HistoryBetaDiffusion(eta=ETA, alphas=alphas.to(device))
    model = make_base_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total = 0.0
        for batch in loader:
            Y = batch["Y"].to(device)
            A = batch["A"].to(device)
            opt.zero_grad()
            mask = torch.zeros_like(Y)
            mask[:, :, -1, :] = 1.0
            Y_obs = mask * Y
            loss, metrics = klub_loss_backward(diffusion, model, A, Y, Y_obs, mask, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * Y.size(0)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch+1} | loss {total/256:.4f}", flush=True)

    # Evaluate with cascade sampling
    model.eval()
    results = []
    t0_true_all, t0_pred_all = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= EVAL_SAMPLES:
                break
            Y_true = batch["Y"].to(device)
            A = batch["A"].to(device)
            mask = torch.zeros_like(Y_true)
            mask[:, :, -1, :] = 1.0
            Y_obs = mask * Y_true
            Y_pred = sample_backward_cascade(diffusion, model, A, Y_obs, mask,
                                             num_steps=50, refine_passes=3)
            metrics = evaluate_history(Y_true, Y_pred)
            results.append(metrics)
            for b in range(Y_true.shape[0]):
                t0_true_all.append((Y_true[b, :, 0, :].argmax(-1) == 1).sum().item())
                t0_pred_all.append((Y_pred[b, :, 0, :].argmax(-1) == 1).sum().item())

    agg = {k: sum(m[k] for m in results) / len(results) for k in results[0]}
    agg["t0_true_avg"] = np.mean(t0_true_all)
    agg["t0_pred_avg"] = np.mean(t0_pred_all)
    print(f"  RESULT: F1={agg['macro_f1']:.4f}  NRMSE={agg['nrmse']:.4f}  "
          f"t0_true={agg['t0_true_avg']:.1f}  t0_pred={agg['t0_pred_avg']:.1f}")
    return model, agg


# ============================================================
# Main
# ============================================================
METHODS = {
    "baseline": train_baseline,
    "reweight": train_reweight,
    "selfcond": train_selfcond,
    "seedhead": train_seedhead,
    "backward": train_backward,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=list(METHODS.keys()))
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda")
    print(f"Running method={args.method} on GPU {args.gpu}")

    t0 = time.time()
    model, result = METHODS[args.method](device)
    elapsed = time.time() - t0

    # Save checkpoint
    ckpt_path = f"runs/ablation-{args.method}.pt"
    torch.save(model.state_dict(), ckpt_path)

    print(f"\nDone in {elapsed:.0f}s. Saved to {ckpt_path}")
    print(f"Final: {result}")
