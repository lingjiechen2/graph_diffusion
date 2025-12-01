import torch

from graph_diffusion.beta_diffusion import HistoryBetaDiffusion
from graph_diffusion.model import SpaceTimeGNN
from graph_diffusion.schedules import make_alpha_schedule
from graph_diffusion.sampling import sample_history


def test_forward_reverse_shapes():
    B, N, T, d_s = 2, 4, 3, 3
    K = 10
    Y = torch.rand(B, N, T + 1, d_s)
    A = torch.randint(0, 2, (N, N)).float()
    alphas = make_alpha_schedule(K)
    diffusion = HistoryBetaDiffusion(eta=10.0, alphas=alphas)
    model = SpaceTimeGNN(history_dim=(T + 1) * d_s, hidden_dim=32, num_layers=2)

    loss, _ = diffusion.klub_loss(model, A, Y, k=1)
    assert loss.shape == ()

    # mask only last slice as observed
    mask = torch.zeros_like(Y)
    mask[:, :, -1, :] = 1.0
    Y_obs = mask * Y
    recon = sample_history(diffusion, model, A, Y_obs, mask, num_steps=2)
    assert recon.shape == Y.shape
    assert torch.allclose(recon[:, :, -1, :], Y_obs[:, :, -1, :], atol=1e-4)
