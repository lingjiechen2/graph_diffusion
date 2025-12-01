import torch

from graph_diffusion.metrics import compute_macro_f1, compute_nrmse, evaluate_history


def test_macro_f1_perfect():
    y_true = torch.tensor(
        [
            [[[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1]]],  # node 1
            [[[1, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]]],
        ],
        dtype=torch.float,
    )  # shape (2,2,2,3)
    # identical predictions
    f1 = compute_macro_f1(y_true, y_true)
    assert abs(f1 - 1.0) < 1e-6


def test_nrmse_hitting_time():
    # simple sequence where node hits infected at t=1 and recovered at t=2
    y_true = torch.tensor(
        [[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]], dtype=torch.float
    )  # (1,1,3,3)
    # prediction delays infection/recovery by 1 step
    y_pred = torch.tensor(
        [[[[1, 0, 0], [1, 0, 0], [0, 1, 0]]]], dtype=torch.float
    )
    nrmse = compute_nrmse(y_true, y_pred)
    # should be positive but small
    assert nrmse > 0


def test_evaluate_history_keys():
    y_true = torch.rand(1, 2, 2, 3)
    y_pred = y_true.clone()
    metrics = evaluate_history(y_true, y_pred)
    for key in ["macro_f1", "nrmse", "f1_gap", "nrmse_gap"]:
        assert key in metrics
