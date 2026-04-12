import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def random_graph(num_nodes: int, p: float, device=None) -> torch.Tensor:
    """Symmetric Erdos-Renyi adjacency with zeros on diagonal."""
    tri = torch.bernoulli(torch.full((num_nodes, num_nodes), p, device=device))
    A = torch.triu(tri, diagonal=1)
    A = A + A.t()
    return A


def barabasi_albert_graph(num_nodes: int, attach: int, device=None) -> torch.Tensor:
    """Simple BA graph using torch sampling; not optimized for huge graphs."""
    if attach < 1 or attach >= num_nodes:
        raise ValueError("attach must be in [1, num_nodes).")
    A = torch.zeros(num_nodes, num_nodes, device=device)
    # start with a connected seed
    for i in range(attach):
        for j in range(i + 1, attach):
            A[i, j] = A[j, i] = 1
    degrees = A.sum(0)
    for new_node in range(attach, num_nodes):
        probs = degrees[:new_node] / degrees[:new_node].sum()
        targets = torch.multinomial(probs, attach, replacement=False)
        A[new_node, targets] = 1
        A[targets, new_node] = 1
        degrees = A.sum(0)
    return A


def sir_history(
    A: torch.Tensor,
    timesteps: int,
    infection_rate: float,
    recovery_rate: float,
    seed: int = 0,
    source_rate: float = 0.05,
    model: str = "sir",
):
    """
    Discrete-time SI/SIR simulator returning a history tensor Y in one-hot form.
    States: 0=S, 1=I, 2=R. Shape: (N, T+1, 3)
    """
    torch.manual_seed(seed)
    device = A.device
    N = A.size(0)
    Y = torch.zeros(N, timesteps + 1, 3, device=device)

    num_sources = max(1, int(source_rate * N))
    start_nodes = torch.randperm(N)[:num_sources]
    states = torch.zeros(N, dtype=torch.long, device=device)
    states[start_nodes] = 1

    # For SI, force recovery rate to zero.
    effective_recovery = 0.0 if model.lower() == "si" else recovery_rate

    for t in range(timesteps + 1):
        Y[:, t] = torch.nn.functional.one_hot(states, num_classes=3).float()
        if t == timesteps:
            break
        infected = states == 1
        susceptible = states == 0
        if infected.any():
            num_infected_neighbors = A[infected].float().sum(0)
            # Correct independent-channel model: P(infect) = 1 - (1-beta)^m
            neigh_prob = 1.0 - (1.0 - infection_rate) ** num_infected_neighbors
            neigh_prob = neigh_prob.clamp(0.0, 1.0)
            probs = torch.rand_like(neigh_prob)
            new_infections = (probs < neigh_prob) & susceptible
        else:
            new_infections = torch.zeros_like(states, dtype=torch.bool)

        recov_probs = torch.rand_like(states.float())
        recoveries = (recov_probs < effective_recovery) & infected

        states = states.clone()
        states[new_infections] = 1
        states[recoveries] = 2
    return Y


def load_ditto_adjacency(pt_path: str, device=None) -> torch.Tensor:
    """
    Load a dense adjacency matrix from a DITTO .pt file (torch_geometric Data).
    Returns A: (N, N) float symmetric adjacency.
    Requires torch_geometric to be installed.
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    N = data.num_nodes
    edge_index = data.edge_index  # (2, E), both directions already stored
    A = torch.zeros(N, N)
    A[edge_index[0], edge_index[1]] = 1.0
    if device is not None:
        A = A.to(device)
    return A


def load_ditto_history(pt_path: str, device=None):
    """
    Load (A, Y, T) from a DITTO .pt file.
    A: (N, N) float, symmetric adjacency.
    Y: (N, T+1, 3) float, one-hot history (always 3 channels for S/I/R).
    T: int, epidemic timespan.
    Requires torch_geometric to be installed.
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    N = data.num_nodes
    T = int(data.T.item())
    edge_index = data.edge_index
    A = torch.zeros(N, N)
    A[edge_index[0], edge_index[1]] = 1.0
    # data.y is (N, T+1) int64 with values in {0,1} (SI) or {0,1,2} (SIR).
    # Always encode as 3-channel one-hot so formats are uniform.
    Y = F.one_hot(data.y.long(), num_classes=3).float()  # (N, T+1, 3)
    if device is not None:
        A = A.to(device)
        Y = Y.to(device)
    return A, Y, T


class RealHistoryDataset(Dataset):
    """
    Dataset wrapping a single real diffusion history (D3 evaluation).
    __len__ returns 1; can be used directly with DataLoader for evaluation.
    """

    def __init__(self, A: torch.Tensor, Y: torch.Tensor):
        self.A = A
        self.Y = Y  # (N, T+1, d_s)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"Y": self.Y, "A": self.A}


class SyntheticHistoryDataset(Dataset):
    """
    On-the-fly synthetic dataset of histories on a fixed graph.
    """

    def __init__(
        self,
        A: torch.Tensor,
        timesteps: int,
        num_samples: int,
        infection_rate: float = 0.3,
        recovery_rate: float = 0.1,
        seed: int = 0,
        source_rate: float = 0.05,
        model: str = "sir",
    ):
        self.A = A
        self.timesteps = timesteps
        self.num_samples = num_samples
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.seed = seed
        self.source_rate = source_rate
        self.model = model

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Use a random seed each call so every epoch sees fresh epidemic simulations
        # (avoids memorisation of a fixed 256-sample set).
        import random
        seed = random.randint(0, 2**31 - 1)
        Y = sir_history(
            self.A,
            self.timesteps,
            self.infection_rate,
            self.recovery_rate,
            seed,
            source_rate=self.source_rate,
            model=self.model,
        )
        return {"Y": Y, "A": self.A}
