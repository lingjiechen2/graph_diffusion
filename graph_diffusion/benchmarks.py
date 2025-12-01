from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .data import (
    SyntheticHistoryDataset,
    barabasi_albert_graph,
    random_graph,
    sir_history,
)


@dataclass
class BenchmarkSpec:
    name: str
    graph_type: str
    timesteps: int
    infection_rate: float
    recovery_rate: float
    source_rate: float
    num_nodes: int
    graph_params: Dict
    history_dim: int = 3
    description: str = ""
    url: Optional[str] = None
    model: str = "sir"


# Synthetic graphs + synthetic diffusion (D1)
BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "D1-BA-SI": BenchmarkSpec(
        name="D1-BA-SI",
        graph_type="ba",
        timesteps=10,
        infection_rate=0.1,
        recovery_rate=0.0,
        source_rate=0.05,
        num_nodes=1000,
        graph_params={"attach": 4},
        description="Barabasi-Albert, SI diffusion",
        model="si",
    ),
    "D1-BA-SIR": BenchmarkSpec(
        name="D1-BA-SIR",
        graph_type="ba",
        timesteps=10,
        infection_rate=0.1,
        recovery_rate=0.1,
        source_rate=0.05,
        num_nodes=1000,
        graph_params={"attach": 4},
        description="Barabasi-Albert, SIR diffusion",
    ),
    "D1-ER-SI": BenchmarkSpec(
        name="D1-ER-SI",
        graph_type="er",
        timesteps=10,
        infection_rate=0.1,
        recovery_rate=0.0,
        source_rate=0.05,
        num_nodes=1000,
        graph_params={"p": 0.008},
        description="Erdos-Renyi, SI diffusion",
        model="si",
    ),
    "D1-ER-SIR": BenchmarkSpec(
        name="D1-ER-SIR",
        graph_type="er",
        timesteps=10,
        infection_rate=0.1,
        recovery_rate=0.1,
        source_rate=0.05,
        num_nodes=1000,
        graph_params={"p": 0.008},
        description="Erdos-Renyi, SIR diffusion",
    ),
    # Synthetic diffusion on real graphs (D2)
    "D2-Oregon2-SI": BenchmarkSpec(
        name="D2-Oregon2-SI",
        graph_type="external",
        timesteps=15,
        infection_rate=0.1,
        recovery_rate=0.0,
        source_rate=0.10,
        num_nodes=-1,
        graph_params={"path_hint": "http://snap.stanford.edu/data/Oregon-2.html"},
        description="Oregon2 AS graph (external edges), SI diffusion",
        url="http://snap.stanford.edu/data/Oregon-2.html",
        model="si",
    ),
    "D2-Oregon2-SIR": BenchmarkSpec(
        name="D2-Oregon2-SIR",
        graph_type="external",
        timesteps=15,
        infection_rate=0.1,
        recovery_rate=0.05,
        source_rate=0.10,
        num_nodes=-1,
        graph_params={"path_hint": "http://snap.stanford.edu/data/Oregon-2.html"},
        description="Oregon2 AS graph (external edges), SIR diffusion",
        url="http://snap.stanford.edu/data/Oregon-2.html",
    ),
    "D2-Prost-SI": BenchmarkSpec(
        name="D2-Prost-SI",
        graph_type="external",
        timesteps=15,
        infection_rate=0.1,
        recovery_rate=0.0,
        source_rate=0.10,
        num_nodes=-1,
        graph_params={"path_hint": "Prost dataset (see paper repo)"},
        description="Prost bipartite graph, SI diffusion",
        model="si",
    ),
    "D2-Prost-SIR": BenchmarkSpec(
        name="D2-Prost-SIR",
        graph_type="external",
        timesteps=15,
        infection_rate=0.1,
        recovery_rate=0.05,
        source_rate=0.10,
        num_nodes=-1,
        graph_params={"path_hint": "Prost dataset (see paper repo)"},
        description="Prost bipartite graph, SIR diffusion",
    ),
    # Real diffusion on real graphs (D3) – require user-provided histories.
    "D3-BrFarmers": BenchmarkSpec(
        name="D3-BrFarmers",
        graph_type="external_history",
        timesteps=-1,
        infection_rate=0.0,
        recovery_rate=0.0,
        source_rate=0.0,
        num_nodes=-1,
        graph_params={"url": "https://usccana.github.io/netdiffuseR/reference/brfarmers.html"},
        description="Brazilian farmers adoption diffusion (real histories).",
        url="https://usccana.github.io/netdiffuseR/reference/brfarmers.html",
    ),
    "D3-Pol": BenchmarkSpec(
        name="D3-Pol",
        graph_type="external_history",
        timesteps=-1,
        infection_rate=0.0,
        recovery_rate=0.0,
        source_rate=0.0,
        num_nodes=-1,
        graph_params={"url": "https://networkrepository.com/rt-pol.php"},
        description="Retweet network about US political event (real histories).",
        url="https://networkrepository.com/rt-pol.php",
    ),
    "D3-Covid": BenchmarkSpec(
        name="D3-Covid",
        graph_type="external_history",
        timesteps=-1,
        infection_rate=0.0,
        recovery_rate=0.0,
        source_rate=0.0,
        num_nodes=-1,
        graph_params={"url": "https://data.cdc.gov/Public-Health-Surveillance/United-States-COVID-19-Community-Levels-by-County/3nnm-4jni"},
        description="US county-level COVID-19 community levels (real histories).",
        url="https://data.cdc.gov/Public-Health-Surveillance/United-States-COVID-19-Community-Levels-by-County/3nnm-4jni",
    ),
    "D3-Hebrew": BenchmarkSpec(
        name="D3-Hebrew",
        graph_type="external_history",
        timesteps=-1,
        infection_rate=0.0,
        recovery_rate=0.0,
        source_rate=0.0,
        num_nodes=-1,
        graph_params={"url": "see paper repo (dataset [5])"},
        description="Hebrew election retweet network (real histories).",
    ),
}


def load_external_graph(edge_path: str, num_nodes: Optional[int] = None, device=None) -> torch.Tensor:
    """
    Load an undirected graph from an edge list file (u v per line).
    """
    edges = []
    with open(edge_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            edges.append((u, v))
    if num_nodes is None:
        num_nodes = max(max(u, v) for u, v in edges) + 1
    A = torch.zeros(num_nodes, num_nodes, device=device)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    return A


def make_graph_from_spec(spec: BenchmarkSpec, device=None, edge_path: Optional[str] = None) -> torch.Tensor:
    if spec.graph_type == "ba":
        return barabasi_albert_graph(spec.num_nodes, spec.graph_params["attach"], device=device)
    if spec.graph_type == "er":
        return random_graph(spec.num_nodes, spec.graph_params["p"], device=device)
    if spec.graph_type in {"external", "external_history"}:
        if edge_path is None:
            raise ValueError(f"edge_path required for {spec.name}. Hint: {spec.graph_params.get('path_hint', '')}")
        return load_external_graph(edge_path, device=device)
    raise ValueError(f"Unknown graph_type {spec.graph_type}")


def make_dataset_from_spec(
    spec: BenchmarkSpec,
    num_samples: int,
    device=None,
    edge_path: Optional[str] = None,
) -> SyntheticHistoryDataset:
    """
    Build a synthetic diffusion dataset for specs that require simulation.
    """
    if spec.graph_type not in {"ba", "er", "external"}:
        raise ValueError(f"{spec.name} requires real histories; provide them separately.")
    A = make_graph_from_spec(spec, device=device, edge_path=edge_path)
    ds = SyntheticHistoryDataset(
        A=A,
        timesteps=spec.timesteps,
        num_samples=num_samples,
        infection_rate=spec.infection_rate,
        recovery_rate=spec.recovery_rate,
        source_rate=spec.source_rate,
        model=spec.model,
    )
    return ds
