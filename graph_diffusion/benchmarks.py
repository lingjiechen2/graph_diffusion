import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from .data import (
    RealHistoryDataset,
    SyntheticHistoryDataset,
    barabasi_albert_graph,
    load_ditto_adjacency,
    load_ditto_history,
    random_graph,
    sir_history,
)


@dataclass
class BenchmarkSpec:
    name: str
    # graph_type: "ba" | "er" | "ditto" | "ditto_real"
    #   ditto       — load graph from DITTO .pt, simulate diffusion for training
    #   ditto_real  — load graph + single real history from DITTO .pt (D3)
    graph_type: str
    timesteps: int
    infection_rate: float
    recovery_rate: float
    source_rate: float
    num_nodes: int          # used for ba/er; ignored for ditto types (-1)
    graph_params: Dict
    history_dim: int = 3
    description: str = ""
    url: Optional[str] = None
    model: str = "sir"
    # Relative path from ditto_dir to the dataset .pt file.
    pt_subpath: Optional[str] = None


# ---------------------------------------------------------------------------
# D1 — synthetic graphs + synthetic diffusion
# ---------------------------------------------------------------------------
BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "D1-BA-SI": BenchmarkSpec(
        name="D1-BA-SI",
        graph_type="ditto",
        timesteps=10,
        infection_rate=0.1,
        recovery_rate=0.0,
        source_rate=0.05,
        num_nodes=-1,
        graph_params={},
        description="Barabasi-Albert (DITTO graph), SI diffusion",
        model="si",
        pt_subpath="synthetic/ba-si.pt",
    ),
    "D1-BA-SIR": BenchmarkSpec(
        name="D1-BA-SIR",
        graph_type="ditto",
        timesteps=10,
        infection_rate=0.1,
        recovery_rate=0.1,
        source_rate=0.05,
        num_nodes=-1,
        graph_params={},
        description="Barabasi-Albert (DITTO graph), SIR diffusion",
        pt_subpath="synthetic/ba-sir.pt",
    ),
    "D1-ER-SI": BenchmarkSpec(
        name="D1-ER-SI",
        graph_type="ditto",
        timesteps=10,
        infection_rate=0.1,
        recovery_rate=0.0,
        source_rate=0.05,
        num_nodes=-1,
        graph_params={},
        description="Erdos-Renyi (DITTO graph), SI diffusion",
        model="si",
        pt_subpath="synthetic/er-si.pt",
    ),
    "D1-ER-SIR": BenchmarkSpec(
        name="D1-ER-SIR",
        graph_type="ditto",
        timesteps=10,
        infection_rate=0.1,
        recovery_rate=0.1,
        source_rate=0.05,
        num_nodes=-1,
        graph_params={},
        description="Erdos-Renyi (DITTO graph), SIR diffusion",
        pt_subpath="synthetic/er-sir.pt",
    ),
    # ---------------------------------------------------------------------------
    # D2 — synthetic diffusion on real graphs (load topology from DITTO .pt)
    # ---------------------------------------------------------------------------
    "D2-Oregon2-SI": BenchmarkSpec(
        name="D2-Oregon2-SI",
        graph_type="ditto",
        timesteps=15,
        infection_rate=0.1,
        recovery_rate=0.0,
        source_rate=0.10,
        num_nodes=-1,
        graph_params={},
        description="Oregon2 AS graph, SI diffusion",
        url="http://snap.stanford.edu/data/Oregon-2.html",
        model="si",
        pt_subpath="oregon2/oregon2-si.pt",
    ),
    "D2-Oregon2-SIR": BenchmarkSpec(
        name="D2-Oregon2-SIR",
        graph_type="ditto",
        timesteps=15,
        infection_rate=0.1,
        recovery_rate=0.05,
        source_rate=0.10,
        num_nodes=-1,
        graph_params={},
        description="Oregon2 AS graph, SIR diffusion",
        url="http://snap.stanford.edu/data/Oregon-2.html",
        pt_subpath="oregon2/oregon2-sir.pt",
    ),
    "D2-Prost-SI": BenchmarkSpec(
        name="D2-Prost-SI",
        graph_type="ditto",
        timesteps=15,
        infection_rate=0.1,
        recovery_rate=0.0,
        source_rate=0.10,
        num_nodes=-1,
        graph_params={},
        description="Prost bipartite graph, SI diffusion",
        model="si",
        pt_subpath="prost/prost-si.pt",
    ),
    "D2-Prost-SIR": BenchmarkSpec(
        name="D2-Prost-SIR",
        graph_type="ditto",
        timesteps=15,
        infection_rate=0.1,
        recovery_rate=0.05,
        source_rate=0.10,
        num_nodes=-1,
        graph_params={},
        description="Prost bipartite graph, SIR diffusion",
        pt_subpath="prost/prost-sir.pt",
    ),
    # ---------------------------------------------------------------------------
    # D3 — real diffusion on real graphs (graph + single history from DITTO .pt)
    # Training uses synthetic diffusion on the real graph topology.
    # Evaluation uses the single real history loaded from the .pt file.
    # ---------------------------------------------------------------------------
    "D3-BrFarmers": BenchmarkSpec(
        name="D3-BrFarmers",
        graph_type="ditto_real",
        timesteps=16,
        infection_rate=0.3,
        recovery_rate=0.0,
        source_rate=0.10,
        num_nodes=-1,
        graph_params={},
        description="Brazilian farmers adoption diffusion (real SI history).",
        url="https://usccana.github.io/netdiffuseR/reference/brfarmers.html",
        model="si",
        pt_subpath="farmers/farmers-si.pt",
    ),
    "D3-Pol": BenchmarkSpec(
        name="D3-Pol",
        graph_type="ditto_real",
        timesteps=40,
        infection_rate=0.05,
        recovery_rate=0.0,
        source_rate=0.05,
        num_nodes=-1,
        graph_params={},
        description="Political retweet network (real SI history).",
        url="https://networkrepository.com/rt-pol.php",
        model="si",
        pt_subpath="pol/pol-si.pt",
    ),
    "D3-Covid": BenchmarkSpec(
        name="D3-Covid",
        graph_type="ditto_real",
        timesteps=10,
        infection_rate=0.1,
        recovery_rate=0.1,
        source_rate=0.10,
        num_nodes=-1,
        graph_params={},
        description="US county-level COVID-19 community levels (real SIR history).",
        url="https://data.cdc.gov/Public-Health-Surveillance/United-States-COVID-19-Community-Levels-by-County/3nnm-4jni",
        pt_subpath="covid/covid-sir.pt",
    ),
    "D3-Hebrew": BenchmarkSpec(
        name="D3-Hebrew",
        graph_type="ditto_real",
        timesteps=9,
        infection_rate=0.1,
        recovery_rate=0.1,
        source_rate=0.05,
        num_nodes=-1,
        graph_params={},
        description="Hebrew election retweet network (real SIR history).",
        pt_subpath="heb/heb-sir.pt",
    ),
}


# ---------------------------------------------------------------------------
# Graph / dataset builders
# ---------------------------------------------------------------------------

def _resolve_pt(spec: BenchmarkSpec, ditto_dir: Optional[str]) -> str:
    """Return the full path to a DITTO .pt file, raising a clear error if missing."""
    if ditto_dir is None:
        raise ValueError(
            f"{spec.name} requires --ditto-dir pointing to the KDD23-DITTO input/ folder."
        )
    path = os.path.join(ditto_dir, spec.pt_subpath)
    if not os.path.exists(path):
        raise FileNotFoundError(f"DITTO dataset not found: {path}")
    return path


def make_graph_from_spec(
    spec: BenchmarkSpec,
    device=None,
    ditto_dir: Optional[str] = None,
) -> torch.Tensor:
    if spec.graph_type == "ba":
        return barabasi_albert_graph(spec.num_nodes, spec.graph_params["attach"], device=device)
    if spec.graph_type == "er":
        return random_graph(spec.num_nodes, spec.graph_params["p"], device=device)
    if spec.graph_type in {"ditto", "ditto_real"}:
        pt_path = _resolve_pt(spec, ditto_dir)
        return load_ditto_adjacency(pt_path, device=device)
    raise ValueError(f"Unknown graph_type '{spec.graph_type}' for {spec.name}")


def make_dataset_from_spec(
    spec: BenchmarkSpec,
    num_samples: int,
    device=None,
    ditto_dir: Optional[str] = None,
    # legacy kwarg — kept for backward compat but ignored (use ditto_dir)
    edge_path: Optional[str] = None,
) -> SyntheticHistoryDataset:
    """
    Build a synthetic diffusion dataset for this spec.
    For all types (D1, D2, D3) this generates `num_samples` simulated histories
    on the appropriate graph topology.
    For D3 evaluation on real histories use load_real_history_from_spec() instead.
    """
    A = make_graph_from_spec(spec, device=device, ditto_dir=ditto_dir)
    return SyntheticHistoryDataset(
        A=A,
        timesteps=spec.timesteps,
        num_samples=num_samples,
        infection_rate=spec.infection_rate,
        recovery_rate=spec.recovery_rate,
        source_rate=spec.source_rate,
        model=spec.model,
    )


def load_real_history_from_spec(
    spec: BenchmarkSpec,
    device=None,
    ditto_dir: Optional[str] = None,
) -> RealHistoryDataset:
    """
    Return a RealHistoryDataset containing the single real history for D3 specs.
    Raises ValueError for non-ditto_real specs.
    """
    if spec.graph_type != "ditto_real":
        raise ValueError(f"{spec.name} is not a D3 real-history benchmark.")
    pt_path = _resolve_pt(spec, ditto_dir)
    A, Y, T = load_ditto_history(pt_path, device=device)
    return RealHistoryDataset(A, Y)
