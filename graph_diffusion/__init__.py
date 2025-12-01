"""
Graph Beta Diffusion for history reconstruction.

Expose key interfaces:
- HistoryBetaDiffusion: forward/reverse utilities and loss.
- SpaceTimeGNN: simple message passing network over node–time–state tensors.
- sample_history: conditional reverse diffusion with snapshot clamping.
"""

from .beta_diffusion import HistoryBetaDiffusion
from .model import SpaceTimeGNN
from .sampling import sample_history
from .schedules import make_alpha_schedule

__all__ = [
    "HistoryBetaDiffusion",
    "SpaceTimeGNN",
    "sample_history",
    "make_alpha_schedule",
]
