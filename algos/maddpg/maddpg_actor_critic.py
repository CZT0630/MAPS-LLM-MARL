"""Actor and centralized critic used by the Phase 1 MADDPG baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..common.hybrid_action import HybridActionCodec


class Actor(nn.Module):
    """Hybrid-action policy network.

    Outputs:
      - 3 partition logits  → softmax → (local, edge, cloud) ratios
      - E edge logits       → softmax → edge server probability distribution

    Total output dim = 3 + num_edges.
    """

    def __init__(self, state_dim: int, num_edges: int):
        super().__init__()
        if num_edges < 1:
            raise ValueError(f"num_edges must be >= 1, got {num_edges}")
        self.num_edges = int(num_edges)
        self.codec = HybridActionCodec(self.num_edges)
        out_dim = 3 + self.num_edges
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return concatenated probability vector [phi_0..2, p_0..p_{E-1}]."""
        logits = self.logits(state)
        return self.codec.logits_to_probs(logits[..., :3], logits[..., 3:])


class Critic(nn.Module):
    """Centralized Q function over the real joint state and joint action."""

    def __init__(self, state_dim: int, num_edges: int, num_agents: int):
        super().__init__()
        action_dim = 3 + num_edges  # per-agent action dim
        joint_dim = state_dim * num_agents + action_dim * num_agents
        self.network = nn.Sequential(
            nn.Linear(joint_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        joint_states: torch.Tensor,
        joint_actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(torch.cat((joint_states, joint_actions), dim=-1))
