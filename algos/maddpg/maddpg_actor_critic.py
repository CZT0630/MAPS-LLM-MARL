"""Actor and centralized critic used by the Phase 1 MADDPG baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Legacy continuous policy representation.

    The first three values are task partition ratios. The last value is a
    normalized edge selector in [0, 1]. Phase 3 will replace the selector with
    a categorical action head; keeping it continuous here preserves the legacy
    baseline while removing hard-coded edge counts.
    """

    def __init__(self, state_dim: int, action_dim: int = 4):
        super().__init__()
        if action_dim != 4:
            raise ValueError("Phase 1 actor expects 4 actions: 3 ratios + edge selector")
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.network(state)
        ratios = torch.softmax(raw[..., :3], dim=-1)
        edge_selector = torch.sigmoid(raw[..., 3:4])
        return torch.cat((ratios, edge_selector), dim=-1)


class Critic(nn.Module):
    """Centralized Q function over the real joint state and joint action."""

    def __init__(self, state_dim: int, action_dim: int, num_agents: int):
        super().__init__()
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
