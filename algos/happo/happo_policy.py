import torch
import torch.nn as nn
from torch.distributions import Categorical, Dirichlet

from ..common.hybrid_action import HybridActionCodec


class HAPPOActor(nn.Module):
    """Hybrid-action HAPPO actor (Dirichlet partition + categorical edge)."""

    def __init__(self, state_dim, num_edges, partition_concentration=10.0):
        super().__init__()
        self.num_edges = int(num_edges)
        if self.num_edges < 1:
            raise ValueError(f"num_edges must be >= 1, got {num_edges}")
        self.codec = HybridActionCodec(self.num_edges)
        self.partition_concentration = float(partition_concentration)
        if self.partition_concentration <= 0:
            raise ValueError("partition_concentration must be positive")
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.partition_head = nn.Linear(128, 3)
        self.edge_head = nn.Linear(128, self.num_edges)

    def forward(self, state):
        x = self.encoder(state)
        partition_logits = self.partition_head(x)
        edge_logits = self.edge_head(x)
        return partition_logits, edge_logits

    def get_dist(self, state):
        partition_logits, edge_logits = self.forward(state)
        concentration = self.codec.partition_concentration_from_logits(
            partition_logits,
            self.partition_concentration,
        )
        part_dist = Dirichlet(concentration)
        edge_dist = Categorical(logits=edge_logits)
        return (
            part_dist,
            edge_dist,
            partition_logits,
            edge_logits,
            concentration,
        )


class HAPPOCritic(nn.Module):
    def __init__(self, global_state_dim):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, global_state):
        return self.v(global_state)
