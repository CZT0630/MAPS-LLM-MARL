import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class MAPPOActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # 输出Beta参数，匹配[0,1]动作区间
        self.out = nn.Linear(128, action_dim * 2)

    def forward(self, state):
        x = self.encoder(state)
        params = self.out(x)
        alpha_raw, beta_raw = torch.chunk(params, 2, dim=-1)
        alpha = F.softplus(alpha_raw) + 1.0
        beta = F.softplus(beta_raw) + 1.0
        return alpha, beta

    def get_dist(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)
        return dist, alpha, beta


class MAPPOCritic(nn.Module):
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