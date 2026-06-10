import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class HAPPOActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
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


def beta_kl_divergence(alpha_old, beta_old, alpha_new, beta_new):
    t1 = torch.lgamma(alpha_new) + torch.lgamma(beta_new) - torch.lgamma(alpha_new + beta_new)
    t2 = -(torch.lgamma(alpha_old) + torch.lgamma(beta_old) - torch.lgamma(alpha_old + beta_old))
    psi_sum_old = torch.digamma(alpha_old + beta_old)
    psi_sum_new = torch.digamma(alpha_new + beta_new)
    t3 = (alpha_old - alpha_new) * (torch.digamma(alpha_old) - psi_sum_old)
    t4 = (beta_old - beta_new) * (torch.digamma(beta_old) - psi_sum_old)
    t5 = (alpha_new + beta_new - alpha_old - beta_old) * (psi_sum_new - (torch.digamma(alpha_new) + torch.digamma(beta_new)))
    kl = t1 + t2 + t3 + t4 + t5
    return kl.sum(dim=-1)