import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Beta
from .mappo_policy import MAPPOActor, MAPPOCritic


class MAPPOAgent:
    def __init__(self, state_dim, action_dim, global_state_dim, agent_idx, config=None, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.agent_idx = agent_idx

        if config is None:
            config = {}

        self.lr_actor = kwargs.get('lr_actor', config.get('lr_actor', 3e-4))
        self.lr_critic = kwargs.get('lr_critic', config.get('lr_critic', 1e-3))
        self.gamma = kwargs.get('gamma', config.get('gamma', 0.99))
        self.lam = kwargs.get('lam', config.get('lam', 0.95))
        self.clip_range = kwargs.get('clip_range', config.get('clip_range', 0.2))
        self.entropy_coeff = kwargs.get('entropy_coeff', config.get('entropy_coeff', 0.01))
        self.value_coeff = kwargs.get('value_coeff', config.get('value_coeff', 0.5))
        self.update_epochs = kwargs.get('update_epochs', config.get('update_epochs', 4))
        self.batch_size = kwargs.get('batch_size', config.get('batch_size', 64))

        self.actor = MAPPOActor(state_dim, action_dim).to(self.device)
        self.critic = MAPPOCritic(global_state_dim).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def select_action(self, agent_state, global_state, deterministic=False):
        s = torch.as_tensor(agent_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        gs = torch.as_tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist, alpha, beta = self.actor.get_dist(s)
        if deterministic:
            action = (alpha / (alpha + beta)).detach()
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(gs).squeeze(-1)
        action_np = action.squeeze(0).detach().cpu().numpy()
        info = {
            'log_prob': float(log_prob.item()),
            'value': float(value.item()),
            'alpha': alpha.squeeze(0).detach().cpu().numpy(),
            'beta': beta.squeeze(0).detach().cpu().numpy(),
        }
        return action_np, info

    def _compute_policy_loss(self, batch):
        states = torch.as_tensor(batch['states'], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch['actions'], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch['log_probs'], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch['advantages'], dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dist_new, _, _ = self.actor.get_dist(states)
        new_log_probs = dist_new.log_prob(actions).sum(dim=-1)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        ppo_loss = -torch.min(surr1, surr2).mean()
        entropy = dist_new.entropy().sum(dim=-1).mean()
        loss = ppo_loss - self.entropy_coeff * entropy
        return loss, {
            'ppo_loss': float(ppo_loss.item()),
            'entropy': float(entropy.item()),
        }

    def _compute_value_loss(self, batch):
        global_states = torch.as_tensor(batch['global_states'], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch['returns'], dtype=torch.float32, device=self.device)
        values = self.critic(global_states).squeeze(-1)
        value_loss = F.mse_loss(values, returns)
        return value_loss

    def update(self, batch):
        losses = {}
        for _ in range(self.update_epochs):
            pl, stats = self._compute_policy_loss(batch)
            vl = self._compute_value_loss(batch)
            total_loss = pl + self.value_coeff * vl
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            total_loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()
            losses = {
                'policy_loss': float(pl.item()),
                'value_loss': float(vl.item()),
                'entropy': stats['entropy'],
            }
        return losses