"""Correct centralized-training MADDPG agent for the Phase 1 baseline."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .maddpg_actor_critic import Actor, Critic
from .noise import OUNoise
from ..common.hybrid_action import HybridActionCodec


class MADDPGAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        agent_idx: int,
        num_edges: int,
        config: dict | None = None,
        **kwargs,
    ):
        config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = int(state_dim)
        self.num_agents = int(num_agents)
        self.agent_idx = int(agent_idx)
        self.num_edges = int(num_edges)
        self.codec = HybridActionCodec(self.num_edges)
        self.action_dim = self.codec.action_dim
        if int(action_dim) != self.action_dim:
            raise ValueError(
                f"action_dim must be 3 + num_edges = {self.action_dim}, "
                f"got {action_dim}"
            )
        self.gamma = float(kwargs.get("gamma", config.get("gamma", 0.99)))
        self.tau = float(kwargs.get("tau", config.get("tau", 0.01)))
        self.lr_actor = float(kwargs.get("lr_actor", config.get("lr_actor", 3e-4)))
        self.lr_critic = float(kwargs.get("lr_critic", config.get("lr_critic", 3e-4)))
        self.distill_weight = float(
            kwargs.get("distill_weight", config.get("distill_weight", 0.0))
        )
        self.eta_edge = float(
            kwargs.get("eta_edge", config.get("eta_edge", 1.0))
        )
        if not math.isfinite(self.distill_weight) or self.distill_weight < 0.0:
            raise ValueError("distill_weight must be finite and non-negative")
        if not math.isfinite(self.eta_edge) or self.eta_edge < 0.0:
            raise ValueError("eta_edge must be finite and non-negative")

        self.actor = Actor(self.state_dim, self.num_edges).to(self.device)
        self.actor_target = Actor(self.state_dim, self.num_edges).to(self.device)
        self.critic = Critic(
            self.state_dim, self.num_edges, self.num_agents
        ).to(self.device)
        self.critic_target = Critic(
            self.state_dim, self.num_edges, self.num_agents
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr_critic
        )
        self.noise = OUNoise(
            self.action_dim,
            theta=float(config.get("noise_theta", 0.15)),
            sigma=float(config.get("noise_sigma", 0.2)),
        )
        self.training_count = 0

    def reset_noise(self) -> None:
        self.noise.reset()

    def select_action(self, state, add_noise: bool = True) -> np.ndarray:
        """Select action and return policy-format probabilities [3+E].

        Noise is added to logits before softmax to maintain valid probability
        distributions during exploration.
        """
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor.logits(state_tensor).squeeze(0)
        if add_noise:
            logits = logits + torch.as_tensor(
                self.noise.sample(), dtype=logits.dtype, device=self.device
            )
        action = self.codec.logits_to_probs(logits[:3], logits[3:])
        return action.cpu().numpy().astype(np.float32)

    def policy_to_env_action(self, policy_action) -> np.ndarray:
        """Convert policy probabilities to env action [a1, a2, a3, edge_id]."""
        return self.codec.policy_to_env_action(policy_action)

    def env_to_policy_action(self, env_action) -> np.ndarray:
        """Convert env action to policy format [3+E] with one-hot edge."""
        return self.codec.env_to_policy_action(env_action)

    def _target_joint_actions(
        self,
        next_states: torch.Tensor,
        all_agents: Iterable["MADDPGAgent"],
    ) -> torch.Tensor:
        actions = []
        for index, agent in enumerate(all_agents):
            actions.append(agent.actor_target(next_states[:, index, :]))
        return torch.stack(actions, dim=1)

    def _current_joint_actions_for_actor(
        self,
        states: torch.Tensor,
        replay_actions: torch.Tensor,
        all_agents: Iterable["MADDPGAgent"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        own_action = self.actor(states[:, self.agent_idx, :])
        joint_actions = []
        for index, agent in enumerate(all_agents):
            if index == self.agent_idx:
                joint_actions.append(own_action)
            else:
                with torch.no_grad():
                    joint_actions.append(agent.actor(states[:, index, :]))
        return own_action, torch.stack(joint_actions, dim=1)

    def update(
        self,
        batch: dict[str, np.ndarray],
        all_agents: list["MADDPGAgent"],
        lambda_distill: float | None = None,
    ) -> dict[str, float]:
        """Run one actor-critic update step.

        Parameters
        ----------
        batch : dict
            Joint replay batch with states, actions, rewards, etc.
        all_agents : list
            All MADDPG agents for centralized critic.
        lambda_distill : float or None
            Override distillation weight for this step.  When *None*, falls
            back to ``self.distill_weight``.  When ``0.0`` the distillation
            loss is not computed and produces no actor gradients.
        """
        lam = (
            float(lambda_distill)
            if lambda_distill is not None
            else self.distill_weight
        )
        if not math.isfinite(lam) or lam < 0.0:
            raise ValueError("lambda_distill must be finite and non-negative")

        states = torch.as_tensor(
            batch["states"], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(
            batch["rewards"], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            batch["next_states"], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            batch["done"], dtype=torch.float32, device=self.device
        ).reshape(-1, 1)
        expert_actions = torch.as_tensor(
            batch["expert_actions"], dtype=torch.float32, device=self.device
        )
        expert_mask = torch.as_tensor(
            batch["expert_mask"], dtype=torch.float32, device=self.device
        )

        batch_size = states.shape[0]
        flat_states = states.reshape(batch_size, -1)
        flat_actions = actions.reshape(batch_size, -1)
        flat_next_states = next_states.reshape(batch_size, -1)

        # --- Critic update ---
        with torch.no_grad():
            next_joint_actions = self._target_joint_actions(next_states, all_agents)
            target_q = rewards[:, self.agent_idx : self.agent_idx + 1]
            target_q = target_q + self.gamma * (1.0 - dones) * self.critic_target(
                flat_next_states, next_joint_actions.reshape(batch_size, -1)
            )

        current_q = self.critic(flat_states, flat_actions)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        # --- Actor update ---
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        own_action, joint_actions = self._current_joint_actions_for_actor(
            states, actions, all_agents
        )
        policy_loss = -self.critic(
            flat_states, joint_actions.reshape(batch_size, -1)
        ).mean()

        # --- Distillation loss (partition squared L2 + edge CE) ---
        # When lambda is 0, skip entirely so no distillation gradients flow.
        L_part_val = 0.0
        L_edge_val = 0.0
        distill_loss = torch.zeros((), device=self.device)
        if lam > 0.0:
            mask = expert_mask[:, self.agent_idx]
            if torch.any(mask > 0):
                expert_part = expert_actions[:, self.agent_idx, :3]
                expert_edge = expert_actions[
                    :, self.agent_idx, 3:
                ].argmax(dim=-1)
                distill_loss, L_part_t, L_edge_t = self.codec.distillation_loss(
                    own_action, expert_part, expert_edge, mask, self.eta_edge
                )
                L_part_val = float(L_part_t.detach().cpu())
                L_edge_val = float(L_edge_t.detach().cpu())
        actor_loss = policy_loss + lam * distill_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        self.training_count += 1

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "distill_loss": float(distill_loss.detach().cpu()),
            "L_distill": float(distill_loss.detach().cpu()),
            "L_part": L_part_val,
            "L_edge": L_edge_val,
            "lambda_distill": lam,
        }

    def _soft_update(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters()
        ):
            target_parameter.data.mul_(1.0 - self.tau)
            target_parameter.data.add_(self.tau * source_parameter.data)

    def save_model(self, path, metadata: dict | None = None) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "training_count": self.training_count,
                "num_edges": self.num_edges,
                "distill_weight": self.distill_weight,
                "eta_edge": self.eta_edge,
                "metadata": metadata or {},
            },
            path,
        )

    def load_model(self, path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.training_count = int(checkpoint.get("training_count", 0))
        self.distill_weight = float(
            checkpoint.get("distill_weight", self.distill_weight)
        )
        self.eta_edge = float(checkpoint.get("eta_edge", self.eta_edge))
