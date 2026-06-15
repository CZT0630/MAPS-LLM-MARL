"""Tests for C1: Hybrid Action Codec.

Validates:
  - Codec round-trip (logits → probs → env action)
  - Distillation loss (partition MSE + edge CE)
  - Actor output shape (3+E)
  - MADDPG agent select_action shape
  - MAPPO agent select_action shape
  - Replay buffer with new action dim
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from LLM4RL.algos.common.hybrid_action import HybridActionCodec
from LLM4RL.algos.happo.happo_agent import HAPPOAgent
from LLM4RL.algos.maddpg.maddpg_actor_critic import Actor
from LLM4RL.algos.mappo.mappo_agent import MAPPOAgent


class TestHybridActionCodec:
    """Unit tests for HybridActionCodec."""

    def test_action_dim(self):
        codec = HybridActionCodec(num_edges=5)
        assert codec.action_dim == 3 + 5

    def test_action_dim_single_edge(self):
        codec = HybridActionCodec(num_edges=1)
        assert codec.action_dim == 4

    def test_invalid_num_edges(self):
        with pytest.raises(ValueError):
            HybridActionCodec(num_edges=0)
        with pytest.raises(ValueError):
            HybridActionCodec(num_edges=3.5)

    def test_logits_to_probs_shape(self):
        codec = HybridActionCodec(num_edges=3)
        part_logits = torch.randn(4, 3)
        edge_logits = torch.randn(4, 3)
        probs = codec.logits_to_probs(part_logits, edge_logits)
        assert probs.shape == (4, 6)

    def test_logits_to_probs_partition_sums_to_one(self):
        codec = HybridActionCodec(num_edges=3)
        part_logits = torch.randn(8, 3)
        edge_logits = torch.randn(8, 3)
        probs = codec.logits_to_probs(part_logits, edge_logits)
        part_sum = probs[:, :3].sum(dim=-1)
        torch.testing.assert_close(part_sum, torch.ones(8), atol=1e-5, rtol=0)

    def test_logits_to_probs_edge_sums_to_one(self):
        codec = HybridActionCodec(num_edges=4)
        part_logits = torch.randn(8, 3)
        edge_logits = torch.randn(8, 4)
        probs = codec.logits_to_probs(part_logits, edge_logits)
        edge_sum = probs[:, 3:].sum(dim=-1)
        torch.testing.assert_close(edge_sum, torch.ones(8), atol=1e-5, rtol=0)

    def test_policy_to_env_action(self):
        codec = HybridActionCodec(num_edges=3)
        # partition = [0.2, 0.5, 0.3], edge probs = [0.1, 0.7, 0.2]
        policy = np.array([0.2, 0.5, 0.3, 0.1, 0.7, 0.2], dtype=np.float32)
        env = codec.policy_to_env_action(policy)
        assert env.shape == (4,)
        np.testing.assert_allclose(env[:3], [0.2, 0.5, 0.3])
        assert env[3] == pytest.approx(1.0)  # argmax of [0.1, 0.7, 0.2]

    def test_batch_policy_to_env_actions(self):
        codec = HybridActionCodec(num_edges=2)
        policies = np.array([
            [0.5, 0.3, 0.2, 0.9, 0.1],  # edge 0
            [0.1, 0.8, 0.1, 0.3, 0.7],  # edge 1
        ], dtype=np.float32)
        envs = codec.batch_policy_to_env_actions(policies)
        assert envs.shape == (2, 4)
        assert envs[0, 3] == pytest.approx(0.0)
        assert envs[1, 3] == pytest.approx(1.0)

    def test_env_to_policy_action(self):
        codec = HybridActionCodec(num_edges=3)
        env_action = np.array([0.2, 0.5, 0.3, 2.0], dtype=np.float32)
        policy = codec.env_to_policy_action(env_action)
        assert policy.shape == (6,)
        np.testing.assert_allclose(policy[:3], [0.2, 0.5, 0.3])
        np.testing.assert_allclose(policy[3:], [0.0, 0.0, 1.0])  # one-hot edge 2

    def test_env_to_policy_action_rejects_out_of_range_edge(self):
        codec = HybridActionCodec(num_edges=3)
        env_action = np.array([1.0, 0.0, 0.0, 99.0], dtype=np.float32)
        with pytest.raises(ValueError, match="edge_id"):
            codec.env_to_policy_action(env_action)

    @pytest.mark.parametrize("num_edges", [3, 5, 10])
    def test_round_trip(self, num_edges):
        codec = HybridActionCodec(num_edges=num_edges)
        original_env = np.array(
            [0.3, 0.4, 0.3, float(num_edges - 1)], dtype=np.float32
        )
        policy = codec.env_to_policy_action(original_env)
        recovered_env = codec.policy_to_env_action(policy)
        np.testing.assert_allclose(recovered_env[:3], original_env[:3], atol=1e-6)
        assert recovered_env[3] == pytest.approx(original_env[3])

    @pytest.mark.parametrize(
        "policy_action",
        [
            np.array([0.2, 0.5, 0.3, 1.0], dtype=np.float32),
            np.array([0.2, 0.5, 0.3, 0.5, -0.5, 1.0], dtype=np.float32),
            np.array([0.2, np.nan, 0.8, 0.0, 0.0, 1.0], dtype=np.float32),
        ],
    )
    def test_policy_to_env_action_rejects_invalid_action(self, policy_action):
        codec = HybridActionCodec(num_edges=3)
        with pytest.raises(ValueError):
            codec.policy_to_env_action(policy_action)


class TestHybridActors:
    @pytest.mark.parametrize("num_edges", [3, 5, 10])
    @pytest.mark.parametrize("agent_class", [MAPPOAgent, HAPPOAgent])
    def test_on_policy_action_uses_policy_layout(self, agent_class, num_edges):
        state_dim = 7
        global_state_dim = 20
        agent = agent_class(
            state_dim=state_dim,
            action_dim=3 + num_edges,
            global_state_dim=global_state_dim,
            agent_idx=0,
            num_edges=num_edges,
        )

        action, _ = agent.select_action(
            np.zeros(state_dim, dtype=np.float32),
            np.zeros(global_state_dim, dtype=np.float32),
            deterministic=True,
        )

        assert action.shape == (3 + num_edges,)
        assert action[:3].sum() == pytest.approx(1.0, abs=1e-6)
        assert action[3:].sum() == pytest.approx(1.0, abs=1e-6)
        assert np.count_nonzero(action[3:]) == 1

    @pytest.mark.parametrize("agent_class", [MAPPOAgent, HAPPOAgent])
    def test_on_policy_nonzero_edge_reaches_environment(self, agent_class):
        num_edges = 3
        agent = agent_class(
            state_dim=7,
            action_dim=3 + num_edges,
            global_state_dim=20,
            agent_idx=0,
            num_edges=num_edges,
        )
        with torch.no_grad():
            agent.actor.edge_head.weight.zero_()
            agent.actor.edge_head.bias.copy_(torch.tensor([-10.0, -10.0, 10.0]))

        action, _ = agent.select_action(
            np.zeros(7, dtype=np.float32),
            np.zeros(20, dtype=np.float32),
            deterministic=True,
        )
        env_action = HybridActionCodec(num_edges).policy_to_env_action(action)

        assert env_action[3] == pytest.approx(2.0)

    @pytest.mark.parametrize("num_edges", [3, 5, 10])
    def test_maddpg_actor_both_heads_receive_gradients(self, num_edges):
        actor = Actor(state_dim=7, num_edges=num_edges)
        action = actor(torch.randn(4, 7))
        weights = torch.arange(
            1, 4 + num_edges, dtype=action.dtype
        ).unsqueeze(0)

        (action * weights).sum().backward()

        output_layer = actor.network[-1]
        assert output_layer.weight.grad[:3].abs().sum() > 0
        assert output_layer.weight.grad[3:].abs().sum() > 0

    @pytest.mark.parametrize("agent_class", [MAPPOAgent, HAPPOAgent])
    def test_on_policy_actor_both_heads_receive_gradients(self, agent_class):
        agent = agent_class(
            state_dim=7,
            action_dim=8,
            global_state_dim=20,
            agent_idx=0,
            num_edges=5,
        )
        partition_logits, edge_logits = agent.actor(
            torch.randn(4, 7, device=agent.device)
        )
        policy_probs = agent.actor.codec.logits_to_probs(
            partition_logits, edge_logits
        )
        weights = torch.arange(
            1,
            policy_probs.shape[-1] + 1,
            dtype=policy_probs.dtype,
            device=agent.device,
        ).unsqueeze(0)

        (policy_probs * weights).sum().backward()

        assert agent.actor.partition_head.weight.grad.abs().sum() > 0
        assert agent.actor.edge_head.weight.grad.abs().sum() > 0

    @pytest.mark.parametrize("agent_class", [MAPPOAgent, HAPPOAgent])
    def test_on_policy_distribution_handles_extreme_logits(self, agent_class):
        agent = agent_class(
            state_dim=7,
            action_dim=6,
            global_state_dim=20,
            agent_idx=0,
            num_edges=3,
        )
        with torch.no_grad():
            agent.actor.partition_head.weight.zero_()
            agent.actor.partition_head.bias.copy_(
                torch.tensor([1000.0, -1000.0, -1000.0], device=agent.device)
            )

        part_dist, _, _, _, concentration = agent.actor.get_dist(
            torch.zeros(1, 7, device=agent.device)
        )
        sample = part_dist.sample()
        log_prob = part_dist.log_prob(sample)

        assert torch.all(concentration > 0)
        assert torch.isfinite(sample).all()
        assert torch.isfinite(log_prob).all()

    def test_mappo_entropy_is_batch_size_invariant(self):
        agent = MAPPOAgent(
            state_dim=7,
            action_dim=6,
            global_state_dim=20,
            agent_idx=0,
            num_edges=3,
        )
        state = np.zeros(7, dtype=np.float32)
        global_state = np.zeros(20, dtype=np.float32)
        action, info = agent.select_action(
            state, global_state, deterministic=True
        )

        def make_batch(size):
            return {
                "states": np.repeat(state[None, :], size, axis=0),
                "actions": np.repeat(action[None, :], size, axis=0),
                "log_probs": np.repeat(info["log_prob"], size),
                "advantages": np.arange(size, dtype=np.float32),
            }

        _, single_stats = agent._compute_policy_loss(make_batch(1))
        _, batch_stats = agent._compute_policy_loss(make_batch(4))

        assert batch_stats["entropy"] == pytest.approx(
            single_stats["entropy"], abs=3e-6
        )


class TestDistillationLoss:
    """Test the mixed distillation loss."""

    def test_partition_loss_uses_squared_l2_norm(self):
        codec = HybridActionCodec(num_edges=2)
        actor_probs = torch.tensor([[0.1, 0.1, 0.8, 0.5, 0.5]])
        expert_part = torch.tensor([[0.8, 0.1, 0.1]])
        expert_edge = torch.tensor([0])
        mask = torch.tensor([1.0])

        _, L_part, _ = codec.distillation_loss(
            actor_probs, expert_part, expert_edge, mask
        )

        assert L_part.item() == pytest.approx(0.98)

    def test_perfect_expert_zero_loss(self):
        codec = HybridActionCodec(num_edges=3)
        # Actor output exactly matches expert
        actor_probs = torch.tensor([
            [0.2, 0.5, 0.3, 0.0, 0.0, 1.0],
            [0.1, 0.8, 0.1, 1.0, 0.0, 0.0],
        ])
        expert_part = torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.8, 0.1]])
        expert_edge = torch.tensor([2, 0])  # edge IDs
        mask = torch.tensor([1.0, 1.0])

        total, L_part, L_edge = codec.distillation_loss(
            actor_probs, expert_part, expert_edge, mask
        )
        assert L_part.item() == pytest.approx(0.0, abs=1e-6)
        assert L_edge.item() == pytest.approx(0.0, abs=1e-3)

    def test_masked_out_experts_ignored(self):
        codec = HybridActionCodec(num_edges=3)
        actor_probs = torch.tensor([
            [0.1, 0.1, 0.8, 0.1, 0.1, 0.8],
            [0.1, 0.1, 0.8, 0.1, 0.1, 0.8],
        ])
        expert_part = torch.tensor([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])
        expert_edge = torch.tensor([0, 0])
        mask = torch.tensor([0.0, 0.0])  # all masked out

        total, L_part, L_edge = codec.distillation_loss(
            actor_probs, expert_part, expert_edge, mask
        )
        assert total.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_is_scalar(self):
        codec = HybridActionCodec(num_edges=5)
        actor_probs = codec.logits_to_probs(
            torch.randn(16, 3),
            torch.randn(16, 5),
        )
        expert_part = torch.randn(16, 3).softmax(dim=-1)
        expert_edge = torch.randint(0, 5, (16,))
        mask = torch.ones(16)

        total, L_part, L_edge = codec.distillation_loss(
            actor_probs, expert_part, expert_edge, mask
        )
        assert total.ndim == 0
        assert L_part.ndim == 0
        assert L_edge.ndim == 0

    def test_eta_edge_scales_edge_loss(self):
        codec = HybridActionCodec(num_edges=3)
        actor_probs = torch.tensor([[0.1, 0.1, 0.8, 0.8, 0.1, 0.1]])
        expert_part = torch.tensor([[0.8, 0.1, 0.1]])
        expert_edge = torch.tensor([2])
        mask = torch.tensor([1.0])

        total_1, _, L_edge_1 = codec.distillation_loss(
            actor_probs, expert_part, expert_edge, mask, eta_edge=1.0
        )
        total_2, _, L_edge_2 = codec.distillation_loss(
            actor_probs, expert_part, expert_edge, mask, eta_edge=2.0
        )
        # With higher eta_edge, total should be larger
        assert total_2.item() > total_1.item()

    @pytest.mark.parametrize(
        ("actor_probs", "expert_part", "expert_edge", "mask"),
        [
            (
                torch.tensor([[0.2, 0.5, 0.3, 0.4, 0.6]]),
                torch.tensor([[0.2, 0.5, 0.3]]),
                torch.tensor([0]),
                torch.tensor([1.0]),
            ),
            (
                torch.tensor([[0.2, 0.5, 0.3, 0.2, 0.3, 0.5]]),
                torch.tensor([[0.2, 0.8]]),
                torch.tensor([0]),
                torch.tensor([1.0]),
            ),
            (
                torch.tensor([[0.2, 0.5, 0.3, 0.2, 0.3, 0.5]]),
                torch.tensor([[0.2, 0.5, 0.3]]),
                torch.tensor([3]),
                torch.tensor([1.0]),
            ),
            (
                torch.tensor([[0.2, 0.5, 0.3, 0.2, 0.3, 0.5]]),
                torch.tensor([[0.2, 0.5, 0.3]]),
                torch.tensor([0]),
                torch.tensor([[1.0]]),
            ),
            (
                torch.tensor([[0.2, 0.5, 0.3, 0.2, 0.3, 0.5]]),
                torch.tensor([[0.2, 0.5, 0.3]]),
                torch.tensor([0]),
                torch.tensor([0.5]),
            ),
        ],
    )
    def test_invalid_distillation_inputs_are_rejected(
        self, actor_probs, expert_part, expert_edge, mask
    ):
        codec = HybridActionCodec(num_edges=3)
        with pytest.raises(ValueError):
            codec.distillation_loss(
                actor_probs, expert_part, expert_edge, mask
            )
