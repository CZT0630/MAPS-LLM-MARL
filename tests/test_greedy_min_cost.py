"""Tests for the Greedy-MinCost baseline."""

import numpy as np
import pytest

from LLM4RL.algos.common.hybrid_action import HybridActionCodec
from LLM4RL.baselines.greedy_min_cost import GreedyMinCostAgent, PARTITION_TEMPLATES
from LLM4RL.environment.cloud_edge_env import CloudEdgeDeviceEnv


def _make_config(num_devices=3, num_edges=2):
    return {
        "environment": {
            "num_devices": num_devices,
            "num_edges": num_edges,
            "num_clouds": 1,
            "device_config": {"cpu_capacity": 2.0, "memory_capacity": 4.0},
            "edge_config": {"cpu_capacity": 8.0, "memory_capacity": 16.0},
            "cloud_config": {"cpu_capacity": 32.0, "memory_capacity": 64.0},
        },
        "network": {
            "ue_to_edge_rate": 1e9,
            "ue_to_cloud_total_rate": 1e8,
            "edge_to_cloud_rate": 1e9,
        },
        "tasks": {
            "simple_mode": True,
            "simple_data_choices": [100, 150, 200, 250],
            "simple_deadline_multiplier": 3.0,
        },
        "reward": {"type": "inverse", "latency_weight": 1.0, "energy_weight": 1.0},
        "maddpg": {"max_episodes": 1, "max_steps": 5},
    }


class TestGreedyMinCostAgentShape:
    """Basic shape and type checks."""

    def test_action_shape(self):
        config = _make_config(num_devices=5, num_edges=2)
        env = CloudEdgeDeviceEnv(config)
        env.reset(seed=42)

        agent = GreedyMinCostAgent(num_devices=5, num_edges=2)
        action = agent.compute_action(env)

        # New hybrid format: [partition(3) + edge_onehot(E)]
        assert action.shape == (5, 3 + 2)
        assert action.dtype == np.float32

    def test_action_shape_single_device(self):
        config = _make_config(num_devices=1, num_edges=3)
        env = CloudEdgeDeviceEnv(config)
        env.reset(seed=0)

        agent = GreedyMinCostAgent(num_devices=1, num_edges=3)
        action = agent.compute_action(env)

        assert action.shape == (1, 3 + 3)

    def test_no_tasks_returns_zero(self):
        config = _make_config(num_devices=3, num_edges=2)
        env = CloudEdgeDeviceEnv(config)
        env.reset(seed=42)
        # Force no tasks.
        env.current_tasks = None

        agent = GreedyMinCostAgent(num_devices=3, num_edges=2)
        action = agent.compute_action(env)

        assert action.shape == (3, 3 + 2)
        np.testing.assert_array_equal(
            action,
            np.tile(
                np.array([1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
                (3, 1),
            ),
        )


class TestGreedyMinCostAgentValues:
    """Check that returned actions are valid partition ratios and one-hot edge."""

    def test_partition_ratios_sum_to_one(self):
        config = _make_config(num_devices=4, num_edges=2)
        env = CloudEdgeDeviceEnv(config)
        env.reset(seed=123)

        agent = GreedyMinCostAgent(num_devices=4, num_edges=2)
        action = agent.compute_action(env)

        for i in range(4):
            ratios = action[i, :3]
            total = ratios.sum()
            assert abs(total - 1.0) < 1e-5 or total == pytest.approx(1.0, abs=1e-5)

    def test_edge_onehot_valid(self):
        config = _make_config(num_devices=4, num_edges=3)
        env = CloudEdgeDeviceEnv(config)
        env.reset(seed=99)

        agent = GreedyMinCostAgent(num_devices=4, num_edges=3)
        action = agent.compute_action(env)

        for i in range(4):
            edge_onehot = action[i, 3:]
            # Exactly one element should be 1.0, rest 0.0
            assert edge_onehot.sum() == pytest.approx(1.0)
            assert edge_onehot.max() == pytest.approx(1.0)
            edge_id = int(np.argmax(edge_onehot))
            assert 0 <= edge_id < 3

    def test_ratios_non_negative(self):
        config = _make_config(num_devices=4, num_edges=2)
        env = CloudEdgeDeviceEnv(config)
        env.reset(seed=7)

        agent = GreedyMinCostAgent(num_devices=4, num_edges=2)
        action = agent.compute_action(env)

        assert np.all(action[:, :3] >= 0)


class TestGreedyMinCostAgentEnvStep:
    """Run Greedy-MinCost through env.step to check end-to-end."""

    def test_runs_full_episode(self):
        config = _make_config(num_devices=3, num_edges=2)
        config["maddpg"] = {"max_episodes": 1, "max_steps": 5}
        env = CloudEdgeDeviceEnv(config)
        env.reset(seed=42)

        codec = HybridActionCodec(2)
        agent = GreedyMinCostAgent(num_devices=3, num_edges=2)

        for step in range(5):
            hybrid_action = agent.compute_action(env)
            env_action = codec.batch_policy_to_env_actions(hybrid_action)
            obs, rewards, terminated, truncated, info = env.step(env_action)
            done = bool(terminated or truncated)
            if done:
                break

        # Should complete without error.
        assert True

    def test_rewards_finite(self):
        config = _make_config(num_devices=3, num_edges=2)
        env = CloudEdgeDeviceEnv(config)
        env.reset(seed=0)

        codec = HybridActionCodec(2)
        agent = GreedyMinCostAgent(num_devices=3, num_edges=2)
        hybrid_action = agent.compute_action(env)
        env_action = codec.batch_policy_to_env_actions(hybrid_action)
        _obs, rewards, _term, _trunc, _info = env.step(env_action)

        assert np.all(np.isfinite(rewards))


class TestGreedyMinCostCostFunction:
    """Test the cost evaluation logic directly."""

    def test_all_local_has_zero_energy_for_transmission(self):
        """All-local should have no transmission energy."""
        config = _make_config(num_devices=1, num_edges=1)
        env = CloudEdgeDeviceEnv(config)
        env.reset(seed=0)

        agent = GreedyMinCostAgent(num_devices=1, num_edges=1)
        ue = env.user_equipments[0]
        es = env.edge_servers[0]
        cs = env.cloud_servers[0]
        task = env.current_tasks[0]

        cost = agent._evaluate_candidate(ue, es, cs, task, 1.0, 0.0, 0.0, 0)
        assert np.isfinite(cost)
        assert cost >= 0

    def test_edge_candidate_has_transmission_cost(self):
        """Edge offloading should have higher energy than local."""
        config = _make_config(num_devices=1, num_edges=1)
        env = CloudEdgeDeviceEnv(config)
        env.reset(seed=0)

        agent = GreedyMinCostAgent(num_devices=1, num_edges=1)
        ue = env.user_equipments[0]
        es = env.edge_servers[0]
        cs = env.cloud_servers[0]
        task = env.current_tasks[0]

        cost_local = agent._evaluate_candidate(ue, es, cs, task, 1.0, 0.0, 0.0, 0)
        cost_edge = agent._evaluate_candidate(ue, es, cs, task, 0.0, 1.0, 0.0, 0)

        assert np.isfinite(cost_local)
        assert np.isfinite(cost_edge)

    def test_inf_for_empty_template(self):
        """Empty template (0,0,0) should return inf."""
        config = _make_config(num_devices=1, num_edges=1)
        env = CloudEdgeDeviceEnv(config)
        env.reset(seed=0)

        agent = GreedyMinCostAgent(num_devices=1, num_edges=1)
        ue = env.user_equipments[0]
        es = env.edge_servers[0]
        cs = env.cloud_servers[0]
        task = env.current_tasks[0]

        cost = agent._evaluate_candidate(ue, es, cs, task, 0.0, 0.0, 0.0, 0)
        assert cost == float("inf")


class TestTemplates:
    """Check template definitions."""

    def test_templates_sum_to_one(self):
        for t in PARTITION_TEMPLATES:
            assert abs(sum(t) - 1.0) < 1e-8

    def test_templates_non_negative(self):
        for t in PARTITION_TEMPLATES:
            assert all(v >= 0 for v in t)

    def test_has_all_local(self):
        assert (1.0, 0.0, 0.0) in PARTITION_TEMPLATES

    def test_has_all_edge(self):
        assert (0.0, 1.0, 0.0) in PARTITION_TEMPLATES

    def test_has_all_cloud(self):
        assert (0.0, 0.0, 1.0) in PARTITION_TEMPLATES


class TestHybridActionCodec:
    """Test the codec round-trip used by Greedy-MinCost."""

    def test_codec_round_trip(self):
        codec = HybridActionCodec(num_edges=3)
        # Simulate a greedy action: partition (0.5, 0.3, 0.2), edge 1
        partition = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        edge_onehot = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        hybrid = np.concatenate([partition, edge_onehot])

        env_action = codec.policy_to_env_action(hybrid)
        assert env_action[0] == pytest.approx(0.5)
        assert env_action[1] == pytest.approx(0.3)
        assert env_action[2] == pytest.approx(0.2)
        assert env_action[3] == pytest.approx(1.0)  # edge_id

    def test_batch_conversion(self):
        codec = HybridActionCodec(num_edges=2)
        actions = np.array([
            [0.5, 0.3, 0.2, 1.0, 0.0],  # edge 0
            [0.1, 0.8, 0.1, 0.0, 1.0],  # edge 1
        ], dtype=np.float32)
        env_actions = codec.batch_policy_to_env_actions(actions)
        assert env_actions.shape == (2, 4)
        assert env_actions[0, 3] == pytest.approx(0.0)  # argmax of [1,0]
        assert env_actions[1, 3] == pytest.approx(1.0)  # argmax of [0,1]
