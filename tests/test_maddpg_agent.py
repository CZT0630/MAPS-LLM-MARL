import numpy as np
import torch
from torch import nn

from LLM4RL.algos.maddpg.maddpg_agent import MADDPGAgent


class ConstantActor(nn.Module):
    """Mock actor that returns fixed probabilities [3+E]."""
    def __init__(self, num_edges, partition, edge_id):
        super().__init__()
        # Build a fixed probability vector
        edge_onehot = [0.0] * num_edges
        edge_onehot[edge_id] = 1.0
        value = partition + edge_onehot
        self.register_buffer("value", torch.tensor(value, dtype=torch.float32))

    def forward(self, state):
        batch = state.shape[0] if state.dim() > 1 else 1
        return self.value.unsqueeze(0).expand(batch, -1)


NUM_EDGES = 5
ACTION_DIM = 3 + NUM_EDGES  # hybrid action


def make_agents(num_edges=NUM_EDGES):
    return [
        MADDPGAgent(
            state_dim=5,
            action_dim=3 + num_edges,
            num_agents=3,
            agent_idx=index,
            num_edges=num_edges,
            config={"noise_sigma": 0.0},
        )
        for index in range(3)
    ]


def test_target_joint_actions_use_each_corresponding_agent():
    agents = make_agents()
    agents[0].actor_target = ConstantActor(NUM_EDGES, [0.1, 0.2, 0.7], 0)
    agents[1].actor_target = ConstantActor(NUM_EDGES, [0.2, 0.3, 0.5], 2)
    agents[2].actor_target = ConstantActor(NUM_EDGES, [0.3, 0.3, 0.4], 4)
    next_states = torch.zeros((2, 3, 5))

    actions = agents[0]._target_joint_actions(next_states, agents)

    assert actions.shape == (2, 3, ACTION_DIM)
    # agent 0: partition [0.1, 0.2, 0.7], edge 0 one-hot
    assert torch.allclose(actions[0, 0, :3], torch.tensor([0.1, 0.2, 0.7]))
    assert actions[0, 0, 3] == 1.0  # edge 0 active
    assert actions[0, 0, 4:].sum() == 0.0  # rest zero


def test_policy_to_env_action_uses_codec():
    agent = make_agents(num_edges=7)[0]
    # partition [0.2, 0.3, 0.5], edge 3 one-hot
    edge_onehot = [0.0] * 7
    edge_onehot[3] = 1.0
    policy = [0.2, 0.3, 0.5] + edge_onehot
    env_action = agent.policy_to_env_action(policy)
    assert env_action[3] == 3


def test_select_action_shape():
    agent = make_agents()[0]
    state = np.random.randn(5).astype(np.float32)
    action = agent.select_action(state, add_noise=False)
    assert action.shape == (ACTION_DIM,)
    # First 3 should be partition (sum to 1)
    assert abs(action[:3].sum() - 1.0) < 1e-5
    # Last E should be edge probs (sum to 1)
    assert abs(action[3:].sum() - 1.0) < 1e-5


def test_joint_update_produces_finite_losses():
    agents = make_agents()
    batch_size = 4
    batch = {
        "states": np.random.rand(batch_size, 3, 5).astype(np.float32),
        "actions": np.random.rand(batch_size, 3, ACTION_DIM).astype(np.float32),
        "rewards": np.random.rand(batch_size, 3).astype(np.float32),
        "next_states": np.random.rand(batch_size, 3, 5).astype(np.float32),
        "done": np.zeros(batch_size, dtype=np.float32),
        "expert_actions": np.zeros((batch_size, 3, ACTION_DIM), dtype=np.float32),
        "expert_mask": np.zeros((batch_size, 3), dtype=np.float32),
    }

    losses = agents[0].update(batch, agents)

    assert all(np.isfinite(value) for value in losses.values())
