import numpy as np
import torch
from torch import nn

from LLM4RL.algos.maddpg.maddpg_agent import MADDPGAgent


class ConstantActor(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer("value", torch.tensor(value, dtype=torch.float32))

    def forward(self, state):
        return self.value.expand(state.shape[0], -1)


def make_agents(num_edges=3):
    return [
        MADDPGAgent(
            state_dim=5,
            action_dim=4,
            num_agents=3,
            agent_idx=index,
            num_edges=num_edges,
            config={"noise_sigma": 0.0},
        )
        for index in range(3)
    ]


def test_target_joint_actions_use_each_corresponding_agent():
    agents = make_agents()
    agents[0].actor_target = ConstantActor([0.1, 0.2, 0.7, 0.0])
    agents[1].actor_target = ConstantActor([0.2, 0.3, 0.5, 0.5])
    agents[2].actor_target = ConstantActor([0.3, 0.3, 0.4, 1.0])
    next_states = torch.zeros((2, 3, 5))

    actions = agents[0]._target_joint_actions(next_states, agents)

    assert actions.shape == (2, 3, 4)
    assert torch.allclose(actions[0, 0], torch.tensor([0.1, 0.2, 0.7, 0.0]))
    assert torch.allclose(actions[0, 1], torch.tensor([0.2, 0.3, 0.5, 0.5]))
    assert torch.allclose(actions[0, 2], torch.tensor([0.3, 0.3, 0.4, 1.0]))


def test_edge_decoding_uses_configured_edge_count():
    agent = make_agents(num_edges=7)[0]
    assert agent.policy_to_env_action([0.2, 0.3, 0.5, 0.0])[3] == 0
    assert agent.policy_to_env_action([0.2, 0.3, 0.5, 0.51])[3] == 3
    assert agent.policy_to_env_action([0.2, 0.3, 0.5, 1.0])[3] == 6


def test_joint_update_produces_finite_losses():
    agents = make_agents()
    batch_size = 4
    batch = {
        "states": np.random.rand(batch_size, 3, 5).astype(np.float32),
        "actions": np.random.rand(batch_size, 3, 4).astype(np.float32),
        "rewards": np.random.rand(batch_size, 3).astype(np.float32),
        "next_states": np.random.rand(batch_size, 3, 5).astype(np.float32),
        "done": np.zeros(batch_size, dtype=np.float32),
        "expert_actions": np.zeros((batch_size, 3, 4), dtype=np.float32),
        "expert_mask": np.zeros((batch_size, 3), dtype=np.float32),
    }

    losses = agents[0].update(batch, agents)

    assert all(np.isfinite(value) for value in losses.values())
