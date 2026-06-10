import numpy as np
import pytest

from LLM4RL.algos.maddpg.replay_buffer import JointReplayBuffer


def test_joint_replay_buffer_preserves_agent_axes():
    buffer = JointReplayBuffer(10)
    states = np.zeros((3, 7), dtype=np.float32)
    actions = np.zeros((3, 4), dtype=np.float32)
    rewards = np.arange(3, dtype=np.float32)
    buffer.add(states, actions, rewards, states + 1, False)

    batch = buffer.sample(1)

    assert batch["states"].shape == (1, 3, 7)
    assert batch["actions"].shape == (1, 3, 4)
    assert batch["rewards"].shape == (1, 3)
    assert batch["expert_actions"].shape == (1, 3, 4)
    assert batch["expert_mask"].shape == (1, 3)


def test_joint_replay_buffer_rejects_single_agent_transition():
    buffer = JointReplayBuffer(10)
    with pytest.raises(ValueError):
        buffer.add(
            states=np.zeros(7, dtype=np.float32),
            actions=np.zeros(4, dtype=np.float32),
            rewards=np.zeros(1, dtype=np.float32),
            next_states=np.zeros(7, dtype=np.float32),
            done=False,
        )
