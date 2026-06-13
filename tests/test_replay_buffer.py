import numpy as np
import pytest

from LLM4RL.algos.maddpg.replay_buffer import JointReplayBuffer


def test_joint_replay_buffer_preserves_agent_axes():
    buffer = JointReplayBuffer(10, num_edges=5)
    num_edges = 5
    action_dim = 3 + num_edges  # hybrid action
    states = np.zeros((3, 7), dtype=np.float32)
    actions = np.tile(
        np.array(
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        (3, 1),
    )
    rewards = np.arange(3, dtype=np.float32)
    buffer.add(states, actions, rewards, states + 1, False)

    batch = buffer.sample(1)

    assert batch["states"].shape == (1, 3, 7)
    assert batch["actions"].shape == (1, 3, action_dim)
    assert batch["rewards"].shape == (1, 3)
    assert batch["expert_actions"].shape == (1, 3, action_dim)
    assert batch["expert_mask"].shape == (1, 3)


def test_joint_replay_buffer_rejects_single_agent_transition():
    buffer = JointReplayBuffer(10, num_edges=5)
    with pytest.raises(ValueError):
        buffer.add(
            states=np.zeros(7, dtype=np.float32),
            actions=np.zeros(8, dtype=np.float32),
            rewards=np.zeros(1, dtype=np.float32),
            next_states=np.zeros(7, dtype=np.float32),
            done=False,
        )


@pytest.mark.parametrize("kind", ["nan", "invalid_probability"])
def test_joint_replay_buffer_rejects_invalid_hybrid_actions(kind):
    buffer = JointReplayBuffer(10, num_edges=3)
    states = np.zeros((2, 5), dtype=np.float32)
    actions = np.tile(
        np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        (2, 1),
    )
    if kind == "nan":
        actions[0, 0] = np.nan
    else:
        actions[0, :3] = [0.8, 0.8, 0.0]

    with pytest.raises(ValueError):
        buffer.add(
            states=states,
            actions=actions,
            rewards=np.zeros(2, dtype=np.float32),
            next_states=states,
            done=False,
        )


def test_joint_replay_buffer_rejects_invalid_expert_actions():
    buffer = JointReplayBuffer(10, num_edges=3)
    states = np.zeros((2, 5), dtype=np.float32)
    actions = np.tile(
        np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        (2, 1),
    )
    expert_actions = np.zeros_like(actions)

    with pytest.raises(ValueError):
        buffer.add(
            states=states,
            actions=actions,
            rewards=np.zeros(2, dtype=np.float32),
            next_states=states,
            done=False,
            expert_actions=expert_actions,
            expert_mask=np.ones(2, dtype=np.float32),
        )


def test_joint_replay_buffer_rejects_mask_without_expert_actions():
    buffer = JointReplayBuffer(10, num_edges=3)
    states = np.zeros((2, 5), dtype=np.float32)
    actions = np.tile(
        np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        (2, 1),
    )

    with pytest.raises(ValueError):
        buffer.add(
            states=states,
            actions=actions,
            rewards=np.zeros(2, dtype=np.float32),
            next_states=states,
            done=False,
            expert_mask=np.ones(2, dtype=np.float32),
        )
