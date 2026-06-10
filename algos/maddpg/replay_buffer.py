"""Joint replay buffer for centralized-training multi-agent algorithms."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


class JointReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=int(capacity))
        self._state_shape: tuple[int, ...] | None = None
        self._action_shape: tuple[int, ...] | None = None

    def add(
        self,
        states,
        actions,
        rewards,
        next_states,
        done,
        expert_actions=None,
        expert_mask=None,
    ) -> None:
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)

        if states.ndim != 2:
            raise ValueError(f"states must have shape [agents, state_dim], got {states.shape}")
        if actions.ndim != 2:
            raise ValueError(f"actions must have shape [agents, action_dim], got {actions.shape}")
        if next_states.shape != states.shape:
            raise ValueError(
                f"next_states shape {next_states.shape} does not match states {states.shape}"
            )
        if rewards.shape != (states.shape[0],):
            raise ValueError(
                f"rewards must have shape [{states.shape[0]}], got {rewards.shape}"
            )
        if actions.shape[0] != states.shape[0]:
            raise ValueError("state and action agent counts differ")

        if self._state_shape is None:
            self._state_shape = states.shape
            self._action_shape = actions.shape
        elif states.shape != self._state_shape or actions.shape != self._action_shape:
            raise ValueError(
                "joint transition shape changed: "
                f"states {states.shape}/{self._state_shape}, "
                f"actions {actions.shape}/{self._action_shape}"
            )

        if expert_actions is None:
            expert_actions_array = np.zeros_like(actions)
            expert_mask_array = np.zeros(states.shape[0], dtype=np.float32)
        else:
            expert_actions_array = np.asarray(expert_actions, dtype=np.float32)
            if expert_actions_array.shape != actions.shape:
                raise ValueError(
                    f"expert_actions shape {expert_actions_array.shape} "
                    f"does not match actions {actions.shape}"
                )
            if expert_mask is None:
                expert_mask_array = np.ones(states.shape[0], dtype=np.float32)
            else:
                expert_mask_array = np.asarray(expert_mask, dtype=np.float32)
                if expert_mask_array.shape != (states.shape[0],):
                    raise ValueError(
                        f"expert_mask must have shape [{states.shape[0]}], "
                        f"got {expert_mask_array.shape}"
                    )

        self.buffer.append(
            {
                "states": states.copy(),
                "actions": actions.copy(),
                "rewards": rewards.copy(),
                "next_states": next_states.copy(),
                "done": float(bool(done)),
                "expert_actions": expert_actions_array.copy(),
                "expert_mask": expert_mask_array.copy(),
            }
        )

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size > len(self.buffer):
            raise ValueError(f"cannot sample {batch_size} transitions from {len(self.buffer)}")
        samples = random.sample(self.buffer, int(batch_size))
        return {
            key: np.asarray([sample[key] for sample in samples], dtype=np.float32)
            for key in samples[0]
        }

    def __len__(self) -> int:
        return len(self.buffer)


ReplayBuffer = JointReplayBuffer
