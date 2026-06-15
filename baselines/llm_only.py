"""LLM-only evaluator using the same frozen cache as MAPS."""

from __future__ import annotations

from typing import Any

import numpy as np

from LLM4RL.algos.common.hybrid_action import HybridActionCodec
from LLM4RL.llm_assistant.cached_expert_provider import CachedExpertProvider
from LLM4RL.llm_assistant.expert_cache import ExpertCache
from LLM4RL.llm_assistant.expert_provider import ExpertBatch


class LLMOnlyAgent:
    """Apply cached joint LLM actions without RL training or online API calls."""

    def __init__(
        self,
        cache: ExpertCache,
        num_agents: int,
        num_edges: int,
        fallback_policy: str = "all_local",
    ):
        self.num_agents = int(num_agents)
        self.num_edges = int(num_edges)
        self.codec = HybridActionCodec(self.num_edges)
        self.provider = CachedExpertProvider(
            cache=cache,
            num_agents=self.num_agents,
            num_edges=self.num_edges,
            fallback_policy=fallback_policy,
        )

    def select_joint_actions(self, state: Any) -> ExpertBatch:
        return self.provider.get_actions(state)

    def policy_to_env_actions(
        self, policy_actions: np.ndarray
    ) -> np.ndarray:
        return self.codec.batch_policy_to_env_actions(policy_actions)

    @property
    def runtime_stats(self) -> dict[str, float | int]:
        return self.provider.runtime_stats()
