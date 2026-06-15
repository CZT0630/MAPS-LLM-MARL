"""Joint-state expert provider backed by a frozen formal cache."""

from __future__ import annotations

from typing import Any

import numpy as np

from LLM4RL.algos.common.hybrid_action import HybridActionCodec

from .expert_cache import ExpertCache, hash_state
from .expert_provider import ExpertBatch


class CachedExpertProvider:
    """Retrieve one joint action vector for one canonical joint state."""

    def __init__(
        self,
        cache: ExpertCache,
        num_agents: int,
        num_edges: int,
        fallback_policy: str = "all_local",
    ):
        if fallback_policy not in ("raise", "all_local"):
            raise ValueError(
                "fallback_policy must be 'raise' or 'all_local'"
            )
        self.cache = cache
        self.num_agents = int(num_agents)
        self.num_edges = int(num_edges)
        self.codec = HybridActionCodec(self.num_edges)
        self.fallback_policy = fallback_policy
        self.lookup_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self.fallback_action_count = 0

    def _fallback_action(self) -> np.ndarray:
        return self.codec.env_to_policy_action(
            [1.0, 0.0, 0.0, 0.0]
        )

    def _metadata(
        self,
        *,
        state_hash: str,
        cache_hit: bool | None,
        paper_evidence: bool,
        no_decision: bool = False,
    ) -> dict[str, Any]:
        return {
            "source": "state_keyed_cache",
            "state_hash": state_hash,
            "cache_hit": cache_hit,
            "no_decision": no_decision,
            "cache_size": len(self.cache),
            "cache_path": self.cache.source_path,
            "cache_sha256": self.cache.file_sha256,
            "cache_schema_version": self.cache.SCHEMA_VERSION,
            "cache_version": self.cache.SCHEMA_VERSION,
            "cache_stats": self.cache.stats(),
            "runtime_stats": self.runtime_stats(),
            "fallback_policy": self.fallback_policy,
            "prompt_version": self.cache.prompt_version,
            "provider": self.cache.provider,
            "requested_model": self.cache.requested_model,
            "state_key_version": self.cache.state_key_version,
            "paper_evidence": paper_evidence,
        }

    def get_noop_actions(self, state: Any) -> ExpertBatch:
        """Return masked all-local actions without counting a cache lookup."""
        state_hash = hash_state(state)
        actions = [
            self._fallback_action() for _ in range(self.num_agents)
        ]
        return ExpertBatch(
            policy_actions=np.asarray(actions, dtype=np.float32),
            valid_mask=np.zeros(self.num_agents, dtype=np.float32),
            metadata=self._metadata(
                state_hash=state_hash,
                cache_hit=None,
                paper_evidence=False,
                no_decision=True,
            ),
        )

    def get_actions(self, state: Any) -> ExpertBatch:
        state_hash = hash_state(state)
        entry = self.cache.get(state_hash)
        self.lookup_count += 1
        if entry is None:
            self.miss_count += 1
            if self.fallback_policy == "raise":
                raise KeyError(f"state_hash {state_hash!r} not in cache")
            actions = [
                self._fallback_action() for _ in range(self.num_agents)
            ]
            valid_mask = [False] * self.num_agents
            self.fallback_action_count += self.num_agents
            cache_hit = False
        else:
            self.hit_count += 1
            cache_hit = True
            if len(entry.parsed_action) != self.num_agents:
                raise ValueError(
                    "cached joint action count does not match num_agents"
                )
            if len(entry.valid_mask) != self.num_agents:
                raise ValueError(
                    "cached valid_mask count does not match num_agents"
                )
            actions = []
            valid_mask = []
            for action_data, valid in zip(
                entry.parsed_action, entry.valid_mask
            ):
                if valid:
                    env_action = [
                        action_data["partition"]["local"],
                        action_data["partition"]["edge"],
                        action_data["partition"]["cloud"],
                        float(action_data["edge_id"]),
                    ]
                    actions.append(
                        self.codec.env_to_policy_action(env_action)
                    )
                    valid_mask.append(True)
                elif self.fallback_policy == "raise":
                    raise KeyError(
                        f"state_hash {state_hash!r} contains fallback actions"
                    )
                else:
                    actions.append(self._fallback_action())
                    valid_mask.append(False)
                    self.fallback_action_count += 1

        return ExpertBatch(
            policy_actions=np.asarray(actions, dtype=np.float32),
            valid_mask=np.asarray(valid_mask, dtype=np.float32),
            metadata=self._metadata(
                state_hash=state_hash,
                cache_hit=cache_hit,
                paper_evidence=bool(
                    entry is not None and entry.paper_evidence
                ),
            ),
        )

    def get_actions_from_obs(self, observations: np.ndarray) -> ExpertBatch:
        """Compatibility wrapper; observations are treated as one joint state."""
        return self.get_actions(observations)

    def runtime_stats(self) -> dict[str, float | int]:
        total_actions = self.lookup_count * self.num_agents
        return {
            "lookups": self.lookup_count,
            "hits": self.hit_count,
            "misses": self.miss_count,
            "hit_rate": (
                self.hit_count / self.lookup_count
                if self.lookup_count
                else 0.0
            ),
            "fallback_actions": self.fallback_action_count,
            "fallback_rate": (
                self.fallback_action_count / total_actions
                if total_actions
                else 0.0
            ),
        }
