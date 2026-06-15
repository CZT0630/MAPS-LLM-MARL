"""Expert providers used by the Phase 1 Legacy MAPS baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from LLM4RL.algos.common.hybrid_action import HybridActionCodec

from .response_parser import ParseMetadata, ResponseParser


@dataclass(frozen=True)
class ExpertBatch:
    # policy_actions: shape (num_agents, 3+E)
    #   First 3 cols: partition ratios (local, edge, cloud)
    #   Last E cols: one-hot edge server selection
    policy_actions: np.ndarray
    valid_mask: np.ndarray
    metadata: dict


class FixedCacheExpertProvider:
    """Loads a fixed expert response for deterministic engineering smoke tests."""

    def __init__(self, cache_path: Path, num_agents: int, num_edges: int):
        self.cache_path = Path(cache_path)
        self.num_agents = int(num_agents)
        self.num_edges = int(num_edges)
        self.codec = HybridActionCodec(self.num_edges)
        cache_bytes = self.cache_path.read_bytes()
        payload = json.loads(cache_bytes.decode("utf-8"))
        self.source = payload.get("source", "unknown_cache")
        self.paper_evidence = bool(payload.get("paper_evidence", False))
        self.cache_schema_version = str(payload.get("schema_version", "unknown"))
        self.cache_version = str(
            payload.get("cache_version", self.cache_schema_version)
        )
        self.prompt_version = str(payload.get("prompt_version", "unknown"))
        self.cache_sha256 = hashlib.sha256(cache_bytes).hexdigest()
        self.raw_response = payload.get("default", payload)
        self.strategies, self.parse_metadata = ResponseParser.parse_with_metadata(
            self.raw_response,
            num_devices=self.num_agents,
            num_edges=self.num_edges,
        )

    @property
    def metadata(self) -> dict:
        return {
            "source": self.source,
            "cache_path": str(self.cache_path),
            "cache_schema_version": self.cache_schema_version,
            "cache_version": self.cache_version,
            "cache_sha256": self.cache_sha256,
            "prompt_version": self.prompt_version,
            "paper_evidence": self.paper_evidence,
            "parse_valid": self.parse_metadata.valid,
            "fallback_count": self.parse_metadata.fallback_count,
            "valid_mask": list(self.parse_metadata.valid_mask),
        }

    def get_actions(self, episode: int, step: int) -> ExpertBatch:
        del episode, step
        actions = []
        for strategy in self.strategies:
            env_action = [
                strategy["local_ratio"],
                strategy["edge_ratio"],
                strategy["cloud_ratio"],
                strategy["target_edge"],
            ]
            actions.append(self.codec.env_to_policy_action(env_action))
        return ExpertBatch(
            policy_actions=np.asarray(actions, dtype=np.float32),
            valid_mask=np.asarray(
                self.parse_metadata.valid_mask, dtype=np.float32
            ),
            metadata=self.metadata,
        )
