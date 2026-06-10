"""Expert providers used by the Phase 1 Legacy MAPS baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .response_parser import ParseMetadata, ResponseParser


@dataclass(frozen=True)
class ExpertBatch:
    policy_actions: np.ndarray
    valid_mask: np.ndarray
    metadata: dict


class FixedCacheExpertProvider:
    """Loads a fixed expert response for deterministic engineering smoke tests."""

    def __init__(self, cache_path: Path, num_agents: int, num_edges: int):
        self.cache_path = Path(cache_path)
        self.num_agents = int(num_agents)
        self.num_edges = int(num_edges)
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.source = payload.get("source", "unknown_cache")
        self.paper_evidence = bool(payload.get("paper_evidence", False))
        self.raw_response = payload.get("default", payload)
        self.strategies, self.parse_metadata = ResponseParser.parse_with_metadata(
            self.raw_response,
            num_devices=self.num_agents,
            num_edges=self.num_edges,
        )

    def get_actions(self, episode: int, step: int) -> ExpertBatch:
        del episode, step
        actions = []
        for strategy in self.strategies:
            denominator = max(self.num_edges - 1, 1)
            actions.append(
                [
                    strategy["local_ratio"],
                    strategy["edge_ratio"],
                    strategy["cloud_ratio"],
                    strategy["target_edge"] / denominator,
                ]
            )
        valid = self.parse_metadata.valid
        return ExpertBatch(
            policy_actions=np.asarray(actions, dtype=np.float32),
            valid_mask=np.full(self.num_agents, float(valid), dtype=np.float32),
            metadata={
                "source": self.source,
                "cache_path": str(self.cache_path),
                "paper_evidence": self.paper_evidence,
                "parse_valid": valid,
                "fallback_count": self.parse_metadata.fallback_count,
            },
        )
