"""Prompt construction for cached or live legacy expert generation."""

from __future__ import annotations

import json


class PromptBuilder:
    VERSION = "phase1-v1"

    @staticmethod
    def build_offloading_strategy_prompt(
        env_state,
        device_info,
        edge_info,
        cloud_info,
        tasks_info,
    ) -> str:
        payload = {
            "prompt_version": PromptBuilder.VERSION,
            "devices": device_info,
            "edges": edge_info,
            "clouds": cloud_info,
            "tasks": tasks_info,
        }
        return (
            "You are a terminal-edge-cloud scheduling expert. Return JSON only. "
            "For every device, provide device_id, local_ratio, edge_ratio, "
            "cloud_ratio, and target_edge. Ratios must be non-negative and sum "
            "to one. Optimize latency and energy while respecting deadlines.\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )
