"""EI-v1 prompt builder for state-conditioned LLM expert generation.

The prompt encodes per-UE task semantics, per-UE/per-ES queue state,
channel summaries, and output constraints.  It is designed for the
MiMo-V2.5 model used in the EI submission.
"""

from __future__ import annotations

import json
from typing import Any


class EIPromptBuilder:
    """Build the EI-v1 expert prompt from environment observations.

    The prompt version ``ei-v1`` is frozen once the cache generation
    script starts collecting formal expert data.  Any prompt change
    that could alter the LLM output MUST bump this version.
    """

    VERSION = "ei-v1"

    def build(
        self,
        *,
        ue_info: list[dict[str, Any]],
        es_info: list[dict[str, Any]],
        cs_info: list[dict[str, Any]],
        tasks_info: list[dict[str, Any]],
        backhaul_info: dict[str, Any],
        num_edges: int,
    ) -> str:
        """Return the full user-role prompt string.

        Parameters
        ----------
        ue_info : list[dict]
            Per-UE observations.  Each dict contains at least
            ``device_id``, ``cpu_frequency``, ``pending_queue_size``,
            and ``ue_to_edge_rates`` (list[float], length *num_edges*).
        es_info : list[dict]
            Per-ES observations.  Each dict contains at least
            ``server_id``, ``cpu_frequency``, ``queue_load``.
        cs_info : list[dict]
            Per-CS observations.  Each dict contains ``server_id``,
            ``cpu_frequency``.
        tasks_info : list[dict]
            Per-UE task (or None).  Each dict contains at least
            ``task_id``, ``device_id``, ``data_size``, ``cpu_cycles``,
            ``deadline_slack``, ``semantic_type``, ``priority``,
            ``output_ratio``.
        num_edges : int
            Number of edge servers; constrains ``edge_id`` choices.
        """
        payload = self.build_payload(
            ue_info=ue_info,
            es_info=es_info,
            cs_info=cs_info,
            tasks_info=tasks_info,
            backhaul_info=backhaul_info,
            num_edges=num_edges,
        )
        return self.build_from_payload(payload)

    def build_payload(
        self,
        *,
        ue_info: list[dict[str, Any]],
        es_info: list[dict[str, Any]],
        cs_info: list[dict[str, Any]],
        tasks_info: list[dict[str, Any] | None],
        backhaul_info: dict[str, Any],
        num_edges: int,
    ) -> dict[str, Any]:
        if num_edges <= 0:
            raise ValueError("num_edges must be positive")
        return {
            "prompt_version": self.VERSION,
            "num_edges": num_edges,
            "ue_states": ue_info,
            "es_states": es_info,
            "cs_states": cs_info,
            "backhaul": backhaul_info,
            "tasks": tasks_info,
            "constraints": {
                "partition_ratios": "non-negative, sum to 1",
                "edge_id_range": f"[0, {num_edges - 1}]",
                "numeric_fields": (
                    "ue_id, edge_id, local, edge, and cloud must be JSON "
                    "numbers, not quoted strings"
                ),
                "required_actions": (
                    "return one action for every non-null task; omit UEs "
                    "whose task entry is null"
                ),
                "json_only": "no Markdown fences, prose, or comments",
            },
            "output_schema": {
                "schema_version": "ei-v1",
                "actions": [
                    {
                        "ue_id": 0,
                        "partition": {
                            "local": 0.2,
                            "edge": 0.6,
                            "cloud": 0.2,
                        },
                        "edge_id": 1,
                        "reason_code": "<string>",
                    }
                ],
            },
        }

    def build_from_payload(self, payload: dict[str, Any]) -> str:
        num_edges = int(payload["num_edges"])
        max_edge = num_edges - 1
        payload_str = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return (
            "You are a terminal-edge-cloud scheduling expert.\n"
            "For every UE that currently has a task, provide a JSON action.\n"
            "Each action must include:\n"
            '  - "ue_id": integer device id\n'
            '  - "partition": {"local", "edge", "cloud"}, non-negative, sum to 1\n'
            f'  - "edge_id": integer in [0, {max_edge}]\n'
            '  - "reason_code": short reason for this decision\n'
            "All numeric fields must be JSON numbers, not strings.\n"
            "Omit UEs whose task entry is null.\n"
            "Optimize task latency and device energy while respecting deadlines.\n"
            "Return JSON only, without Markdown fences or explanatory prose.\n\n"
            + payload_str
        )
