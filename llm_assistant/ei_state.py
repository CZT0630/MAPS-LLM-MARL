"""Canonical EI state extraction shared by cache generation and evaluation."""

from __future__ import annotations

from typing import Any

from .expert_cache import hash_state


_ENVIRONMENT_FINGERPRINT_KEYS = (
    "model_version",
    "environment",
    "device_specs",
    "network",
    "channel",
    "backhaul",
    "energy",
    "tasks",
)


def ei_environment_fingerprint(config: dict[str, Any]) -> str:
    """Hash only configuration that can change EI state trajectories."""
    payload = {
        key: config[key]
        for key in _ENVIRONMENT_FINGERPRINT_KEYS
        if key in config
    }
    return hash_state(payload)


def _queue_load(node: Any) -> float:
    if hasattr(node, "calculate_task_load"):
        return float(node.calculate_task_load())
    current = getattr(node, "current_execution", None)
    current_cycles = (
        float(current.remaining_cycles)
        if current is not None and not getattr(current, "completed", False)
        else 0.0
    )
    queued_cycles = sum(
        float(getattr(task, "remaining_cycles", 0.0))
        for task in getattr(node, "task_queue", [])
        if not getattr(task, "completed", False)
    )
    return current_cycles + queued_cycles


def _ue_rates(env: Any, ue: Any) -> list[float]:
    if hasattr(env, "channel") and hasattr(ue, "distance_to_edges"):
        return [
            float(env.channel.achievable_rate(distance))
            for distance in ue.distance_to_edges
        ]
    configured_rate = float(
        env.config.get("network", {}).get("ue_to_edge_rate", 0.0)
    )
    return [configured_rate] * env.num_edges


def build_ei_state_payload(env: Any) -> dict[str, Any]:
    """Return the exact joint state payload used for hashing and prompting."""
    ue_states = []
    tasks = []
    for agent_idx, ue in enumerate(env.user_equipments):
        ue_states.append(
            {
                "device_id": agent_idx,
                "cpu_frequency": float(ue.cpu_frequency),
                "pending_queue_size": len(env.pending_task_queues[agent_idx]),
                "queue_load": _queue_load(ue),
                "ue_to_edge_rates": _ue_rates(env, ue),
            }
        )
        task = (
            env.current_tasks[agent_idx]
            if env.current_tasks and agent_idx < len(env.current_tasks)
            else None
        )
        if task is None:
            tasks.append(None)
            continue
        elapsed = max(float(env.global_time) - float(task.arrival_time), 0.0)
        tasks.append(
            {
                "task_id": str(task.task_id),
                "device_id": agent_idx,
                "data_size": float(task.task_data_size),
                "cpu_cycles": float(task.task_workload),
                "deadline_slack": max(float(task.deadline) - elapsed, 0.0),
                "semantic_type": str(task.semantic_type),
                "priority": int(task.priority),
                "output_ratio": float(task.output_ratio),
            }
        )

    es_states = [
        {
            "server_id": index,
            "cpu_frequency": float(server.cpu_frequency),
            "queue_load": _queue_load(server),
        }
        for index, server in enumerate(env.edge_servers)
    ]
    cs_states = [
        {
            "server_id": index,
            "cpu_frequency": float(server.cpu_frequency),
            "queue_load": _queue_load(server),
            "parallel_factor": float(
                getattr(server, "parallel_factor", 1.0)
            ),
        }
        for index, server in enumerate(env.cloud_servers)
    ]
    if hasattr(env, "backhaul"):
        backhaul = env.backhaul.to_dict()
    else:
        network = env.config.get("network", {})
        backhaul = {
            "rate_bps": float(network.get("edge_to_cloud_rate", 0.0)),
            "energy_per_bit": None,
            "propagation_latency_s": None,
        }

    return {
        "state_key_version": "ei-state-v1",
        "physics_version": int(getattr(env, "physics_version", 1)),
        "global_time": float(env.global_time),
        "num_edges": int(env.num_edges),
        "ue_states": ue_states,
        "es_states": es_states,
        "cs_states": cs_states,
        "backhaul": backhaul,
        "tasks": tasks,
    }
