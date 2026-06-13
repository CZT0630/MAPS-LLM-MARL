"""Greedy-MinCost baseline — 一步贪心启发式调度。

Algorithm (per step):
  1. Sort active UEs by deadline slack (ascending).
  2. For each UE, enumerate (partition_template, edge_id) candidates.
  3. Evaluate each candidate's cost using Phase-1 physics.
  4. Select the candidate with minimum cost.
  5. Return the joint action for all UEs.

Cost function (consistent with env reward_type='normalized'):
  J = w_T * (1 - latency / baseline_latency)
    + w_E * (1 - energy / baseline_energy)
  Lower cost ⟹ higher reward.

This module is self-contained; it does NOT modify environment state.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


# Standard partition templates: [alpha1_local, alpha2_edge, alpha3_cloud]
PARTITION_TEMPLATES: list[tuple[float, float, float]] = [
    (1.0, 0.0, 0.0),   # all local
    (0.0, 1.0, 0.0),   # all edge
    (0.0, 0.0, 1.0),   # all cloud
    (0.5, 0.5, 0.0),   # half local + half edge
    (0.5, 0.0, 0.5),   # half local + half cloud
    (0.0, 0.5, 0.5),   # half edge + half cloud
    (1.0 / 3, 1.0 / 3, 1.0 / 3),  # equal split
]


class GreedyMinCostAgent:
    """Greedy-MinCost: one-step myopic scheduling heuristic.

    Parameters
    ----------
    num_devices : int
        Number of UEs.
    num_edges : int
        Number of edge servers.
    latency_weight : float
        Weight for latency term in cost.
    energy_weight : float
        Weight for energy term in cost.
    templates : list[tuple[float, float, float]] | None
        Partition templates to enumerate.  Defaults to PARTITION_TEMPLATES.
    """

    def __init__(
        self,
        num_devices: int,
        num_edges: int,
        latency_weight: float = 1.0,
        energy_weight: float = 1.0,
        templates: list[tuple[float, float, float]] | None = None,
    ) -> None:
        self.num_devices = num_devices
        self.num_edges = num_edges
        self.latency_weight = latency_weight
        self.energy_weight = energy_weight
        self.templates = templates or PARTITION_TEMPLATES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_action(
        self,
        env: Any,
    ) -> np.ndarray:
        """Compute joint action for the current environment step.

        Returns
        -------
        np.ndarray
            Joint action of shape ``(num_devices, 3 + num_edges)``.
            First 3 cols: partition ratios.
            Last num_edges cols: one-hot edge server selection.
        """
        tasks = env.current_tasks
        action_dim = 3 + self.num_edges
        joint_action = np.zeros((self.num_devices, action_dim), dtype=np.float32)
        # Default: all local, edge 0.
        for i in range(self.num_devices):
            joint_action[i] = (
                [1.0, 0.0, 0.0]
                + [1.0]
                + [0.0] * (self.num_edges - 1)
            )
        if tasks is None:
            return joint_action

        # Build (ue_idx, deadline_slack) for active UEs.
        slack_list: list[tuple[int, float]] = []
        for idx, task in enumerate(tasks):
            if task is None:
                continue
            slack = max(task.deadline - env.global_time, 0.0)
            slack_list.append((idx, slack))
        # Most urgent first.
        slack_list.sort(key=lambda x: x[1])

        cloud_server = env.cloud_servers[0]

        for ue_idx, _slack in slack_list:
            task = tasks[ue_idx]
            ue = env.user_equipments[ue_idx]
            best_cost = float("inf")
            best_partition = [1.0, 0.0, 0.0]
            best_edge_id = 0

            for alpha1, alpha2, alpha3 in self.templates:
                for edge_id in range(self.num_edges):
                    es = env.edge_servers[edge_id]
                    cost = self._evaluate_candidate(
                        ue, es, cloud_server, task, alpha1, alpha2, alpha3, edge_id
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_partition = [alpha1, alpha2, alpha3]
                        best_edge_id = edge_id

            # Build one-hot edge vector
            edge_onehot = [0.0] * self.num_edges
            edge_onehot[best_edge_id] = 1.0
            joint_action[ue_idx] = best_partition + edge_onehot

        return joint_action

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate_candidate(
        self,
        ue: Any,
        es: Any,
        cs: Any,
        task: Any,
        alpha1: float,
        alpha2: float,
        alpha3: float,
        edge_id: int,
    ) -> float:
        """Compute cost for a single (template, edge) candidate.

        Uses Phase-1 physics formulas that match
        ``CloudEdgeDeviceEnv._schedule_task_execution_optimized``.
        """
        total_cycles = task.task_workload
        total_data_mb = task.task_data_size

        # --- All-local baseline ---
        baseline_latency = ue.calculate_execution_time(total_cycles)
        baseline_energy = ue.calculate_energy_consumption(total_cycles)
        # Guard against zero baseline.
        if baseline_latency <= 0:
            baseline_latency = 1e-8
        if baseline_energy <= 0:
            baseline_energy = 1e-8

        # --- Candidate latency / energy (parallel branches) ---
        branch_latencies: list[float] = []
        branch_energies: list[float] = []

        # Local branch
        if alpha1 > 0:
            local_cycles = total_cycles * alpha1
            local_lat = ue.calculate_execution_time(local_cycles)
            local_eng = ue.calculate_energy_consumption(local_cycles)
            branch_latencies.append(local_lat)
            branch_energies.append(local_eng)

        # Edge branch
        if alpha2 > 0:
            edge_data_mb = total_data_mb * alpha2
            edge_cycles = total_cycles * alpha2
            tx_time = ue.calculate_transmission_time_to_edge(edge_data_mb)
            tx_energy = ue.calculate_transmission_energy(tx_time)
            exec_time = es.calculate_execution_time(edge_cycles)
            branch_latencies.append(tx_time + exec_time)
            branch_energies.append(tx_energy)

        # Cloud branch
        if alpha3 > 0:
            cloud_data_mb = total_data_mb * alpha3
            cloud_cycles = total_cycles * alpha3
            tx_time = ue.calculate_transmission_time_to_cloud(cloud_data_mb)
            tx_energy = ue.calculate_transmission_energy(tx_time)
            exec_time = cs.calculate_execution_time(cloud_cycles)
            branch_latencies.append(tx_time + exec_time)
            branch_energies.append(tx_energy)

        if not branch_latencies:
            return float("inf")

        # Parallel branches: total latency = max branch latency.
        total_latency = max(branch_latencies)
        total_energy = sum(branch_energies)

        # Cost (lower is better): reward-like formulation.
        lat_ratio = total_latency / baseline_latency
        eng_ratio = total_energy / baseline_energy
        cost = self.latency_weight * lat_ratio + self.energy_weight * eng_ratio

        # Deadline penalty.
        if task.deadline and total_latency > task.deadline:
            cost += 10.0 * (total_latency - task.deadline)

        return cost
