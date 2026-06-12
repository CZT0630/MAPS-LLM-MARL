# environment/snapshot.py
"""
环境快照与纯函数式动作评估 — Phase 2 实现

提供无副作用的 ``evaluate`` 接口，用于：
  - ConstraintVerifier 一步预测
  - 反事实动作评估
  - 候选动作比较

NOTE: 本文件由人工完成 Phase 2 初稿，已由 Codex 于 2026-06-11 审查并修复。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ======================================================================
# 快照数据结构
# ======================================================================

@dataclass
class QueueSnapshot:
    """单个节点的队列快照。"""
    pending_cycles: float = 0.0       # 队列中等待的总 CPU cycles
    current_remaining_cycles: float = 0.0  # 当前执行任务剩余 cycles
    num_pending_tasks: int = 0
    available_at: float = 0.0
    jobs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class UESnapshot:
    """UE 快照。"""
    device_id: int
    cpu_frequency_ghz: float
    transmission_power_w: float
    distance_to_edges_m: list[float] = field(default_factory=list)
    queue: QueueSnapshot = field(default_factory=QueueSnapshot)
    current_task: dict[str, Any] | None = None


@dataclass
class ESSnapshot:
    """ES 快照。"""
    server_id: int
    cpu_frequency_ghz: float
    queue: QueueSnapshot = field(default_factory=QueueSnapshot)


@dataclass
class CSSnapshot:
    """CS 快照。"""
    server_id: int
    cpu_frequency_ghz: float
    parallel_factor: float
    queue: QueueSnapshot = field(default_factory=QueueSnapshot)


@dataclass
class EnvSnapshot:
    """环境快照。"""
    delta_t: float
    global_time: float
    ues: list[UESnapshot] = field(default_factory=list)
    ess: list[ESSnapshot] = field(default_factory=list)
    css: list[CSSnapshot] = field(default_factory=list)
    # 物理模型参数
    channel_cfg: dict = field(default_factory=dict)
    backhaul_cfg: dict = field(default_factory=dict)
    energy_cfg: dict = field(default_factory=dict)
    task_cfg: dict = field(default_factory=dict)


# ======================================================================
# 评估结果
# ======================================================================

@dataclass
class EvaluationResult:
    """evaluate() 的返回值。"""
    feasible: bool = True
    predicted_cost: float = 0.0
    latency_per_ue: list[float] = field(default_factory=list)
    energy_per_ue: list[float] = field(default_factory=list)
    deadline_violations: list[int] = field(default_factory=list)  # 0/1 per UE
    queue_backlog: list[float] = field(default_factory=list)
    constraint_violations: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


# ======================================================================
# 物理计算函数（纯函数，无副作用）
# ======================================================================

def _wireless_rate(
    bandwidth_hz: float,
    transmit_power_w: float,
    distance_m: float,
    noise_psd_w_hz: float,
    path_loss_exponent: float,
    ref_distance_m: float,
    ref_distance_gain_linear: float = 1.0,
) -> float:
    """Shannon 速率 (bits/s)，正交接入无干扰。"""
    import math
    distance_m = max(float(distance_m), ref_distance_m)
    h = (
        (ref_distance_m / distance_m) ** path_loss_exponent
        * ref_distance_gain_linear
    )
    noise = noise_psd_w_hz * bandwidth_hz
    if bandwidth_hz <= 0 or noise <= 0:
        raise ValueError("wireless bandwidth and noise power must be positive")
    sinr = transmit_power_w * h / noise
    return bandwidth_hz * math.log2(1.0 + sinr)


def _backhaul_time(data_size_bits: float, rate_bps: float, propagation_s: float) -> float:
    """有线回传时延 (s)。"""
    if data_size_bits <= 0:
        return 0.0
    if rate_bps <= 0:
        raise ValueError("backhaul rate must be positive")
    return propagation_s + data_size_bits / rate_bps


def _compute_energy(
    cycles: float,
    kappa: float,
    cpu_freq_hz: float,
) -> float:
    """计算能耗：E = kappa * cycles * f^2 (J)。"""
    return kappa * cycles * (cpu_freq_hz ** 2)


def _queue_wait_time(pending_cycles: float, cpu_freq_hz: float) -> float:
    """队列等待时间 (s)。"""
    if cpu_freq_hz <= 0:
        return float("inf")
    return pending_cycles / cpu_freq_hz


def _snapshot_available_at(
    queue: QueueSnapshot,
    global_time: float,
    cpu_freq_hz: float,
) -> float:
    if queue.available_at > global_time:
        return queue.available_at
    queued_cycles = queue.pending_cycles + queue.current_remaining_cycles
    return global_time + _queue_wait_time(queued_cycles, cpu_freq_hz)


def _schedule_queue(
    queue: QueueSnapshot,
    global_time: float,
    candidates: list[dict[str, Any]],
) -> dict[Any, dict[str, float]]:
    """Schedule candidate jobs with the same release-time FCFS rule as nodes."""
    cursor = float(global_time)
    waiting = []
    existing_sequences = []

    for job in queue.jobs:
        sequence = int(job["sequence"])
        existing_sequences.append(sequence)
        if job.get("started_at") is not None:
            cursor = max(cursor, global_time) + float(job["remaining_time"])
        else:
            waiting.append(
                {
                    "key": None,
                    "release_time": float(job["release_time"]),
                    "duration": float(job["remaining_time"]),
                    "sequence": sequence,
                }
            )

    if not queue.jobs and queue.available_at > global_time:
        cursor = queue.available_at

    next_sequence = max(existing_sequences, default=-1) + 1
    for order, candidate in enumerate(candidates):
        waiting.append(
            {
                **candidate,
                "sequence": next_sequence + order,
            }
        )
    waiting.sort(key=lambda item: (item["release_time"], item["sequence"]))

    scheduled = {}
    for job in waiting:
        start = max(cursor, job["release_time"])
        completion = start + job["duration"]
        cursor = completion
        if job["key"] is not None:
            scheduled[job["key"]] = {
                "start_time": start,
                "completion_time": completion,
            }
    return scheduled


def normalize_action(action, num_edges: int) -> list[float]:
    """Project a legacy action to non-negative partition ratios and valid edge."""
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.size != 4:
        raise ValueError(f"expected action with 4 values, got shape {values.shape}")
    ratios = np.clip(values[:3], 0.0, None)
    ratio_sum = float(ratios.sum())
    if ratio_sum <= 0:
        ratios = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ratios = ratios / ratio_sum
    if num_edges <= 0:
        raise ValueError("at least one edge server is required")
    edge_id = int(np.clip(values[3], 0, num_edges - 1))
    return [float(ratios[0]), float(ratios[1]), float(ratios[2]), edge_id]


# ======================================================================
# 快照构建
# ======================================================================

def capture_snapshot(env: Any) -> EnvSnapshot:
    """从环境对象捕获快照（只读）。

    Parameters
    ----------
    env : CloudEdgeDeviceEnv
        环境实例。
    """
    cfg = env.config
    ch_cfg = cfg.get("channel", {})
    bh_cfg = cfg.get("backhaul", {})
    energy_cfg = cfg.get("energy", {})
    task_cfg = cfg.get("tasks", {})

    def queue_snapshot(node):
        jobs = []
        if node.current_execution and not node.current_execution.completed:
            jobs.append(
                {
                    "task_id": node.current_execution.task_id,
                    "release_time": node.current_execution.release_time,
                    "remaining_time": node.current_execution.remaining_time,
                    "sequence": node.current_execution.sequence,
                    "started_at": node.current_execution.started_at,
                }
            )
        jobs.extend(
            {
                "task_id": task.task_id,
                "release_time": task.release_time,
                "remaining_time": task.remaining_time,
                "sequence": task.sequence,
                "started_at": task.started_at,
            }
            for task in node.task_queue
            if not task.completed
        )
        pending = sum(
            task.remaining_cycles
            for task in node.task_queue
            if not task.completed
        )
        current_remaining = 0.0
        if node.current_execution and not node.current_execution.completed:
            current_remaining = node.current_execution.remaining_cycles
        return QueueSnapshot(
            pending_cycles=pending,
            current_remaining_cycles=current_remaining,
            num_pending_tasks=(
                len([task for task in node.task_queue if not task.completed])
                + int(
                    node.current_execution is not None
                    and not node.current_execution.completed
                )
            ),
            available_at=node.get_available_time(),
            jobs=jobs,
        )

    # UE 快照
    ues = []
    for ue in env.user_equipments:
        task = None
        idx = ue.device_id
        if env.current_tasks and idx < len(env.current_tasks) and env.current_tasks[idx] is not None:
            t = env.current_tasks[idx]
            task = {
                "task_id": t.task_id,
                "data_size_mb": t.task_data_size,
                "cpu_cycles": t.task_workload,
                "deadline": t.deadline,
                "task_type": t.task_type,
                "semantic_type": t.semantic_type,
                "output_ratio": t.output_ratio,
                "priority": t.priority,
                "arrival_time": t.arrival_time,
                "arrival_slot": t.arrival_slot,
            }
        ues.append(UESnapshot(
            device_id=ue.device_id,
            cpu_frequency_ghz=ue.cpu_frequency,
            transmission_power_w=ue.transmission_power,
            distance_to_edges_m=list(ue.distance_to_edges),
            queue=queue_snapshot(ue),
            current_task=task,
        ))

    # ES 快照
    ess = []
    for es in env.edge_servers:
        ess.append(ESSnapshot(
            server_id=es.server_id,
            cpu_frequency_ghz=es.cpu_frequency,
            queue=queue_snapshot(es),
        ))

    # CS 快照
    css = []
    for cs in env.cloud_servers:
        css.append(CSSnapshot(
            server_id=cs.server_id,
            cpu_frequency_ghz=cs.cpu_frequency,
            parallel_factor=cs.parallel_factor,
            queue=queue_snapshot(cs),
        ))

    return EnvSnapshot(
        delta_t=getattr(env, "time_step_duration", 1.0),
        global_time=env.global_time,
        ues=ues,
        ess=ess,
        css=css,
        channel_cfg=ch_cfg,
        backhaul_cfg=bh_cfg,
        energy_cfg=energy_cfg,
        task_cfg=task_cfg,
    )


# ======================================================================
# 纯函数式评估
# ======================================================================

def evaluate(
    snapshot: EnvSnapshot,
    joint_action: list[list[float]],
) -> EvaluationResult:
    """纯函数式一步评估，不修改任何环境状态。

    Parameters
    ----------
    snapshot : EnvSnapshot
        环境快照。
    joint_action : list[list[float]]
        每个 UE 的动作 [alpha1, alpha2, alpha3, edge_id]。

    Returns
    -------
    EvaluationResult
        包含可行性、代价、时延、能耗、违约和约束违规信息。
    """
    ch_cfg = snapshot.channel_cfg
    bh_cfg = snapshot.backhaul_cfg
    energy_cfg = snapshot.energy_cfg
    task_cfg = snapshot.task_cfg

    bandwidth_hz = float(ch_cfg.get("bandwidth_hz", 1e6))
    noise_psd = float(ch_cfg.get("noise_psd_w_hz", 1e-17))
    pl_exp = float(ch_cfg.get("path_loss_exponent", 3.0))
    ref_dist = float(ch_cfg.get("ref_distance_m", 1.0))
    ref_gain = 10.0 ** (
        float(ch_cfg.get("ref_distance_gain_db", 0.0)) / 10.0
    )
    downlink_power = float(
        ch_cfg.get(
            "downlink_transmit_power_w",
            ch_cfg.get("transmit_power_w", 0.5),
        )
    )
    bh_rate = float(bh_cfg.get("rate_bps", 1e9))
    bh_prop = float(bh_cfg.get("propagation_latency_s", 0.002))
    bh_ebit = float(bh_cfg.get("energy_per_bit", 1e-9))
    kappa_ue = float(energy_cfg.get("kappa_ue", 1e-28))
    kappa_es = float(energy_cfg.get("kappa_es", 3e-28))
    kappa_cs = float(energy_cfg.get("kappa_cs", 3e-28))
    default_output_ratio = float(task_cfg.get("output_ratio", 0.1))
    now = float(snapshot.global_time)

    contexts = [None for _ in snapshot.ues]
    edge_candidates = [[] for _ in snapshot.ess]
    cloud_candidates = [[] for _ in snapshot.css]
    local_schedules = {}
    baseline_schedules = {}

    for ue_idx, ue_snap in enumerate(snapshot.ues):
        task = ue_snap.current_task
        if task is None:
            continue
        raw_action = (
            joint_action[ue_idx]
            if ue_idx < len(joint_action)
            else [1.0, 0.0, 0.0, 0.0]
        )
        alpha1, alpha2, alpha3, edge_id = normalize_action(
            raw_action, len(snapshot.ess)
        )
        data_mb = float(task["data_size_mb"])
        cycles = float(task["cpu_cycles"])
        output_ratio = float(task.get("output_ratio", default_output_ratio))
        tx_power = float(ue_snap.transmission_power_w)
        distance = (
            ue_snap.distance_to_edges_m[edge_id]
            if edge_id < len(ue_snap.distance_to_edges_m)
            else float(ch_cfg.get("default_distance_m", 100.0))
        )
        uplink_rate = _wireless_rate(
            bandwidth_hz,
            tx_power,
            distance,
            noise_psd,
            pl_exp,
            ref_dist,
            ref_gain,
        )
        downlink_rate = _wireless_rate(
            bandwidth_hz,
            downlink_power,
            distance,
            noise_psd,
            pl_exp,
            ref_dist,
            ref_gain,
        )
        ue_freq = ue_snap.cpu_frequency_ghz * 1e9
        baseline_schedules[ue_idx] = _schedule_queue(
            ue_snap.queue,
            now,
            [{
                "key": ("baseline", ue_idx),
                "release_time": now,
                "duration": cycles / ue_freq,
            }],
        )[("baseline", ue_idx)]

        context = {
            "task": task,
            "action": [alpha1, alpha2, alpha3, edge_id],
            "data_mb": data_mb,
            "cycles": cycles,
            "output_ratio": output_ratio,
            "tx_power": tx_power,
            "uplink_rate": uplink_rate,
            "downlink_rate": downlink_rate,
            "ue_freq": ue_freq,
        }
        contexts[ue_idx] = context

        if alpha1 > 0:
            local_cycles = cycles * alpha1
            local_schedules[ue_idx] = _schedule_queue(
                ue_snap.queue,
                now,
                [{
                    "key": ("local", ue_idx),
                    "release_time": now,
                    "duration": local_cycles / ue_freq,
                }],
            )[("local", ue_idx)]

        if alpha2 > 0:
            edge_cycles = cycles * alpha2
            edge_data_bits = data_mb * alpha2 * 8e6
            uplink_time = edge_data_bits / uplink_rate
            context["edge"] = {
                "cycles": edge_cycles,
                "data_bits": edge_data_bits,
                "uplink_time": uplink_time,
                "release_time": now + uplink_time,
                "duration": (
                    edge_cycles
                    / (snapshot.ess[edge_id].cpu_frequency_ghz * 1e9)
                ),
            }
            edge_candidates[edge_id].append(
                {
                    "key": ("edge", ue_idx),
                    "release_time": context["edge"]["release_time"],
                    "duration": context["edge"]["duration"],
                }
            )

        if alpha3 > 0:
            cloud_cycles = cycles * alpha3
            cloud_data_bits = data_mb * alpha3 * 8e6
            wireless_uplink_time = cloud_data_bits / uplink_rate
            backhaul_uplink_time = _backhaul_time(
                cloud_data_bits, bh_rate, bh_prop
            )
            cloud_freq = (
                snapshot.css[0].cpu_frequency_ghz
                * 1e9
                * snapshot.css[0].parallel_factor
            )
            context["cloud"] = {
                "cycles": cloud_cycles,
                "data_bits": cloud_data_bits,
                "wireless_uplink_time": wireless_uplink_time,
                "backhaul_uplink_time": backhaul_uplink_time,
                "release_time": (
                    now + wireless_uplink_time + backhaul_uplink_time
                ),
                "duration": cloud_cycles / cloud_freq,
            }
            cloud_candidates[0].append(
                {
                    "key": ("cloud", ue_idx),
                    "release_time": context["cloud"]["release_time"],
                    "duration": context["cloud"]["duration"],
                }
            )

    edge_schedules = {}
    for edge_id, candidates in enumerate(edge_candidates):
        edge_schedules.update(
            _schedule_queue(snapshot.ess[edge_id].queue, now, candidates)
        )
    cloud_schedules = {}
    for cloud_id, candidates in enumerate(cloud_candidates):
        cloud_schedules.update(
            _schedule_queue(snapshot.css[cloud_id].queue, now, candidates)
        )

    latency_per_ue = []
    energy_per_ue = []
    deadline_violations = []
    queue_backlog = []
    constraint_violations = []
    per_ue_detail = []
    total_cost = 0.0

    for ue_idx, ue_snap in enumerate(snapshot.ues):
        context = contexts[ue_idx]
        backlog = (
            ue_snap.queue.pending_cycles
            + ue_snap.queue.current_remaining_cycles
        )
        queue_backlog.append(backlog)
        if context is None:
            latency_per_ue.append(0.0)
            energy_per_ue.append(0.0)
            deadline_violations.append(0)
            per_ue_detail.append(None)
            continue

        task = context["task"]
        alpha1, alpha2, alpha3, edge_id = context["action"]
        cycles = context["cycles"]
        output_ratio = context["output_ratio"]
        tx_power = context["tx_power"]
        downlink_rate = context["downlink_rate"]
        baseline = baseline_schedules[ue_idx]
        admission_wait = max(
            now - float(task.get("arrival_time", now)), 0.0
        )
        baseline_latency = (
            admission_wait + baseline["completion_time"] - now
        )
        baseline_energy = _compute_energy(
            cycles, kappa_ue, context["ue_freq"]
        )

        branches = {}
        active_latencies = []
        communication_latencies = []
        computation_latencies = []
        system_energy = 0.0
        user_energy = 0.0

        if alpha1 > 0:
            local_cycles = cycles * alpha1
            schedule = local_schedules[ue_idx]
            local_exec = local_cycles / context["ue_freq"]
            local_energy = _compute_energy(
                local_cycles, kappa_ue, context["ue_freq"]
            )
            local_latency = (
                admission_wait + schedule["completion_time"] - now
            )
            active_latencies.append(local_latency)
            computation_latencies.append(local_exec)
            system_energy += local_energy
            user_energy += local_energy
            branches["local"] = {
                "cycles": local_cycles,
                "release_time": now,
                **schedule,
                "latency": local_latency,
                "communication_latency": 0.0,
                "computation_latency": local_exec,
                "system_energy": local_energy,
                "user_energy": local_energy,
            }

        if alpha2 > 0:
            edge = context["edge"]
            schedule = edge_schedules[("edge", ue_idx)]
            uplink_energy = tx_power * edge["uplink_time"]
            result_bits = edge["data_bits"] * output_ratio
            downlink_time = result_bits / downlink_rate
            downlink_energy = downlink_power * downlink_time
            edge_compute_energy = _compute_energy(
                edge["cycles"],
                kappa_es,
                snapshot.ess[edge_id].cpu_frequency_ghz * 1e9,
            )
            edge_latency = (
                admission_wait
                + schedule["completion_time"]
                - now
                + downlink_time
            )
            branch_energy = (
                uplink_energy + edge_compute_energy + downlink_energy
            )
            branch_comm = edge["uplink_time"] + downlink_time
            active_latencies.append(edge_latency)
            communication_latencies.append(branch_comm)
            computation_latencies.append(edge["duration"])
            system_energy += branch_energy
            user_energy += uplink_energy
            branches["edge"] = {
                "edge_id": edge_id,
                "cycles": edge["cycles"],
                "release_time": edge["release_time"],
                **schedule,
                "latency": edge_latency,
                "communication_latency": branch_comm,
                "computation_latency": edge["duration"],
                "uplink_time": edge["uplink_time"],
                "downlink_time": downlink_time,
                "system_energy": branch_energy,
                "user_energy": uplink_energy,
            }

        if alpha3 > 0:
            cloud = context["cloud"]
            schedule = cloud_schedules[("cloud", ue_idx)]
            wireless_uplink_energy = tx_power * cloud["wireless_uplink_time"]
            backhaul_uplink_energy = bh_ebit * cloud["data_bits"]
            result_bits = cloud["data_bits"] * output_ratio
            backhaul_downlink_time = _backhaul_time(
                result_bits, bh_rate, bh_prop
            )
            wireless_downlink_time = result_bits / downlink_rate
            backhaul_downlink_energy = bh_ebit * result_bits
            wireless_downlink_energy = (
                downlink_power * wireless_downlink_time
            )
            cloud_compute_energy = _compute_energy(
                cloud["cycles"],
                kappa_cs,
                snapshot.css[0].cpu_frequency_ghz * 1e9,
            )
            return_time = backhaul_downlink_time + wireless_downlink_time
            cloud_latency = (
                admission_wait
                + schedule["completion_time"]
                - now
                + return_time
            )
            branch_energy = (
                wireless_uplink_energy
                + backhaul_uplink_energy
                + cloud_compute_energy
                + backhaul_downlink_energy
                + wireless_downlink_energy
            )
            branch_comm = (
                cloud["wireless_uplink_time"]
                + cloud["backhaul_uplink_time"]
                + return_time
            )
            active_latencies.append(cloud_latency)
            communication_latencies.append(branch_comm)
            computation_latencies.append(cloud["duration"])
            system_energy += branch_energy
            user_energy += wireless_uplink_energy
            branches["cloud"] = {
                "cloud_id": 0,
                "cycles": cloud["cycles"],
                "release_time": cloud["release_time"],
                **schedule,
                "latency": cloud_latency,
                "communication_latency": branch_comm,
                "computation_latency": cloud["duration"],
                "wireless_uplink_time": cloud["wireless_uplink_time"],
                "backhaul_uplink_time": cloud["backhaul_uplink_time"],
                "backhaul_downlink_time": backhaul_downlink_time,
                "wireless_downlink_time": wireless_downlink_time,
                "system_energy": branch_energy,
                "user_energy": wireless_uplink_energy,
            }

        task_latency = max(active_latencies, default=0.0)
        communication_latency = max(communication_latencies, default=0.0)
        computation_latency = max(computation_latencies, default=0.0)
        deadline = float(task["deadline"])
        violated = int(task_latency > deadline)
        if violated:
            constraint_violations.append(
                f"UE{ue_idx}: latency {task_latency:.4f}s > deadline {deadline:.4f}s"
            )

        latency_per_ue.append(task_latency)
        energy_per_ue.append(system_energy)
        deadline_violations.append(violated)
        total_cost += task_latency + system_energy
        per_ue_detail.append(
            {
                "normalized_action": context["action"],
                "branches": branches,
                "communication_latency": communication_latency,
                "computation_latency": computation_latency,
                "system_energy": system_energy,
                "user_energy": user_energy,
                "baseline_latency": baseline_latency,
                "baseline_energy": baseline_energy,
                "admission_wait": admission_wait,
            }
        )

    return EvaluationResult(
        feasible=len(constraint_violations) == 0,
        predicted_cost=total_cost,
        latency_per_ue=latency_per_ue,
        energy_per_ue=energy_per_ue,
        deadline_violations=deadline_violations,
        queue_backlog=queue_backlog,
        constraint_violations=constraint_violations,
        detail={
            "per_ue": per_ue_detail,
            "energy_scope": "system",
        },
    )
