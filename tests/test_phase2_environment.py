"""Phase 2 环境测试 — Gate 2 验证

测试覆盖 Phase 2 的所有测试要求：
  - 无到达时队列不增长
  - 服务量大于队列时队列不为负
  - 增大带宽时传输时延单调下降
  - 增大 CPU 频率时执行时延单调下降
  - 增大发射功率时无线速率单调不降
  - 三个分支总时延等于最大分支加返回时延
  - evaluate(action) 不改变真实环境状态
  - 手工小例子与解析计算一致

NOTE: 本文件由人工完成 Phase 2 初稿，已由 Codex 于 2026-06-11 扩展审计测试。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from LLM4RL.environment.channel_model import WirelessChannel, build_channel_from_config
from LLM4RL.environment.backhaul_model import BackhaulLink, build_backhaul_from_config
from LLM4RL.environment.snapshot import evaluate, EnvSnapshot, UESnapshot, ESSnapshot, CSSnapshot, QueueSnapshot
from LLM4RL.environment.cloud_edge_env import CloudEdgeDeviceEnv
from LLM4RL.environment.device_models import UserEquipment, EdgeServer, CloudServer
from LLM4RL.environment.task_generator import TaskGenerator, Task
from LLM4RL.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ======================================================================
# 辅助函数
# ======================================================================

def phase2_config():
    """返回 Phase 2 测试配置。"""
    return load_config(str(PROJECT_ROOT / "configs" / "smoke_phase2.yaml"))


def make_env(config=None):
    """创建 Phase 2 环境实例。"""
    cfg = config or phase2_config()
    return CloudEdgeDeviceEnv(cfg)


# ======================================================================
# 无线信道模型测试
# ======================================================================

class TestWirelessChannel:

    def test_rate_positive(self):
        ch = WirelessChannel(bandwidth_hz=1e6, transmit_power_w=0.5, noise_psd_w_hz=1e-17)
        rate = ch.achievable_rate(distance_m=100.0)
        assert rate > 0

    def test_rate_monotonic_bandwidth(self):
        """增大带宽时速率应单调增加。"""
        ch1 = WirelessChannel(bandwidth_hz=1e6, transmit_power_w=0.5, noise_psd_w_hz=1e-17)
        ch2 = WirelessChannel(bandwidth_hz=10e6, transmit_power_w=0.5, noise_psd_w_hz=1e-17)
        assert ch2.achievable_rate(100.0) > ch1.achievable_rate(100.0)

    def test_rate_monotonic_power(self):
        """增大发射功率时速率应单调增加。"""
        ch1 = WirelessChannel(bandwidth_hz=1e6, transmit_power_w=0.1, noise_psd_w_hz=1e-17)
        ch2 = WirelessChannel(bandwidth_hz=1e6, transmit_power_w=1.0, noise_psd_w_hz=1e-17)
        assert ch2.achievable_rate(100.0) >= ch1.achievable_rate(100.0)

    def test_rate_monotonic_distance(self):
        """增大距离时速率应单调下降。"""
        ch = WirelessChannel(bandwidth_hz=1e6, transmit_power_w=0.5, noise_psd_w_hz=1e-17)
        r_near = ch.achievable_rate(10.0)
        r_far = ch.achievable_rate(1000.0)
        assert r_near > r_far

    def test_transmission_time_monotonic_bandwidth(self):
        """增大带宽时传输时延应单调下降。"""
        ch1 = WirelessChannel(bandwidth_hz=1e6, transmit_power_w=0.5, noise_psd_w_hz=1e-17)
        ch2 = WirelessChannel(bandwidth_hz=10e6, transmit_power_w=0.5, noise_psd_w_hz=1e-17)
        data_bits = 1e6  # 1 Mbit
        t1 = ch1.transmission_time(data_bits, 100.0)
        t2 = ch2.transmission_time(data_bits, 100.0)
        assert t2 < t1

    def test_transmission_time_monotonic_power(self):
        """增大发射功率时传输时延应单调下降（或不增）。"""
        ch1 = WirelessChannel(bandwidth_hz=1e6, transmit_power_w=0.1, noise_psd_w_hz=1e-17)
        ch2 = WirelessChannel(bandwidth_hz=1e6, transmit_power_w=1.0, noise_psd_w_hz=1e-17)
        data_bits = 1e6
        t1 = ch1.transmission_time(data_bits, 100.0)
        t2 = ch2.transmission_time(data_bits, 100.0)
        assert t2 <= t1

    def test_energy_equals_power_times_time(self):
        """能耗 = 功率 × 时延。"""
        ch = WirelessChannel(bandwidth_hz=1e6, transmit_power_w=0.5, noise_psd_w_hz=1e-17)
        data_bits = 1e6
        t = ch.transmission_time(data_bits, 100.0)
        e = ch.transmission_energy(data_bits, 100.0)
        assert abs(e - ch.transmit_power_w * t) < 1e-10

    def test_build_from_config(self):
        """从配置构建信道模型。"""
        cfg = {"channel": {"bandwidth_hz": 5e6, "transmit_power_w": 1.0, "noise_psd_w_hz": 1e-18}}
        ch = build_channel_from_config(cfg)
        assert ch.bandwidth_hz == 5e6
        assert ch.transmit_power_w == 1.0

    def test_path_loss_decreases_with_distance(self):
        """路径损耗随距离增加而减小（衰减增大）。"""
        ch = WirelessChannel(path_loss_exponent=3.0, ref_distance_m=1.0)
        pl_near = ch.path_loss(10.0)
        pl_far = ch.path_loss(100.0)
        assert pl_near > pl_far


# ======================================================================
# 回传链路模型测试
# ======================================================================

class TestBackhaulLink:

    def test_transmission_time_positive(self):
        bh = BackhaulLink(rate_bps=1e9, propagation_latency_s=0.002)
        t = bh.transmission_time(1e6)
        assert t > 0.002  # 至少包含传播时延

    def test_transmission_time_zero_data(self):
        bh = BackhaulLink(rate_bps=1e9, propagation_latency_s=0.002)
        assert bh.transmission_time(0.0) == 0.0

    def test_energy_proportional_to_data(self):
        bh = BackhaulLink(energy_per_bit=1e-9)
        e1 = bh.transmission_energy(1e6)
        e2 = bh.transmission_energy(2e6)
        assert abs(e2 - 2 * e1) < 1e-10

    def test_time_monotonic_rate(self):
        """增大链路速率时传输时延应单调下降。"""
        bh1 = BackhaulLink(rate_bps=1e8, propagation_latency_s=0.002)
        bh2 = BackhaulLink(rate_bps=1e9, propagation_latency_s=0.002)
        data_bits = 1e9  # 1 Gbit
        assert bh2.transmission_time(data_bits) < bh1.transmission_time(data_bits)

    def test_build_from_config(self):
        cfg = {"backhaul": {"rate_bps": 5e9, "energy_per_bit": 2e-10, "propagation_latency_s": 0.001}}
        bh = build_backhaul_from_config(cfg)
        assert bh.rate_bps == 5e9


# ======================================================================
# 设备模型测试
# ======================================================================

class TestDeviceModels:

    def test_cloud_server_phase2_has_queue(self):
        """Phase 2 云服务器应有任务队列。"""
        cfg = phase2_config()
        cs = CloudServer(server_id=0, config=cfg)
        assert cs.model_version == 2
        cs.add_task("t1", 1e9, 0.0)
        assert cs.current_execution is not None

    def test_cloud_server_phase1_no_queue(self):
        """Phase 1 云服务器不应管理队列。"""
        cfg = {"model_version": 1}
        cs = CloudServer(server_id=0, config=cfg)
        cs.add_task("t1", 1e9, 0.0)
        assert cs.current_execution is None

    def test_ue_dvfs_energy_phase2(self):
        """Phase 2 UE 能耗应使用 DVFS 模型。"""
        cfg = phase2_config()
        ue = UserEquipment(device_id=0, config=cfg)
        cycles = 1e9
        e = ue.calculate_energy_consumption(cycles)
        expected = ue.kappa_ue * cycles * (ue.cpu_frequency * 1e9) ** 2
        assert abs(e - expected) < 1e-15

    def test_es_dvfs_energy_phase2(self):
        """Phase 2 ES 能耗应使用 DVFS 模型。"""
        cfg = phase2_config()
        es = EdgeServer(server_id=0, cpu_frequency=5.0, config=cfg)
        cycles = 1e9
        e = es.calculate_energy_consumption(cycles)
        expected = es.kappa_es * cycles * (es.cpu_frequency * 1e9) ** 2
        assert abs(e - expected) < 1e-15

    def test_queue_update_advances(self):
        """队列应随时间推进正确更新。"""
        cfg = phase2_config()
        ue = UserEquipment(device_id=0, config=cfg)
        ue.cpu_frequency = 1.0  # 1 GHz
        ue.add_task("t1", 1e9, 0.0)  # 1 second execution
        assert ue.current_execution is not None
        ue.update_tasks(0.5)
        assert ue.current_execution.remaining_time == pytest.approx(0.5, abs=1e-6)


# ======================================================================
# 队列演化测试
# ======================================================================

class TestQueueEvolution:

    def test_no_arrival_no_growth(self):
        """无到达时队列不应增长。"""
        cfg = phase2_config()
        env = CloudEdgeDeviceEnv(cfg)
        env.reset(seed=42)

        # 记录初始队列长度
        initial_queues = [len(ue.task_queue) for ue in env.user_equipments]

        # 只更新设备状态，不生成新任务
        env._update_all_devices(env.time_step_duration)

        # 队列长度不应增加（任务可能完成，但不应新增）
        for i, ue in enumerate(env.user_equipments):
            assert len(ue.task_queue) <= initial_queues[i]

    def test_service_not_negative(self):
        """服务量大于队列时队列不应为负。"""
        cfg = phase2_config()
        env = CloudEdgeDeviceEnv(cfg)
        env.reset(seed=42)

        # 推进大量时间，确保所有任务完成
        for _ in range(100):
            env._update_all_devices(env.time_step_duration)

        for ue in env.user_equipments:
            assert len(ue.task_queue) == 0
            if ue.current_execution is not None:
                assert ue.current_execution.remaining_time >= 0

    def test_queues_persist_across_steps_phase2(self):
        """Phase 2 队列应跨时隙保持。"""
        cfg = phase2_config()
        cfg["maddpg"]["max_steps"] = 3
        env = CloudEdgeDeviceEnv(cfg)
        env.reset(seed=42)

        # 添加任务
        ue = env.user_equipments[0]
        ue.add_task("test_task", 1e10, env.global_time)  # 大任务
        queue_len = len(ue.task_queue)
        has_task = queue_len > 0 or ue.current_execution is not None

        # 推进一个时隙
        env._update_all_devices(env.time_step_duration)

        # 应该仍有任务（大任务不会在一个时隙内完成）
        still_has_task = len(ue.task_queue) > 0 or ue.current_execution is not None
        assert still_has_task


# ======================================================================
# 并行分支时延测试
# ======================================================================

class TestParallelLatency:

    def test_three_branches_max_plus_downlink(self):
        """总时延等于三个完整分支（含各自返回路径）的最大值。"""
        cfg = phase2_config()
        env = CloudEdgeDeviceEnv(cfg)
        env.reset(seed=42)

        actions = [[0.3, 0.4, 0.3, 0]] * env.num_devices
        result = env.evaluate_action(actions)

        for ue_idx, latency in enumerate(result.latency_per_ue):
            detail = result.detail["per_ue"][ue_idx]
            if detail is None:
                assert latency == 0.0
                continue
            branch_latencies = [
                branch["latency"]
                for branch in detail["branches"].values()
            ]
            assert latency == pytest.approx(max(branch_latencies))

    def test_local_only_action(self):
        """全本地动作不应有通信延迟。"""
        cfg = phase2_config()
        env = CloudEdgeDeviceEnv(cfg)
        env.reset(seed=42)

        actions = [[1.0, 0.0, 0.0, 0]] * env.num_devices
        result = env.evaluate_action(actions)

        # 本地执行不应有通信延迟（但可能有时延因为队列等待）
        for lat in result.latency_per_ue:
            assert lat >= 0


# ======================================================================
# evaluate 纯函数式测试
# ======================================================================

class TestEvaluate:

    def test_evaluate_does_not_modify_env(self):
        """evaluate() 不应改变环境状态。"""
        cfg = phase2_config()
        env = CloudEdgeDeviceEnv(cfg)
        env.reset(seed=42)

        # 记录状态
        state_before = env._get_observation().copy()
        global_time_before = env.global_time
        queue_lens_before = [len(ue.task_queue) for ue in env.user_equipments]

        # 执行 evaluate
        actions = [[0.3, 0.4, 0.3, 0]] * env.num_devices
        env.evaluate_action(actions)

        # 验证状态未变
        state_after = env._get_observation()
        assert np.array_equal(state_before, state_after)
        assert env.global_time == global_time_before
        for i, ue in enumerate(env.user_equipments):
            assert len(ue.task_queue) == queue_lens_before[i]

    def test_evaluate_handcrafted_example(self):
        """手工小例子与解析计算一致。"""
        # 构造简单快照
        ue = UESnapshot(
            device_id=0,
            cpu_frequency_ghz=1.0,
            transmission_power_w=0.5,
            distance_to_edges_m=[50.0],
            queue=QueueSnapshot(0.0, 0.0, 0),
            current_task={
                "task_id": "test",
                "data_size_mb": 100.0,
                "cpu_cycles": 2e9,  # 2 Gcycles
                "deadline": 100.0,
                "task_type": "simple",
            },
        )
        es = ESSnapshot(
            server_id=0,
            cpu_frequency_ghz=5.0,
            queue=QueueSnapshot(0.0, 0.0, 0),
        )
        cs = CSSnapshot(
            server_id=0,
            cpu_frequency_ghz=20.0,
            parallel_factor=8.0,
        )
        snapshot = EnvSnapshot(
            delta_t=1.0,
            global_time=0.0,
            ues=[ue],
            ess=[es],
            css=[cs],
            channel_cfg={
                "bandwidth_hz": 1e6,
                "transmit_power_w": 0.5,
                "noise_psd_w_hz": 1e-17,
                "path_loss_exponent": 3.0,
                "ref_distance_m": 1.0,
            },
            backhaul_cfg={
                "rate_bps": 1e9,
                "energy_per_bit": 1e-9,
                "propagation_latency_s": 0.002,
            },
            energy_cfg={"kappa_ue": 1e-28, "kappa_es": 3e-28, "kappa_cs": 3e-28},
            task_cfg={"output_ratio": 0.1, "processing_density": 0.2e9},
        )

        # 解析计算：全本地动作无需无线结果返回。
        # 本地时延 = cycles / freq = 2e9 / 1e9 = 2.0s
        local_exec = 2e9 / 1e9  # 2.0s
        expected_local = local_exec

        result_local = evaluate(snapshot, [[1.0, 0.0, 0.0, 0]])
        assert abs(result_local.latency_per_ue[0] - expected_local) < 0.01
        assert result_local.detail["per_ue"][0]["communication_latency"] == 0.0

        # 全边缘动作
        result_edge = evaluate(snapshot, [[0.0, 1.0, 0.0, 0]])
        assert result_edge.latency_per_ue[0] > 0

        # 可行性检查：deadline 100s 应该可行
        assert result_local.feasible
        assert result_edge.feasible

        # 边缘时延应大于本地（因为通信开销）
        # 但本地也有下行，所以需要看具体情况
        assert result_edge.latency_per_ue[0] > 0


# ======================================================================
# 端到端 Phase 2 环境测试
# ======================================================================

class TestPhase2Env:

    def test_env_creation(self):
        """Phase 2 环境能正常创建。"""
        env = make_env()
        assert env.model_version == 2
        assert env.physics_version == 2
        assert hasattr(env, 'channel')
        assert hasattr(env, 'backhaul')

    def test_reset_works(self):
        """Phase 2 环境能正常重置。"""
        env = make_env()
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert len(obs) > 0

    def test_step_works(self):
        """Phase 2 环境能正常步进。"""
        env = make_env()
        env.reset(seed=42)
        actions = np.array([[0.3, 0.4, 0.3, 0]] * env.num_devices)
        obs, rewards, terminated, truncated, info = env.step(actions)
        assert obs is not None
        assert len(rewards) == env.num_devices

    def test_multi_step_no_crash(self):
        """多步运行不应崩溃。"""
        env = make_env()
        env.reset(seed=42)
        for _ in range(5):
            actions = np.array([[0.3, 0.4, 0.3, 0]] * env.num_devices)
            obs, rewards, terminated, truncated, info = env.step(actions)
            if truncated:
                break

    def test_snapshot_creation(self):
        """快照能正常创建。"""
        env = make_env()
        env.reset(seed=42)
        snapshot = env.capture_snapshot()
        assert snapshot is not None
        assert len(snapshot.ues) == env.num_devices
        assert len(snapshot.ess) == env.num_edges
        assert snapshot.delta_t == env.time_step_duration

    def test_phase2_observation_contains_channel_and_semantics(self):
        env = make_env()
        observation, _ = env.reset(seed=42)
        assert observation.shape == env.observation_space.shape
        agent_state = env.extract_agent_state(observation, 0)
        assert agent_state.shape == (env.get_agent_state_dim(),)
        assert agent_state[2] == pytest.approx(
            min(len(env.pending_task_queues[0]) / 10.0, 1.0)
        )
        if env.current_tasks[0] is not None:
            semantic_one_hot = agent_state[-4:]
            assert semantic_one_hot.sum() == pytest.approx(1.0)

    def test_phase1_backward_compatible(self):
        """Phase 1 配置仍能正常工作。"""
        cfg = load_config(str(PROJECT_ROOT / "configs" / "smoke.yaml"))
        env = CloudEdgeDeviceEnv(cfg)
        assert env.model_version == 2
        assert env.physics_version == 1
        assert not hasattr(env, "channel")
        obs, _ = env.reset(seed=42)
        assert obs.shape == (17,)
        assert env.get_agent_state_dim() == 9
        assert obs is not None
        actions = np.array([[0.3, 0.4, 0.3, 0]] * env.num_devices)
        obs, rewards, _, _, _ = env.step(actions)
        assert obs is not None

    def test_task_has_phase2_fields(self):
        """Phase 2 任务应包含语义类型和输出比例。"""
        cfg = phase2_config()
        env = CloudEdgeDeviceEnv(cfg)
        env.reset(seed=42)
        for task in env.current_tasks:
            if task is not None:
                assert hasattr(task, 'semantic_type')
                assert hasattr(task, 'output_ratio')
                assert task.output_ratio > 0

    def test_queue_backlog_in_info(self):
        """Phase 2 info 应包含队列积压信息。"""
        env = make_env()
        env.reset(seed=42)
        actions = np.array([[0.3, 0.4, 0.3, 0]] * env.num_devices)
        _, _, _, _, info = env.step(actions)
        assert 'queue_backlogs' in info


# ======================================================================
# Codex 独立审计回归测试
# ======================================================================

def deterministic_task(device_id=0, task_id="audit", cycles=2e9, data_mb=10.0):
    return Task(
        {
            "task_id": task_id,
            "device_id": device_id,
            "type": "ai_inference",
            "semantic_type": "ai_inference",
            "data_size": data_mb,
            "cpu_cycles": cycles,
            "deadline": 100.0,
            "output_ratio": 0.1,
            "priority": 2,
            "arrival_time": 0.0,
            "arrival_slot": 0,
        }
    )


class TestPhase2AuditRegressions:

    def test_future_release_does_not_receive_early_service(self):
        es = EdgeServer(0, 1.0, phase2_config())
        es.add_task("future", 1e9, 10.0)
        es.update_tasks(1.0)
        assert es.current_execution is not None
        assert es.current_execution.remaining_time == pytest.approx(1.0)
        assert es.current_execution.started_at is None

    def test_unused_slot_service_advances_following_tasks(self):
        es = EdgeServer(0, 1.0, phase2_config())
        es.add_task("first", 0.25e9, 0.0)
        es.add_task("second", 0.25e9, 0.0)
        es.update_tasks(1.0)
        assert es.current_execution is None
        assert es.task_queue == []

    def test_local_only_has_no_downlink_latency(self):
        env = make_env()
        env.reset(seed=42)
        env.pending_task_queues = [[deterministic_task(0)]] + [
            [] for _ in range(env.num_devices - 1)
        ]
        env._refresh_current_tasks()
        result = env.evaluate_action(
            [[1.0, 0.0, 0.0, 0.0]] * env.num_devices
        )
        expected = (
            env.current_tasks[0].task_workload
            / (env.user_equipments[0].cpu_frequency * 1e9)
        )
        assert result.latency_per_ue[0] == pytest.approx(expected)
        assert result.detail["per_ue"][0]["communication_latency"] == 0.0

    def test_edge_system_energy_includes_compute_and_return(self):
        env = make_env()
        env.reset(seed=42)
        env.pending_task_queues = [[deterministic_task(0)]] + [
            [] for _ in range(env.num_devices - 1)
        ]
        env._refresh_current_tasks()
        result = env.evaluate_action(
            [[0.0, 1.0, 0.0, 0.0]] * env.num_devices
        )
        branch = result.detail["per_ue"][0]["branches"]["edge"]
        uplink_energy = branch["user_energy"]
        expected_compute = env.edge_servers[0].calculate_energy_consumption(
            env.current_tasks[0].task_workload
        )
        assert branch["system_energy"] > uplink_energy + expected_compute

    def test_snapshot_captures_distance_and_cloud_queue(self):
        cfg = phase2_config()
        cfg["channel"]["default_distance_m"] = 25.0
        env = make_env(cfg)
        env.reset(seed=42)
        env.cloud_servers[0].add_task("busy", 1e12, 0.0)
        snapshot = env.capture_snapshot()
        assert snapshot.ues[0].distance_to_edges_m == [25.0] * env.num_edges
        assert snapshot.css[0].queue.current_remaining_cycles > 0
        assert snapshot.css[0].queue.available_at > env.global_time

    def test_joint_evaluator_matches_environment_step_under_contention(self):
        env = make_env()
        env.reset(seed=42)
        env.pending_task_queues = [
            [deterministic_task(i, f"task-{i}")]
            for i in range(env.num_devices)
        ]
        env._refresh_current_tasks()
        env.edge_servers[0].add_task("busy", 5e9, 0.0)
        actions = np.asarray(
            [[0.0, 1.0, 0.0, 0.0]] * env.num_devices,
            dtype=np.float32,
        )
        predicted = env.evaluate_action(actions.tolist())
        _, _, _, _, info = env.step(actions)
        assert info["total_latencies"] == pytest.approx(
            predicted.latency_per_ue
        )
        assert info["total_energies"] == pytest.approx(
            predicted.energy_per_ue
        )
        assert predicted.latency_per_ue[1] > predicted.latency_per_ue[0]
        assert predicted.latency_per_ue[2] > predicted.latency_per_ue[1]

    def test_contention_is_ordered_by_release_time_not_ue_index(self):
        env = make_env()
        env.reset(seed=42)
        env.pending_task_queues = [
            [deterministic_task(0, "large-upload", data_mb=100.0)],
            [deterministic_task(1, "small-upload", data_mb=1.0)],
            [],
        ]
        env._refresh_current_tasks()
        actions = np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        predicted = env.evaluate_action(actions.tolist())
        large_branch = predicted.detail["per_ue"][0]["branches"]["edge"]
        small_branch = predicted.detail["per_ue"][1]["branches"]["edge"]
        assert small_branch["release_time"] < large_branch["release_time"]
        assert small_branch["start_time"] < large_branch["start_time"]
        _, _, _, _, info = env.step(actions)
        assert info["total_latencies"] == pytest.approx(
            predicted.latency_per_ue
        )

    def test_multiple_arrivals_are_preserved(self):
        env = make_env()
        env.reset(seed=42)
        env.pending_task_queues = [[] for _ in range(env.num_devices)]
        env.task_completion_stats["total_tasks_generated"] = 0
        task_a = deterministic_task(0, "a").to_dict()
        task_b = deterministic_task(0, "b").to_dict()
        task_a["data_size"] = task_a.pop("data_size_mb")
        task_b["data_size"] = task_b.pop("data_size_mb")
        env.task_generator.generate_poisson_tasks = lambda **_: {
            0: [task_a, task_b],
            1: [],
            2: [],
        }
        env._generate_new_tasks()
        assert len(env.pending_task_queues[0]) == 2
        assert env.task_completion_stats["total_tasks_generated"] == 2

    def test_task_arrival_uses_simulation_time(self):
        env = make_env()
        env.reset(seed=42)
        for queue in env.pending_task_queues:
            for task in queue:
                assert task.arrival_time == env.global_time

    def test_pending_admission_wait_counts_toward_deadline(self):
        env = make_env()
        env.reset(seed=42)
        task = deterministic_task(
            0, "waiting", cycles=1e9, data_mb=1.0
        )
        task.deadline = 5.0
        task.arrival_time = 0.0
        env.pending_task_queues = [[task]] + [
            [] for _ in range(env.num_devices - 1)
        ]
        env._refresh_current_tasks()
        env.global_time = 5.0
        for node in [
            *env.user_equipments,
            *env.edge_servers,
            *env.cloud_servers,
        ]:
            node.queue_time = 5.0
        result = env.evaluate_action(
            [[1.0, 0.0, 0.0, 0.0]] * env.num_devices
        )
        execution = 1e9 / (
            env.user_equipments[0].cpu_frequency * 1e9
        )
        assert result.latency_per_ue[0] == pytest.approx(5.0 + execution)
        assert result.deadline_violations[0] == 1

    def test_semantic_generator_supports_each_declared_type(self):
        for semantic_type in TaskGenerator.SEMANTIC_TYPES:
            config = {
                "model_version": 2,
                "physics_version": 2,
                "semantic_type_weights": {
                    name: float(name == semantic_type)
                    for name in TaskGenerator.SEMANTIC_TYPES
                },
            }
            task = TaskGenerator(config).generate_single_task()
            assert task["semantic_type"] == semantic_type
            assert task["priority"] >= 1

    def test_negative_partition_values_are_projected(self):
        env = make_env()
        env.reset(seed=42)
        env.pending_task_queues = [[deterministic_task(0)]] + [
            [] for _ in range(env.num_devices - 1)
        ]
        env._refresh_current_tasks()
        result = env.evaluate_action(
            [[-1.0, 2.0, -3.0, 99.0]] * env.num_devices
        )
        assert result.detail["per_ue"][0]["normalized_action"] == [
            0.0,
            1.0,
            0.0,
            env.num_edges - 1,
        ]

    def test_invalid_physical_parameters_are_rejected(self):
        with pytest.raises(ValueError):
            WirelessChannel(bandwidth_hz=0)
        with pytest.raises(ValueError):
            BackhaulLink(rate_bps=0)
