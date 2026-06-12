# environment/cloud_edge_env.py
"""
云边端三层架构卸载环境 — Phase 1 + Phase 2 双版本兼容

Phase 1 (environment.physics_version < 2):
  - 简化物理模型，常数传输速率
  - simple_mode 下每步清空队列 (memoryless)
Phase 2 (environment.physics_version >= 2):
  - 显式 delta_t 时隙
  - 不丢任务的队列演化
  - Shannon 无线信道速率模型
  - 有线回传链路 (ES→CS)
  - 下行结果返回
  - 并行分支最大完成时延
  - environment snapshot 与纯函数式评估

NOTE: 本文件由人工完成 Phase 2 初稿，已由 Codex 于 2026-06-11 审查并修复。
"""
import numpy as np
import gymnasium as gym
import random
import copy
from gymnasium import spaces
from collections import defaultdict
from .device_models import UserEquipment, EdgeServer, CloudServer
from .task_generator import TaskGenerator, Task


class TaskExecution:
    """任务执行状态跟踪"""
    def __init__(self, task_id, device_id, task_workload, data_size, start_time, execution_time, node_type, node_id, original_task_deadline):
        self.task_id = task_id
        self.device_id = device_id  # 发起任务的设备ID
        self.task_workload = task_workload    # CPU周期数
        self.data_size = data_size  # 数据大小
        self.start_time = start_time
        self.execution_time = execution_time
        self.remaining_time = execution_time
        self.node_type = node_type  # 'local', 'edge', 'cloud'
        self.node_id = node_id      # 节点ID
        self.completed = False
        self.original_task_deadline = original_task_deadline  # 原始任务的截止时间
        self.creation_step = 0  # 任务创建的step
        
    def is_deadline_violated(self, current_time):
        """检查是否违反截止时间"""
        expected_completion_time = self.start_time + self.execution_time
        return expected_completion_time > self.original_task_deadline
    
    def get_progress(self):
        """获取执行进度 (0-1)"""
        if self.execution_time == 0:
            return 1.0
        return max(0, (self.execution_time - self.remaining_time) / self.execution_time)


class CloudEdgeDeviceEnv(gym.Env):
    """云边端三层架构卸载环境 - 简化设备模型版本"""
    
    def __init__(self, config):
        super(CloudEdgeDeviceEnv, self).__init__()

        self.config = config
        self.model_version = config.get('model_version', 1)
        self.physics_version = int(
            config.get('environment', {}).get('physics_version', 1)
        )

        # 基础配置
        self.num_devices = config.get('environment', {}).get('num_devices', 10)
        self.num_edges = config.get('environment', {}).get('num_edges', 5)
        self.num_clouds = config.get('environment', {}).get('num_clouds', 1)

        # 创建设备
        self._create_devices()

        # 结果格式版本与物理模型版本相互独立。
        tg_cfg = copy.deepcopy(config.get('tasks', {}))
        tg_cfg['model_version'] = self.model_version
        tg_cfg['physics_version'] = self.physics_version
        self.task_generator = TaskGenerator(tg_cfg)
        self.debug = config.get('environment', {}).get('debug', False)

        # Phase 2: 时隙与物理模型
        self.time_step_duration = float(config.get('environment', {}).get('delta_t', 1.0))
        if self.physics_version >= 2:
            from .channel_model import build_channel_from_config
            from .backhaul_model import build_backhaul_from_config
            self.channel = build_channel_from_config(config)
            self.backhaul = build_backhaul_from_config(config)
            # 为 UE 分配到各 ES 的距离
            default_dist = float(config.get('channel', {}).get('default_distance_m', 100.0))
            for ue in self.user_equipments:
                ue.distance_to_edges = [default_dist] * self.num_edges
        
        # 简化：删除复杂到达与应用混合配置，任务生成统一使用 TaskGenerator 简化模式

        # 状态空间维度计算
        self.phase2_ue_feature_dim = 3
        self.phase2_task_feature_dim = 9
        if self.physics_version >= 2:
            self.state_dim = (
                self.phase2_ue_feature_dim * self.num_devices
                + 2 * self.num_edges
                + 2 * self.num_clouds
                + self.num_devices * self.num_edges
                + self.phase2_task_feature_dim * self.num_devices
            )
        else:
            self.state_dim = (
                2 * self.num_devices
                + 2 * self.num_edges
                + 1 * self.num_clouds
                + 2 * self.num_devices
            )

        # 定义观察和动作空间
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )

        # 动作空间定义：每个设备的三元分割决策 [α1, α2, α3, edge_id]
        self.action_type = config.get('environment', {}).get('action_space', {}).get('type', 'continuous')
        self.hybrid_bins = config.get('environment', {}).get('action_space', {}).get('bins', [0.0, 0.25, 0.5, 0.75, 1.0])
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array(
                [1.0, 1.0, 1.0, float(self.num_edges - 1)],
                dtype=np.float32,
            ),
            dtype=np.float32
        )

        # 任务执行跟踪
        self.current_tasks = None
        self.pending_task_queues = [
            [] for _ in range(self.num_devices)
        ]
        self.task_executions = defaultdict(list)  # 按节点分组的执行队列
        self.completed_tasks_history = []  # 已完成任务的历史记录
        self.global_time = 0.0  # 全局时间步
        
        # Episode控制
        self.episode_step = 0
        
        # 从配置中读取max_steps，而不是硬编码为100
        # 优先从maddpg配置读取，如果不存在则从training配置读取，如果都不存在则默认为200
        self.max_steps = config.get('maddpg', {}).get('max_steps', 
                          config.get('training', {}).get('max_steps_per_episode', 200))
        print(f"环境初始化: 最大步数设置为 {self.max_steps}")

        # 奖励配置
        self.reward_cfg = config.get('reward', {})
        self.reward_type = self.reward_cfg.get('type', 'inverse')
        self.latency_weight = float(self.reward_cfg.get('latency_weight', 1.0))
        self.energy_weight = float(self.reward_cfg.get('energy_weight', 1.0))
        self.deadline_penalty = float(self.reward_cfg.get('deadline_penalty', 0.0))
        self.reward_energy_scope = self.reward_cfg.get(
            'energy_scope', 'system'
        )
        if self.reward_energy_scope not in {'user', 'system'}:
            raise ValueError(
                "reward.energy_scope must be either 'user' or 'system'"
            )
        
        # 任务生成控制（简化）
        self.last_generation_step = 0
        
        # 统计信息
        self.step_stats = {
            'tasks_completed': 0,
            'tasks_timeout': 0,
            'total_latency': 0.0,
            'total_energy': 0.0,
            'communication_latency': 0.0,
            'computation_latency': 0.0
        }
        
        # 任务完成率统计
        self.task_completion_stats = {
            'total_tasks_generated': 0,        # 总生成任务数
            'tasks_completed_on_time': 0,      # 按时完成的任务数
            'tasks_completed_late': 0,         # 超时完成的任务数
            'tasks_failed': 0,                 # 失败任务数
            'completion_times': [],            # 任务完成时间记录
            'deadline_violations': [],         # 截止时间违反记录
            'timeout_reasons': []              # 超时原因记录
        }

    def _create_devices(self):
        """创建云边端三层设备"""
        # 创建端侧设备（异构CPU频率：0.5-1.0 GHz）
        self.user_equipments = []
        for i in range(self.num_devices):
            ue = UserEquipment(i, config=self.config)
            self.user_equipments.append(ue)
            
        # 创建边缘服务器
        edge_frequencies = (
            self.config.get('device_specs', {})
            .get('edge_servers', {})
            .get('cpu_frequencies', [5, 6, 7, 8, 9])
        )
        self.edge_servers = []
        for i in range(self.num_edges):
            es = EdgeServer(i, edge_frequencies[i % len(edge_frequencies)], config=self.config)
            self.edge_servers.append(es)
            
        # 创建云服务器（20 GHz）
        self.cloud_servers = []
        for i in range(self.num_clouds):
            cs = CloudServer(i, config=self.config)
            self.cloud_servers.append(cs)

    @property
    def devices(self):
        """返回所有用户设备（用于LLM咨询）"""
        return self.user_equipments

    @property 
    def edge_servers_list(self):
        """返回边缘服务器列表"""
        return self.edge_servers
        
    @property
    def cloud_servers_list(self):
        """返回云服务器列表"""
        return self.cloud_servers

    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        # 重置所有设备状态
        for ue in self.user_equipments:
            ue.reset()
        for es in self.edge_servers:
            es.reset()
        for cs in self.cloud_servers:
            cs.reset()

        # 重置任务执行跟踪
        self.task_executions.clear()
        self.completed_tasks_history.clear()
        self.global_time = 0.0
        self.episode_step = 0
        self.pending_task_queues = [
            [] for _ in range(self.num_devices)
        ]
        
        
        # 重置统计信息
        self.step_stats = {
            'tasks_completed': 0,
            'tasks_timeout': 0,
            'total_latency': 0.0,
            'total_energy': 0.0,
            'communication_latency': 0.0,
            'computation_latency': 0.0
        }
        
        self.task_generation_state = {
            'total_concurrent_tasks': 0
        }
        
        # 新增：任务完成率统计
        self.task_completion_stats = {
            'total_tasks_generated': 0,        # 总生成任务数
            'tasks_completed_on_time': 0,      # 按时完成的任务数
            'tasks_completed_late': 0,         # 超时完成的任务数
            'tasks_failed': 0,                 # 失败任务数
            'completion_times': [],            # 任务完成时间记录
            'deadline_violations': [],         # 截止时间违反记录
            'timeout_reasons': []              # 超时原因记录
        }
        
        # 生成第一批任务（简化）
        self._generate_new_tasks()

        return self._get_observation(), {}

    def _generate_new_tasks(self):
        if self.debug:
            print(f"\n[Step {self.episode_step}] 生成新任务...")
        if getattr(self.task_generator, 'simple_mode', True):
            device_tasks_dict = self.task_generator.generate_simple_tasks(
                num_devices=self.num_devices,
                step=self.episode_step,
                current_time=self.global_time,
            )
        else:
            device_tasks_dict = self.task_generator.generate_poisson_tasks(
                num_devices=self.num_devices,
                step=self.episode_step,
                current_time=self.global_time,
            )

        if self.physics_version >= 2:
            generated_count = 0
            for device_id in range(self.num_devices):
                for task_data in device_tasks_dict.get(device_id, []):
                    task_data = dict(task_data)
                    task_data['arrival_time'] = self.global_time
                    task_data['arrival_slot'] = self.episode_step
                    task = Task(task_data)
                    task.creation_step = self.episode_step
                    self.pending_task_queues[device_id].append(task)
                    generated_count += 1
            self.task_completion_stats['total_tasks_generated'] += generated_count
            self._refresh_current_tasks()
            if self.debug:
                print(
                    f"   生成结果: {generated_count} 个任务，"
                    f"待决策积压={sum(len(q) for q in self.pending_task_queues)}"
                )
            return

        self.current_tasks = []
        for device_id in range(self.num_devices):
            if device_id in device_tasks_dict and device_tasks_dict[device_id]:
                task_data = device_tasks_dict[device_id][0]
                task = Task(task_data)
                task.creation_step = self.episode_step
                self.current_tasks.append(task)
                self.task_completion_stats['total_tasks_generated'] += 1
            else:
                self.current_tasks.append(None)
        if self.debug:
            valid_tasks = sum(1 for task in self.current_tasks if task is not None)
            print(f"   生成结果: {valid_tasks}/{self.num_devices} 个设备有任务")

    def _refresh_current_tasks(self):
        self.current_tasks = [
            queue[0] if queue else None
            for queue in self.pending_task_queues
        ]

    def _consume_current_tasks(self, has_task_list):
        for device_id, has_task in enumerate(has_task_list):
            if has_task and self.pending_task_queues[device_id]:
                self.pending_task_queues[device_id].pop(0)
        self._refresh_current_tasks()

    def _mark_pending_arrivals_failed(self):
        for device_id, queue in enumerate(self.pending_task_queues):
            for task in queue:
                self.task_completion_stats['tasks_failed'] += 1
                self.task_completion_stats['timeout_reasons'].append(
                    {
                        'task_id': task.task_id,
                        'device_id': device_id,
                        'reason': 'episode_ended_before_admission',
                        'step': self.episode_step,
                    }
                )

    # 删除：时间模式、突发事件、应用混合与系统负载相关生成逻辑

    def step(self, actions, llm_actions=None):
        """
        环境步进

        Args:
            actions: Agent的动作 shape=(num_devices, 4) [α1, α2, α3, edge_id] 或 list
            llm_actions: LLM专家动作 shape=(num_devices, 4) 或 list

        Returns:
            observation, rewards, terminated, truncated, info
        """
        # Phase 2 分支
        if self.physics_version >= 2:
            return self._step_phase2(actions, llm_actions)

        # --- Phase 1 原有逻辑 ---
        if self.debug:
            print(f"\n{'='*80}")
            print(f"开始执行 Step {self.episode_step + 1}")
            print(f"{'='*80}")

        self.episode_step += 1

        # 🆕 简化模式核心逻辑：Memoryless (无记忆)
        # 在每个Step开始时，强制清空所有队列，确保独立性
        if getattr(self.task_generator, 'simple_mode', False):
            self._clear_all_queues()
            if self.debug:
                print("🧹 [Simple Mode] 已清空所有设备任务队列 (Memoryless Update)")

        # 1. 推进全局时间，更新所有设备的任务状态
        self.global_time += self.time_step_duration
        if self.debug:
            print(f"[Step {self.episode_step}] 时间推进到: {self.global_time:.1f}s")
        
        # 更新所有设备的任务执行状态
        self._update_all_devices(self.time_step_duration)
        
        # 🔧 确保actions是NumPy数组格式
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        
        # 2. 显示MADDPG动作解析过程
        if self.debug:
            print(f"\n🔄 MADDPG动作环境交互过程:")
            print(f"{'='*80}")
            print(f"接收到的MADDPG动作维度: {actions.shape}")
            print(f"动作内容:")
        for i, action in enumerate(actions):
            alpha1, alpha2, alpha3, edge_id_raw = action
            edge_id = int(np.clip(edge_id_raw, 0, self.num_edges - 1))
            if self.action_type == 'hybrid':
                def quantize(x):
                    return min(self.hybrid_bins, key=lambda b: abs(b - float(x)))
                alpha1, alpha2, alpha3 = quantize(alpha1), quantize(alpha2), quantize(alpha3)
            
            # 归一化分割比例
            total = alpha1 + alpha2 + alpha3
            if total > 0:
                alpha1_norm, alpha2_norm, alpha3_norm = alpha1/total, alpha2/total, alpha3/total
            else:
                alpha1_norm, alpha2_norm, alpha3_norm = 1.0, 0.0, 0.0
            
            if self.debug:
                print(f"  Device{i}: 原始[{alpha1:.3f}, {alpha2:.3f}, {alpha3:.3f}, {edge_id_raw:.3f}]")
                print(f"           → 解析为[本地:{alpha1_norm:.3f}, 边缘:{alpha2_norm:.3f}, 云端:{alpha3_norm:.3f}, Edge{edge_id}]")
        
        # 3. 核心：执行卸载决策并计算奖励
        rewards = np.zeros(self.num_devices)
        # 初始化列表以防 info 构建失败
        total_latencies = []
        total_energies = []
        communication_latencies = []
        computation_latencies = []
        has_task_list = []

        for i in range(self.num_devices):
            # 🔧 安全地获取单个设备的动作
            if len(actions.shape) > 1:
                action = actions[i]
            else:
                action = actions

            # 执行卸载
            reward, metrics = self._execute_offloading_decision(i, action)
            rewards[i] = reward
            
            # 记录 metrics 到列表
            total_latencies.append(metrics['total_latency'])
            total_energies.append(metrics['total_energy'])
            communication_latencies.append(metrics['communication_latency'])
            computation_latencies.append(metrics['computation_latency'])
            has_task_list.append(
                self.current_tasks is not None
                and i < len(self.current_tasks)
                and self.current_tasks[i] is not None
            )

            # 记录统计信息
            self.step_stats['total_latency'] += metrics['total_latency']
            self.step_stats['total_energy'] += metrics['total_energy']
            
        # 4. 检查是否结束
        max_steps_reached = self.episode_step >= self.max_steps
        terminated = False
        truncated = max_steps_reached

        # 5. 如果还没结束，为下一步生成新任务
        if not (terminated or truncated):
            self._generate_new_tasks()

        # 6. 打印当前状态总结
        if self.debug:
            self._print_step_summary()

        # 构建info字典
        info = {
            'total_latencies': total_latencies,
            'total_energies': total_energies,
            'communication_latencies': communication_latencies,
            'computation_latencies': computation_latencies,
            'episode_step': self.episode_step,
            'global_time': self.global_time,
            'step_stats': self.step_stats.copy(),
            'llm_actions': llm_actions if llm_actions is not None else [],
            # 新增：任务完成率统计
            'task_completion_stats': self.get_task_completion_rate(),
            'deadline_violations': self.task_completion_stats['deadline_violations'].copy(),
            'timeout_reasons': self.task_completion_stats['timeout_reasons'].copy(),
            # 新增：MADDPG动作信息
            'maddpg_actions': actions.tolist(),
            'maddpg_rewards': rewards.tolist(),
            # 新增：每个设备是否有任务的标志
            'has_task_list': has_task_list,
            'ue_wait_times': [ue.calculate_task_load() for ue in self.user_equipments],
            'es_wait_times': [es.calculate_task_load() for es in self.edge_servers]
        }

        return self._get_observation(), rewards, terminated, truncated, info

    def _update_all_devices(self, time_elapsed):
        """更新所有设备的任务执行状态"""
        # 更新端侧设备
        for ue in self.user_equipments:
            ue.update_tasks(time_elapsed)

        # 更新边缘服务器
        for es in self.edge_servers:
            es.update_tasks(time_elapsed)

        # 云服务器（Phase 2 有队列，Phase 1 无操作）
        for cs in self.cloud_servers:
            cs.update_tasks(time_elapsed)

    def _clear_all_queues(self):
        """Clear legacy queues in simple-mode Phase 1 experiments."""
        for ue in self.user_equipments:
            ue.reset()
        for es in self.edge_servers:
            es.reset()
        for cs in self.cloud_servers:
            cs.reset()

    # ------------------------------------------------------------------
    # Phase 2: step 实现
    # ------------------------------------------------------------------

    def _step_phase2(self, actions, llm_actions=None):
        """Phase 2 环境步进。

        特点：
          - 不清空队列（跨时隙演化）
          - 使用 Shannon 无线信道速率
          - 有线回传链路 (ES→CS)
          - 下行结果返回
          - 并行分支最大完成时延
          - DVFS 能耗模型
        """
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 1:
            if self.num_devices != 1:
                raise ValueError(
                    "joint actions must have shape (num_devices, 4)"
                )
            actions = actions.reshape(1, -1)
        if actions.shape != (self.num_devices, 4):
            raise ValueError(
                f"expected actions shape {(self.num_devices, 4)}, "
                f"got {actions.shape}"
            )

        # 1. 在时隙起点对联合动作做无副作用评估。
        evaluation = self.evaluate_action(actions.tolist())
        per_ue_detail = evaluation.detail['per_ue']
        rewards = np.zeros(self.num_devices, dtype=np.float64)
        has_task_list = [
            task is not None for task in self.current_tasks
        ]
        total_latencies = list(evaluation.latency_per_ue)
        total_energies = list(evaluation.energy_per_ue)
        communication_latencies = []
        computation_latencies = []
        user_energies = []

        # 2. 应用与 evaluator 相同的调度计划。
        for device_idx, task in enumerate(self.current_tasks):
            detail = per_ue_detail[device_idx]
            if task is None or detail is None:
                communication_latencies.append(0.0)
                computation_latencies.append(0.0)
                user_energies.append(0.0)
                continue

            normalized_action = detail['normalized_action']
            task.set_split_ratios(*normalized_action[:3])
            self._enqueue_phase2_plan(device_idx, task, detail)
            latency = total_latencies[device_idx]
            system_energy = total_energies[device_idx]
            reward_energy = (
                detail['user_energy']
                if self.reward_energy_scope == 'user'
                else system_energy
            )
            self._check_task_completion(task, latency)
            rewards[device_idx] = self._calculate_reward(
                latency,
                reward_energy,
                detail['baseline_latency'],
                detail['baseline_energy'],
                task.deadline,
                normalized_action[3],
            )
            communication_latencies.append(
                detail['communication_latency']
            )
            computation_latencies.append(
                detail['computation_latency']
            )
            user_energies.append(detail['user_energy'])
            self.step_stats['total_latency'] += latency
            self.step_stats['total_energy'] += system_energy
            self.step_stats['communication_latency'] += detail[
                'communication_latency'
            ]
            self.step_stats['computation_latency'] += detail[
                'computation_latency'
            ]

        # 3. 当前任务已被调度，随后推进一个完整时隙的服务。
        self._consume_current_tasks(has_task_list)
        self._update_all_devices(self.time_step_duration)
        self.global_time += self.time_step_duration
        self.episode_step += 1

        # 4. 检查是否结束
        max_steps_reached = self.episode_step >= self.max_steps
        terminated = False
        truncated = max_steps_reached

        # 5. 为下一步生成新任务
        if not (terminated or truncated):
            self._generate_new_tasks()
        else:
            self._mark_pending_arrivals_failed()

        # 6. info
        info = {
            'total_latencies': total_latencies,
            'total_energies': total_energies,
            'user_energies': user_energies,
            'energy_scope': 'system',
            'reward_energy_scope': self.reward_energy_scope,
            'communication_latencies': communication_latencies,
            'computation_latencies': computation_latencies,
            'episode_step': self.episode_step,
            'global_time': self.global_time,
            'step_stats': self.step_stats.copy(),
            'llm_actions': llm_actions if llm_actions is not None else [],
            'task_completion_stats': self.get_task_completion_rate(),
            'deadline_violations': self.task_completion_stats['deadline_violations'].copy(),
            'timeout_reasons': self.task_completion_stats['timeout_reasons'].copy(),
            'maddpg_actions': actions.tolist(),
            'maddpg_rewards': rewards.tolist(),
            'has_task_list': has_task_list,
            'ue_wait_times': [ue.calculate_task_load() for ue in self.user_equipments],
            'es_wait_times': [es.calculate_task_load() for es in self.edge_servers],
            'cs_wait_times': [cs.calculate_task_load() for cs in self.cloud_servers],
            'queue_backlogs': [
                int(ue.current_execution is not None) + len(ue.task_queue)
                for ue in self.user_equipments
            ],
            'arrival_queue_backlogs': [
                len(queue) for queue in self.pending_task_queues
            ],
            'evaluation': evaluation,
        }

        return self._get_observation(), rewards, terminated, truncated, info

    def _execute_offloading_decision_phase2(self, device_idx, action):
        """Phase 2 卸载决策。

        使用 Shannon 无线信道、有线回传、DVFS 能耗和并行分支最大时延。
        """
        if self.current_tasks is None or device_idx >= len(self.current_tasks):
            return 0.0, {
                'total_latency': 0.0, 'total_energy': 0.0,
                'communication_latency': 0.0, 'computation_latency': 0.0,
            }

        task = self.current_tasks[device_idx]
        if task is None:
            return 0.0, {
                'total_latency': 0.0, 'total_energy': 0.0,
                'communication_latency': 0.0, 'computation_latency': 0.0,
            }

        from .snapshot import capture_snapshot, evaluate

        snapshot = capture_snapshot(self)
        for index, ue_snapshot in enumerate(snapshot.ues):
            if index != device_idx:
                ue_snapshot.current_task = None
        joint_action = [[1.0, 0.0, 0.0, 0.0] for _ in range(self.num_devices)]
        joint_action[device_idx] = list(action)
        evaluation = evaluate(snapshot, joint_action)
        detail = evaluation.detail['per_ue'][device_idx]
        normalized_action = detail['normalized_action']
        task.set_split_ratios(*normalized_action[:3])
        self._enqueue_phase2_plan(device_idx, task, detail)
        total_latency = evaluation.latency_per_ue[device_idx]
        total_energy = evaluation.energy_per_ue[device_idx]
        self._check_task_completion(task, total_latency)
        reward_energy = (
            detail['user_energy']
            if self.reward_energy_scope == 'user'
            else total_energy
        )
        reward = self._calculate_reward(
            total_latency,
            reward_energy,
            detail['baseline_latency'],
            detail['baseline_energy'],
            task.deadline,
            normalized_action[3],
        )

        return reward, {
            'total_latency': total_latency,
            'total_energy': total_energy,
            'communication_latency': detail['communication_latency'],
            'computation_latency': detail['computation_latency'],
            'local_baseline': (
                detail['baseline_latency'],
                detail['baseline_energy'],
            ),
        }

    def _enqueue_phase2_plan(self, device_idx, task, detail):
        """Apply a plan produced by snapshot.evaluate to real compute queues."""
        branches = detail['branches']
        if 'local' in branches:
            branch = branches['local']
            self.user_equipments[device_idx].add_task(
                f"{task.task_id}_local",
                branch['cycles'],
                branch['release_time'],
            )
        if 'edge' in branches:
            branch = branches['edge']
            self.edge_servers[branch['edge_id']].add_task(
                f"{task.task_id}_edge",
                branch['cycles'],
                branch['release_time'],
            )
        if 'cloud' in branches:
            branch = branches['cloud']
            self.cloud_servers[branch['cloud_id']].add_task(
                f"{task.task_id}_cloud",
                branch['cycles'],
                branch['release_time'],
            )

    def _execute_offloading_decision(self, device_idx, action):
        """执行单个设备的卸载决策 - 考虑差异化通信延迟"""
        # 🆕 检查是否有任务需要处理
        if self.current_tasks is None or device_idx >= len(self.current_tasks):
            # 没有任务，返回零奖励
            return 0.0, {
                'total_latency': 0.0,
                'total_energy': 0.0, 
                'communication_latency': 0.0,
                'computation_latency': 0.0,
                'local_baseline': (0.0, 0.0)
            }
        
        task = self.current_tasks[device_idx]
        if task is None:
            # 该设备没有任务，返回零奖励
            if self.debug:
                print(f"  Device{device_idx}: 无任务分配")
            return 0.0, {
                'total_latency': 0.0,
                'total_energy': 0.0,
                'communication_latency': 0.0, 
                'computation_latency': 0.0,
                'local_baseline': (0.0, 0.0)
            }
        
        # 解析动作
        alpha1, alpha2, alpha3, edge_id = action
        edge_id = int(np.clip(edge_id, 0, self.num_edges - 1))
        
        # 归一化分割比例，确保和为1
        total = alpha1 + alpha2 + alpha3
        if total > 0:
            alpha1, alpha2, alpha3 = alpha1/total, alpha2/total, alpha3/total
        else:
            alpha1, alpha2, alpha3 = 1.0, 0.0, 0.0  # 默认全本地
            
        # 获取设备和任务
        ue = self.user_equipments[device_idx]
        task.set_split_ratios(alpha1, alpha2, alpha3)
        
        if self.debug:
            print(f"  Device{device_idx}: Task{task.task_id} 分割比例 "
                  f"[本地:{alpha1:.2f}, 边缘:{alpha2:.2f}, 云端:{alpha3:.2f}] → Edge{edge_id}")
        
        # 分割任务并分配到不同节点
        total_latency, total_energy, comm_latency, comp_latency = self._schedule_task_execution_optimized(
            ue, task, edge_id, device_idx)
        
        # 计算本地基准
        baseline_latency, baseline_energy = self._calculate_local_baseline(ue, task)
        
        # 计算奖励函数
        reward = self._calculate_reward(
            total_latency, total_energy, baseline_latency, baseline_energy, task.deadline, edge_id)
        
        metrics = {
            'total_latency': total_latency,
            'total_energy': total_energy,
            'communication_latency': comm_latency,
            'computation_latency': comp_latency,
            'local_baseline': (baseline_latency, baseline_energy)
        }
        
        return reward, metrics

    def _schedule_task_execution_optimized(self, ue, task, edge_id, device_idx):
        """
        优化的任务调度 - 考虑差异化通信延迟和任务负载
        
        返回: (总延迟, 总能耗, 通信延迟, 计算延迟)
        """
        workloads = task.get_split_workloads()  # [本地, 边缘, 云端]工作负载
        data_sizes = task.get_split_data_sizes()  # [本地, 边缘, 云端]数据大小
        
        latencies = []
        energies = []
        comm_latencies = []
        comp_latencies = []
        
        # 1. 本地计算部分
        if workloads[0] > 0:
            # 获取当前任务负载（等待时间）
            current_load = ue.calculate_task_load()
            exec_time = ue.calculate_execution_time(workloads[0])
            energy = ue.calculate_energy_consumption(workloads[0])
            
            # 添加任务到设备队列
            ue.add_task(f"{task.task_id}_local", workloads[0], self.global_time)
            
            total_time = current_load + exec_time
            latencies.append(total_time)
            energies.append(energy)
            comm_latencies.append(0.0)  # 本地无通信延迟
            comp_latencies.append(exec_time)
            

            
            if self.debug:
                print(f"    本地执行: {workloads[0]/1e9:.2f}Gcycles, 等待{current_load:.2f}s + 计算{exec_time:.2f}s")
        else:
            latencies.append(0.0)
            energies.append(0.0)
            comm_latencies.append(0.0)
            comp_latencies.append(0.0)
            
        # 2. 边缘计算部分
        if workloads[1] > 0:
            es = self.edge_servers[edge_id]
            
            # 通信延迟（UE到边缘）
            comm_time = ue.calculate_transmission_time_to_edge(data_sizes[1])
            comm_energy = ue.calculate_transmission_energy(comm_time)
            
            # 边缘服务器的任务负载（等待时间）
            edge_load = es.calculate_task_load()
            exec_time = es.calculate_execution_time(workloads[1])
            
            # 添加任务到边缘队列
            es.add_task(f"{task.task_id}_edge", workloads[1], self.global_time + comm_time)
            
            total_time = comm_time + edge_load + exec_time
            latencies.append(total_time)
            energies.append(comm_energy)  # 只计算UE的传输能耗
            comm_latencies.append(comm_time)
            comp_latencies.append(exec_time)
            

            
            if self.debug:
                print(f"    边缘执行: {workloads[1]/1e9:.2f}Gcycles → ES{edge_id}, 通信{comm_time:.2f}s + 等待{edge_load:.2f}s + 计算{exec_time:.2f}s")
        else:
            latencies.append(0.0)
            energies.append(0.0)
            comm_latencies.append(0.0)
            comp_latencies.append(0.0)
            
        # 3. 云计算部分（差异化通信延迟）
        if workloads[2] > 0:
            cs = self.cloud_servers[0]
            
            # 通信延迟（UE→边缘→云，总延迟更高）
            comm_time = ue.calculate_transmission_time_to_cloud(data_sizes[2])
            comm_energy = ue.calculate_transmission_energy(comm_time * 0.6)  # 部分传输时间的能耗
            
            # 云计算时间（无等待，资源无限）
            exec_time = cs.calculate_execution_time(workloads[2])
            
            total_time = comm_time + exec_time
            latencies.append(total_time)
            energies.append(comm_energy)
            comm_latencies.append(comm_time)
            comp_latencies.append(exec_time)
            

            
            if self.debug:
                print(f"    云端执行: {workloads[2]/1e9:.2f}Gcycles → Cloud, 通信{comm_time:.2f}s + 计算{exec_time:.2f}s")
        else:
            latencies.append(0.0)
            energies.append(0.0)
            comm_latencies.append(0.0)
            comp_latencies.append(0.0)
            
        # 计算总延迟（取最大值，因为可以并行执行）和总能耗（求和）
        total_latency = max(latencies)
        total_energy = sum(energies)
        total_comm_latency = max(comm_latencies)
        total_comp_latency = max(comp_latencies)
        
        # 检查任务完成状态
        self._check_task_completion(task, total_latency)
        
        return total_latency, total_energy, total_comm_latency, total_comp_latency

    def _calculate_local_baseline(self, ue, task):
        """计算全本地执行的基准时延和能耗"""
        current_load = ue.calculate_task_load()
        exec_time = ue.calculate_execution_time(task.task_workload)
        energy = ue.calculate_energy_consumption(task.task_workload)
        return current_load + exec_time, energy

    def _calculate_reward(self, offload_latency, offload_energy,
                         baseline_latency, baseline_energy, deadline, edge_id=None):
        if self.reward_type == 'normalized':
            lat_improve = (baseline_latency - offload_latency) / baseline_latency if baseline_latency > 1e-8 else 0.0
            eng_improve = (baseline_energy - offload_energy) / baseline_energy if baseline_energy > 1e-8 else 0.0
            reward = self.latency_weight * lat_improve + self.energy_weight * eng_improve
        elif self.reward_type == 'weighted_sum':
            reward = self.latency_weight * (1.0 / (offload_latency + 1e-8)) + self.energy_weight * (1.0 / (offload_energy + 1e-8))
        else:
            latency_term = 1.0 / offload_latency if offload_latency > 1e-8 else 0.0
            energy_term = 1.0 / offload_energy if offload_energy > 1e-8 else 0.0
            reward = 0.5 * latency_term + 0.5 * energy_term
        if deadline and offload_latency > deadline:
            reward -= self.deadline_penalty * (offload_latency - deadline)
        return float(reward)

    def _print_step_summary(self):
        """打印当前步骤的状态总结"""
        print(f"\n[Step {self.episode_step}] 状态总结:")
        print(f"  已完成任务: {self.step_stats['tasks_completed']}")
        print(f"  超时任务: {self.step_stats['tasks_timeout']}")
        
        # 打印设备负载状态
        print("  端侧设备任务负载:")
        for i in range(min(3, self.num_devices)):
            ue = self.user_equipments[i]
            load = ue.calculate_task_load()
            print(f"    UE{i}: 任务负载={load:.1f}s")
        
        print("  边缘服务器负载:")
        for i, es in enumerate(self.edge_servers):
            load = es.calculate_task_load()
            print(f"    ES{i}: CPU={es.cpu_frequency}GHz, 负载={load:.1f}s")

    def _get_observation(self):
        """
        获取环境观察（简化版）
        
        状态组成：
        1. UE状态：CPU频率、任务负载
        2. ES状态：CPU频率、任务负载  
        3. CS状态：CPU频率
        4. 任务状态：类型、数据大小、CPU周期、截止时间、剩余时间、紧急程度
        """
        if self.physics_version >= 2:
            return self._get_phase2_observation()

        observation = []
        
        # 1. UE状态 (每个设备2个特征)
        for ue in self.user_equipments:
            ue_state = ue.get_state()  # [CPU频率, 任务负载]
            if isinstance(ue_state, (list, tuple)):
                observation.extend(ue_state)
            else:
                observation.append(ue_state)
            
        # 2. ES状态 (每个服务器2个特征)
        for es in self.edge_servers:
            es_state = es.get_state()  # [CPU频率, 任务负载]
            if isinstance(es_state, (list, tuple)):
                observation.extend(es_state)
            else:
                observation.append(es_state)
            
        # 3. CS状态 (每个服务器1个特征)
        for cs in self.cloud_servers:
            cs_state = cs.get_state()  # [CPU频率]
            if isinstance(cs_state, (list, tuple)):
                observation.extend(cs_state)
            else:
                observation.append(cs_state)
            
        # 4. 任务状态 (每个任务2个特征)
        if self.current_tasks:
            for i, task in enumerate(self.current_tasks):
                if task is not None:
                    # 数据大小归一化
                    data_size_norm = min(task.task_data_size / 200.0, 1.0)
                    # CPU周期归一化
                    workload_norm = min(task.task_workload / 1e10, 1.0)
                    task_state = [
                        data_size_norm,
                        workload_norm
                    ]
                else:
                    # 没有任务，填充零
                    task_state = [0.0] * 2
                
                observation.extend(task_state)
        else:
            # 如果没有任务，填充零
            for _ in range(self.num_devices):
                observation.extend([0.0] * 2)
                
        return np.array(observation, dtype=np.float32)

    def _get_phase2_observation(self):
        """Encode queues, channels, and task semantics for Phase 2."""
        observation = []

        for device_id, ue in enumerate(self.user_equipments):
            arrival_backlog_norm = min(
                len(self.pending_task_queues[device_id]) / 10.0, 1.0
            )
            observation.extend([*ue.get_state(), arrival_backlog_norm])
        for es in self.edge_servers:
            observation.extend(es.get_state())
        for cs in self.cloud_servers:
            load_norm = min(cs.calculate_task_load() / 300.0, 1.0)
            observation.extend(
                [min(cs.cpu_frequency / 20.0, 1.0), load_norm]
            )

        rate_scale = float(
            self.config.get('channel', {}).get(
                'rate_normalization_bps', 100e6
            )
        )
        if rate_scale <= 0:
            raise ValueError("channel.rate_normalization_bps must be positive")
        for ue in self.user_equipments:
            for edge_id in range(self.num_edges):
                distance = ue.distance_to_edges[edge_id]
                rate = self.channel.achievable_rate(distance)
                observation.append(min(rate / rate_scale, 1.0))

        semantic_types = TaskGenerator.SEMANTIC_TYPES
        for task in self.current_tasks:
            if task is None:
                observation.extend([0.0] * self.phase2_task_feature_dim)
                continue
            semantic_one_hot = [
                float(task.semantic_type == semantic_type)
                for semantic_type in semantic_types
            ]
            elapsed = max(self.global_time - task.arrival_time, 0.0)
            deadline_slack = max(task.deadline - elapsed, 0.0)
            observation.extend(
                [
                    min(task.task_data_size / 200.0, 1.0),
                    min(task.task_workload / 100e9, 1.0),
                    min(deadline_slack / 300.0, 1.0),
                    min(task.output_ratio, 1.0),
                    min(task.priority / 3.0, 1.0),
                    *semantic_one_hot,
                ]
            )

        result = np.asarray(observation, dtype=np.float32)
        if result.shape != self.observation_space.shape:
            raise RuntimeError(
                f"Phase 2 observation shape {result.shape} does not match "
                f"declared {self.observation_space.shape}"
            )
        return result

    def extract_agent_state(self, global_state, agent_id):
        """
        正确提取单个Agent的观察状态
        
        Args:
            global_state: 全局状态向量 (101维)
            agent_id: Agent ID (0到num_devices-1)
            
        Returns:
            agent_state: Agent的局部状态 (20维)
            
        状态结构说明:
        - 全局状态: [UE状态(30维) + ES状态(10维) + CS状态(1维) + 任务状态(60维)] = 101维
        - Agent状态: [自己UE状态(3维) + 所有ES状态(10维) + CS状态(1维) + 自己任务状态(6维)] = 20维
        """
        if agent_id < 0 or agent_id >= self.num_devices:
            raise ValueError(f"Agent ID {agent_id} 超出范围 [0, {self.num_devices-1}]")
        if self.physics_version >= 2:
            return self._extract_phase2_agent_state(global_state, agent_id)
        
        # 状态分割点计算
        ue_states_end = self.num_devices * 2
        es_states_end = ue_states_end + self.num_edges * 2
        cs_states_end = es_states_end + self.num_clouds * 1
        task_states_end = cs_states_end + self.num_devices * 2
        
        # 1. 提取当前Agent的UE状态 (2维)
        agent_ue_start = agent_id * 2
        agent_ue_state = global_state[agent_ue_start:agent_ue_start + 2]
        
        # DEBUG: Check if elements are sequences
        if len(agent_ue_state) > 0 and (isinstance(agent_ue_state[0], (list, tuple, np.ndarray))):
             print(f"DEBUG: agent_ue_state elements are sequences! {agent_ue_state}")
             # Flatten if necessary (though global_state should have been flat)
             # This suggests global_state was constructed with nested lists
             
        # 2. 提取所有边缘服务器状态 (10维) - 共享信息
        es_state = global_state[ue_states_end:es_states_end]
        
        # 3. 提取云服务器状态 (1维) - 共享信息  
        cs_state = global_state[es_states_end:cs_states_end]
        
        # 4. 提取当前Agent的任务状态 (2维)
        agent_task_start = cs_states_end + agent_id * 2
        agent_task_state = global_state[agent_task_start:agent_task_start + 2]
        
        # 组合Agent的完整状态
        # Ensure all components are flat numpy arrays of float32
        agent_ue_state = np.array(agent_ue_state, dtype=np.float32).flatten()
        es_state = np.array(es_state, dtype=np.float32).flatten()
        cs_state = np.array(cs_state, dtype=np.float32).flatten()
        agent_task_state = np.array(agent_task_state, dtype=np.float32).flatten()
        
        agent_state = np.concatenate([
            agent_ue_state,    # 3维：自己的设备状态
            es_state,          # 10维：所有边缘服务器状态  
            cs_state,          # 1维：云服务器状态
            agent_task_state   # 2维：自己的任务状态
        ])
        
        return agent_state.astype(np.float32)

    def _extract_phase2_agent_state(self, global_state, agent_id):
        global_state = np.asarray(global_state, dtype=np.float32).reshape(-1)
        ue_end = self.num_devices * self.phase2_ue_feature_dim
        es_end = ue_end + self.num_edges * 2
        cs_end = es_end + self.num_clouds * 2
        channel_end = cs_end + self.num_devices * self.num_edges

        own_ue_start = agent_id * self.phase2_ue_feature_dim
        own_ue = global_state[
            own_ue_start:own_ue_start + self.phase2_ue_feature_dim
        ]
        edge_states = global_state[ue_end:es_end]
        cloud_states = global_state[es_end:cs_end]
        own_channel_start = cs_end + agent_id * self.num_edges
        own_channels = global_state[
            own_channel_start:own_channel_start + self.num_edges
        ]
        own_task_start = (
            channel_end + agent_id * self.phase2_task_feature_dim
        )
        own_task = global_state[
            own_task_start:own_task_start + self.phase2_task_feature_dim
        ]
        return np.concatenate(
            [own_ue, edge_states, cloud_states, own_channels, own_task]
        ).astype(np.float32)

    def get_agent_state_dim(self):
        """获取单个Agent的状态维度
        
        Agent状态结构：
        - 自己UE状态: 2维 (CPU频率, 任务负载)
        - 所有ES状态: 2×5=10维 (CPU频率, 任务负载)
        - CS状态: 1维 (CPU频率)
        - 自己任务状态: 2维 (数据大小, CPU周期)
        
        总计: 2 + 10 + 1 + 2 = 15维
        """
        if self.physics_version >= 2:
            return (
                self.phase2_ue_feature_dim
                + self.num_edges * 2
                + self.num_clouds * 2
                + self.num_edges
                + self.phase2_task_feature_dim
            )
        return 2 + (self.num_edges * 2) + (self.num_clouds * 1) + 2

    def get_device_info(self):
        """获取设备信息（用于LLM咨询）"""
        device_info = []
        for i, ue in enumerate(self.user_equipments):
            info = {
                'device_id': i,
                'cpu_frequency': ue.cpu_frequency,
                'task_load': ue.calculate_task_load()
            }
            device_info.append(info)
        return device_info

    def get_edge_info(self):
        """获取边缘服务器信息（用于LLM咨询）"""
        edge_info = []
        for i, es in enumerate(self.edge_servers):
            info = {
                'server_id': i,
                'cpu_frequency': es.cpu_frequency,
                'task_load': es.calculate_task_load()
            }
            edge_info.append(info)
        return edge_info

    def get_cloud_info(self):
        """获取云服务器信息（用于LLM咨询）"""
        cloud_info = []
        for i, cs in enumerate(self.cloud_servers):
            info = {
                'server_id': i,
                'cpu_frequency': cs.cpu_frequency,
                'is_available': True  # 云资源始终可用
            }
            cloud_info.append(info)
        return cloud_info

    def get_current_tasks_info(self):
        """获取当前任务信息（用于LLM咨询）"""
        tasks_info = []
        if self.current_tasks:
            for i, task in enumerate(self.current_tasks):
                # 🔧 修复：过滤掉None任务
                if task is not None:
                    info = {
                        'task_id': task.task_id,
                        'device_id': i,
                        'data_size': task.task_data_size,  # 修复属性名称
                        'cpu_cycles': task.task_workload,     # 修复属性名称
                        'deadline': task.deadline,
                        'semantic_type': task.semantic_type,
                        'output_ratio': task.output_ratio,
                        'priority': task.priority,
                        'arrival_time': task.arrival_time,
                    }
                    tasks_info.append(info)
        return tasks_info

    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            print("\n=== 环境状态（简化版）===")
            print(f"Episode Step: {self.episode_step}")
            print(f"Global Time: {self.global_time:.1f}s")
            
            # 显示设备状态
            print("\n设备状态:")
            for i in range(min(3, self.num_devices)):
                ue = self.user_equipments[i]
                load = ue.calculate_task_load()
                print(f"  UE{i}: CPU={ue.cpu_frequency:.1f}GHz, "
                      f"负载={load:.1f}s")
            
            # 显示边缘服务器状态
            print("\n边缘服务器状态:")
            for i, es in enumerate(self.edge_servers):
                load = es.calculate_task_load()
                print(f"  ES{i}: CPU={es.cpu_frequency}GHz, 负载={load:.1f}s")
            
            # 显示云服务器状态
            print("\n云服务器状态:")
            for i, cs in enumerate(self.cloud_servers):
                print(f"  CS{i}: CPU={cs.cpu_frequency}GHz (资源无限)")

    def close(self):
        """关闭环境"""
        pass

    # ------------------------------------------------------------------
    # Phase 2: Snapshot 与纯函数式评估
    # ------------------------------------------------------------------

    def capture_snapshot(self):
        """捕获当前环境快照（只读），用于 verifier 和反事实评估。

        Returns:
            EnvSnapshot 实例
        """
        from .snapshot import capture_snapshot
        return capture_snapshot(self)

    def evaluate_action(self, joint_action):
        """纯函数式动作评估，不修改环境状态。

        Args:
            joint_action: 每个 UE 的动作 [alpha1, alpha2, alpha3, edge_id]

        Returns:
            EvaluationResult 实例
        """
        from .snapshot import evaluate, capture_snapshot
        snapshot = capture_snapshot(self)
        return evaluate(snapshot, joint_action)

    def _check_task_completion(self, task, actual_latency):
        """
        检查并记录任务完成状态
        
        Args:
            task: 任务对象
            actual_latency: 实际完成延迟
        """
        # 🆕 更新并发任务计数
        if self.task_generation_state['total_concurrent_tasks'] > 0:
            self.task_generation_state['total_concurrent_tasks'] -= 1
        
        # 记录任务完成时间
        completion_time = task.arrival_time + actual_latency
        self.task_completion_stats['completion_times'].append(completion_time)
        
        # 检查是否超过截止时间
        if actual_latency <= task.deadline:
            # 按时完成
            self.task_completion_stats['tasks_completed_on_time'] += 1
            self.step_stats['tasks_completed'] += 1
            
            if self.debug:
                print(f"    ✅ 任务{task.task_id}按时完成: {actual_latency:.2f}s <= {task.deadline:.2f}s")
        else:
            # 超时完成
            overtime = actual_latency - task.deadline
            self.task_completion_stats['tasks_completed_late'] += 1
            self.task_completion_stats['deadline_violations'].append({
                'task_id': task.task_id,
                'task_type': task.task_type,
                'deadline': task.deadline,
                'actual_time': actual_latency,
                'overtime': overtime,
                'step': self.episode_step
            })
            self.step_stats['tasks_timeout'] += 1
            
            # 记录超时原因
            if overtime < 1.0:
                reason = "轻微超时"
            elif overtime < 5.0:
                reason = "中度超时"
            else:
                reason = "严重超时"
            
            self.task_completion_stats['timeout_reasons'].append({
                'task_id': task.task_id,
                'reason': reason,
                'overtime': overtime
            })
            
            if self.debug:
                print(f"    ⚠️ 任务{task.task_id}超时完成: {actual_latency:.2f}s > {task.deadline:.2f}s (超时{overtime:.2f}s)")

    def get_task_completion_rate(self):
        """
        计算任务完成率
        
        Returns:
            dict: 包含各种完成率指标的字典
        """
        stats = self.task_completion_stats
        total_attempted = stats['tasks_completed_on_time'] + stats['tasks_completed_late'] + stats['tasks_failed']
        
        if total_attempted == 0:
            return {
                'overall_completion_rate': 1.0,
                'on_time_completion_rate': 1.0,
                'timeout_rate': 0.0,
                'failure_rate': 0.0,
                'total_tasks': 0,
                'completed_on_time': 0,
                'completed_late': 0,
                'failed': 0
            }
        
        # 计算各项指标
        overall_completion_rate = (stats['tasks_completed_on_time'] + stats['tasks_completed_late']) / total_attempted
        on_time_completion_rate = stats['tasks_completed_on_time'] / total_attempted
        timeout_rate = stats['tasks_completed_late'] / total_attempted
        failure_rate = stats['tasks_failed'] / total_attempted
        
        return {
            'overall_completion_rate': overall_completion_rate,        # 总完成率（按时+超时）
            'on_time_completion_rate': on_time_completion_rate,        # 按时完成率
            'timeout_rate': timeout_rate,                              # 超时完成率
            'failure_rate': failure_rate,                              # 失败率
            'total_tasks': total_attempted,                            # 总任务数
            'completed_on_time': stats['tasks_completed_on_time'],     # 按时完成数
            'completed_late': stats['tasks_completed_late'],           # 超时完成数
            'failed': stats['tasks_failed'],                           # 失败任务数
            'avg_completion_time': np.mean(stats['completion_times']) if stats['completion_times'] else 0,
            'avg_overtime': np.mean([v['overtime'] for v in stats['deadline_violations']]) if stats['deadline_violations'] else 0
        }
