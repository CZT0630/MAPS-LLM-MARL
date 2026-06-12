# environment/device_models.py
"""
云边端设备模型 — Phase 1 + Phase 2 双版本兼容

Phase 1 (environment.physics_version < 2):
  - UE: CPU频率 + 任务负载，常数传输速率
  - ES: CPU频率 + 任务负载
  - CS: 资源无限
Phase 2 (environment.physics_version >= 2):
  - UE/ES/CS: 显式 delta_t 时隙、有限资源、DVFS 能耗
  - 无线信道速率模型 (Shannon)
  - 有线回传链路 (ES→CS)
  - 下行结果返回

NOTE: 本文件由人工完成 Phase 2 初稿，已由 Codex 于 2026-06-11 审查并修复。
"""
import random


class TaskExecution:
    """任务执行记录"""
    def __init__(
        self,
        task_id,
        task_workload,
        start_time,
        execution_time,
        sequence=0,
    ):
        self.task_id = task_id
        self.task_workload = task_workload
        self.start_time = float(start_time)
        self.release_time = float(start_time)
        self.execution_time = execution_time
        self.remaining_time = execution_time
        self.sequence = int(sequence)
        self.started_at = None
        self.completed = False

    @property
    def remaining_cycles(self):
        if self.execution_time <= 0:
            return 0.0
        return self.task_workload * max(self.remaining_time, 0.0) / self.execution_time


class _Phase2QueueMixin:
    """Shared non-preemptive FCFS compute queue for Phase 2 nodes."""

    def _init_phase2_queue(self):
        self.task_queue = []
        self.current_execution = None
        self.queue_time = 0.0
        self._next_queue_sequence = 0

    def _reset_phase2_queue(self):
        self.task_queue.clear()
        self.current_execution = None
        self.queue_time = 0.0
        self._next_queue_sequence = 0

    def _phase2_add_task(self, task_id, cpu_cycles, release_time):
        execution = TaskExecution(
            task_id=task_id,
            task_workload=cpu_cycles,
            start_time=release_time,
            execution_time=self.calculate_execution_time(cpu_cycles),
            sequence=self._next_queue_sequence,
        )
        self._next_queue_sequence += 1

        active = (
            self.current_execution is not None
            and self.current_execution.started_at is not None
            and not self.current_execution.completed
        )
        if active:
            self.task_queue.append(execution)
            self.task_queue.sort(key=lambda item: (item.release_time, item.sequence))
            return

        waiting = []
        if self.current_execution is not None and not self.current_execution.completed:
            waiting.append(self.current_execution)
        waiting.extend(task for task in self.task_queue if not task.completed)
        waiting.append(execution)
        waiting.sort(key=lambda item: (item.release_time, item.sequence))
        self.current_execution = waiting[0]
        self.task_queue = waiting[1:]

    def _phase2_update_tasks(self, time_elapsed):
        time_elapsed = float(time_elapsed)
        if time_elapsed < 0:
            raise ValueError("time_elapsed must be non-negative")

        interval_end = self.queue_time + time_elapsed
        cursor = self.queue_time
        epsilon = 1e-12

        while cursor < interval_end - epsilon:
            if self.current_execution is None:
                if not self.task_queue:
                    break
                self.task_queue.sort(
                    key=lambda item: (item.release_time, item.sequence)
                )
                self.current_execution = self.task_queue.pop(0)

            task = self.current_execution
            if task.completed:
                self.current_execution = None
                continue

            service_start = max(cursor, task.release_time)
            if service_start >= interval_end - epsilon:
                break
            if task.started_at is None:
                task.started_at = service_start

            available_service = interval_end - service_start
            if task.remaining_time <= available_service + epsilon:
                cursor = service_start + max(task.remaining_time, 0.0)
                task.remaining_time = 0.0
                task.completed = True
                self.current_execution = None
                continue

            task.remaining_time -= available_service
            cursor = interval_end

        self.queue_time = interval_end

    def _phase2_schedule_entries(self):
        active = (
            self.current_execution is not None
            and self.current_execution.started_at is not None
            and not self.current_execution.completed
        )
        if active:
            cursor = self.queue_time + self.current_execution.remaining_time
            waiting = [
                task for task in self.task_queue if not task.completed
            ]
        else:
            cursor = self.queue_time
            waiting = []
            if self.current_execution is not None and not self.current_execution.completed:
                waiting.append(self.current_execution)
            waiting.extend(task for task in self.task_queue if not task.completed)
        waiting.sort(key=lambda item: (item.release_time, item.sequence))
        return cursor, waiting

    def get_available_time(self):
        """Absolute simulation time when all admitted compute work completes."""
        cursor, waiting = self._phase2_schedule_entries()
        for task in waiting:
            cursor = max(cursor, task.release_time) + task.remaining_time
        return cursor

    def predict_completion_time(self, cpu_cycles, release_time):
        """Predict absolute completion time without modifying the queue."""
        cursor, waiting = self._phase2_schedule_entries()
        candidate_sequence = self._next_queue_sequence
        entries = [
            (task.release_time, task.sequence, task.remaining_time, False)
            for task in waiting
        ]
        entries.append(
            (
                float(release_time),
                candidate_sequence,
                self.calculate_execution_time(cpu_cycles),
                True,
            )
        )
        entries.sort(key=lambda item: (item[0], item[1]))
        for release, _, duration, is_candidate in entries:
            cursor = max(cursor, release) + duration
            if is_candidate:
                return cursor
        raise RuntimeError("candidate task was not scheduled")

    def get_queue_cycles(self):
        tasks = []
        if self.current_execution is not None and not self.current_execution.completed:
            tasks.append(self.current_execution)
        tasks.extend(task for task in self.task_queue if not task.completed)
        return sum(task.remaining_cycles for task in tasks)


class UserEquipment(_Phase2QueueMixin):
    """
    端侧设备（User Equipment, UE）
    简化状态：[CPU频率, 任务负载]
    """

    def __init__(self, device_id, cpu_frequency=None, config=None):
        self.device_id = device_id
        self.config = config or {}
        cfg_env = self.config.get('environment', {})
        cfg_net = self.config.get('network', {})
        cfg_dev = self.config.get('device_specs', {}).get('user_equipment', {})
        self.model_version = self.config.get('model_version', 1)
        self.physics_version = int(cfg_env.get('physics_version', 1))

        # CPU频率：0.5-1.0 GHz异构配置
        if cpu_frequency is None:
            self.cpu_frequency = float(cfg_dev.get('cpu_frequency_range', [0.5, 0.8])[0])
            self.cpu_frequency = random.uniform(*cfg_dev.get('cpu_frequency_range', [0.5, 0.8]))
        else:
            self.cpu_frequency = cpu_frequency
        self.cpu_frequency = float(self.cpu_frequency)
        if self.cpu_frequency <= 0:
            raise ValueError("UE cpu_frequency must be positive")

        # 能耗参数
        self.alpha_ue = float(cfg_dev.get('alpha_ue', 1e-10))
        # Phase 2: DVFS 能耗系数 kappa, frequency is expressed in Hz.
        energy_cfg = self.config.get('energy', {})
        self.kappa_ue = float(energy_cfg.get('kappa_ue', 1e-28))
        self.transmission_power = float(cfg_dev.get('transmission_power', 0.5))
        if self.kappa_ue < 0 or self.transmission_power < 0:
            raise ValueError("UE energy coefficients must be non-negative")

        # 网络参数（差异化通信延迟）
        self.transmission_rate_to_edge = float(cfg_dev.get('transmission_rate_to_edge', cfg_net.get('ue_to_edge_rate', 1e9)))
        self.transmission_rate_to_cloud = float(cfg_dev.get('transmission_rate_to_cloud', cfg_net.get('ue_to_cloud_total_rate', 100e6)))

        # Phase 2: 到各 ES 的距离 (m)，用于信道模型
        self.distance_to_edges = []

        # 任务执行队列
        self._init_phase2_queue()

    def reset(self):
        """重置设备状态"""
        if self.physics_version >= 2:
            self._reset_phase2_queue()
            return
        self.task_queue.clear()
        self.current_execution = None
        
    def calculate_task_load(self):
        """
        计算任务负载：当前执行任务剩余时间 + 队列中所有任务的处理时间总和
        
        返回: 任务负载（秒）
        """
        if self.physics_version >= 2:
            return max(self.get_available_time() - self.queue_time, 0.0)

        total_load = 0.0
        
        # 当前执行任务的剩余时间
        if self.current_execution and not self.current_execution.completed:
            total_load += self.current_execution.remaining_time
            
        # 队列中等待任务的处理时间
        for task in self.task_queue:
            if not task.completed:
                total_load += task.execution_time
                
        return total_load
    
    def add_task(self, task_id, cpu_cycles, current_time):
        """
        添加新任务到队列
        
        参数:
        - task_id: 任务ID
        - cpu_cycles: CPU周期数
        - current_time: 当前时间
        """
        if self.physics_version >= 2:
            self._phase2_add_task(task_id, cpu_cycles, current_time)
            return

        execution_time = self.calculate_execution_time(cpu_cycles)
        
        if self.current_execution is None or self.current_execution.completed:
            # 没有正在执行的任务，立即开始执行
            self.current_execution = TaskExecution(task_id, cpu_cycles, current_time, execution_time)
        else:
            # 添加到等待队列
            task = TaskExecution(task_id, cpu_cycles, current_time, execution_time)
            self.task_queue.append(task)
    
    def update_tasks(self, time_elapsed):
        """
        更新任务执行状态
        
        参数:
        - time_elapsed: 流逝的时间（秒）
        """
        if self.physics_version >= 2:
            self._phase2_update_tasks(time_elapsed)
            return

        # 更新当前执行任务
        if self.current_execution and not self.current_execution.completed:
            self.current_execution.remaining_time -= time_elapsed
            if self.current_execution.remaining_time <= 0:
                self.current_execution.completed = True
                self.current_execution = None
                
                # 开始执行队列中的下一个任务
                if self.task_queue:
                    self.current_execution = self.task_queue.pop(0)
    
    def calculate_execution_time(self, cpu_cycles):
        """计算执行时间"""
        cpu_frequency_hz = self.cpu_frequency * 1e9  # GHz转Hz
        execution_time = cpu_cycles / cpu_frequency_hz
        return execution_time
    
    def calculate_energy_consumption(self, cpu_cycles):
        """计算计算能耗。

        Phase 1: E = alpha * cycles
        Phase 2: E = kappa * cycles * f^2  (DVFS 模型)
        """
        if self.physics_version >= 2:
            f_hz = self.cpu_frequency * 1e9
            return self.kappa_ue * cpu_cycles * (f_hz ** 2)
        return self.alpha_ue * cpu_cycles
    
    def calculate_transmission_time_to_edge(self, data_size_mb):
        """计算到边缘服务器的传输时间"""
        data_size_bits = data_size_mb * 8 * 1e6  # MB转bits
        transmission_time = data_size_bits / self.transmission_rate_to_edge
        return transmission_time
    
    def calculate_transmission_time_to_cloud(self, data_size_mb):
        """
        计算到云服务器的传输时间（经边缘中转）
        包含UE→Edge→Cloud的总传输时间
        """
        data_size_bits = data_size_mb * 8 * 1e6  # MB转bits
        
        # UE到边缘的传输时间
        time_ue_to_edge = data_size_bits / self.transmission_rate_to_edge
        
        # 边缘到云的传输时间（假设边缘到云带宽为1Gbps）
        edge_to_cloud_rate = 1e9  # 1 Gbps
        time_edge_to_cloud = data_size_bits / edge_to_cloud_rate
        
        # 总传输时间
        total_transmission_time = time_ue_to_edge + time_edge_to_cloud
        return total_transmission_time
    
    def calculate_transmission_energy(self, transmission_time):
        """计算传输能耗"""
        energy = self.transmission_power * transmission_time
        return energy
    
    def get_state(self):
        """
        返回简化的设备状态
        返回: [CPU频率(归一化), 任务负载(归一化)]
        """
        # 任务负载归一化（假设最大60秒的任务负载）
        task_load = self.calculate_task_load()
        task_load_norm = min(task_load / 60.0, 1.0)
        
        return [
            self.cpu_frequency / 1.0,      # CPU频率归一化（最大1.0GHz）
            task_load_norm                  # 任务负载归一化
        ]


class EdgeServer(_Phase2QueueMixin):
    """
    边缘服务器（Edge Server, ES）
    简化状态：[CPU频率, 任务负载]
    """

    def __init__(self, server_id, cpu_frequency, config=None):
        self.server_id = server_id
        self.cpu_frequency = float(cpu_frequency)  # GHz
        if self.cpu_frequency <= 0:
            raise ValueError("ES cpu_frequency must be positive")
        self.config = config or {}
        cfg_env = self.config.get('environment', {})
        cfg_net = self.config.get('network', {})
        cfg_edge = self.config.get('device_specs', {}).get('edge_servers', {})
        self.model_version = self.config.get('model_version', 1)
        self.physics_version = int(cfg_env.get('physics_version', 1))

        # 能耗参数
        self.alpha_es = float(cfg_edge.get('alpha_es', 3e-10))
        energy_cfg = self.config.get('energy', {})
        self.kappa_es = float(energy_cfg.get('kappa_es', 3e-28))
        if self.kappa_es < 0:
            raise ValueError("ES kappa_es must be non-negative")

        # 网络参数
        self.transmission_rate_to_cloud = float(cfg_edge.get('transmission_rate_to_cloud', cfg_net.get('edge_to_cloud_rate', 1e9)))

        # 任务执行队列
        self._init_phase2_queue()
        
    def reset(self):
        """重置服务器状态"""
        if self.physics_version >= 2:
            self._reset_phase2_queue()
            return
        self.task_queue.clear()
        self.current_execution = None
        
    def calculate_task_load(self):
        """
        计算任务负载：当前执行任务剩余时间 + 队列中所有任务的处理时间总和
        """
        if self.physics_version >= 2:
            return max(self.get_available_time() - self.queue_time, 0.0)

        total_load = 0.0
        
        # 当前执行任务的剩余时间
        if self.current_execution and not self.current_execution.completed:
            total_load += self.current_execution.remaining_time
            
        # 队列中等待任务的处理时间
        for task in self.task_queue:
            if not task.completed:
                total_load += task.execution_time
                
        return total_load
    
    def add_task(self, task_id, cpu_cycles, current_time):
        """添加新任务到队列"""
        if self.physics_version >= 2:
            self._phase2_add_task(task_id, cpu_cycles, current_time)
            return

        execution_time = self.calculate_execution_time(cpu_cycles)
        
        if self.current_execution is None or self.current_execution.completed:
            self.current_execution = TaskExecution(task_id, cpu_cycles, current_time, execution_time)
        else:
            task = TaskExecution(task_id, cpu_cycles, current_time, execution_time)
            self.task_queue.append(task)
    
    def update_tasks(self, time_elapsed):
        """更新任务执行状态"""
        if self.physics_version >= 2:
            self._phase2_update_tasks(time_elapsed)
            return

        if self.current_execution and not self.current_execution.completed:
            self.current_execution.remaining_time -= time_elapsed
            if self.current_execution.remaining_time <= 0:
                self.current_execution.completed = True
                self.current_execution = None
                
                if self.task_queue:
                    self.current_execution = self.task_queue.pop(0)
    
    def calculate_execution_time(self, cpu_cycles):
        """计算执行时间"""
        cpu_frequency_hz = self.cpu_frequency * 1e9  # GHz转Hz
        execution_time = cpu_cycles / cpu_frequency_hz
        return execution_time
    
    def calculate_energy_consumption(self, cpu_cycles):
        """计算能耗。

        Phase 1: E = alpha * cycles
        Phase 2: E = kappa * cycles * f^2  (DVFS 模型)
        """
        if self.physics_version >= 2:
            f_hz = self.cpu_frequency * 1e9
            return self.kappa_es * cpu_cycles * (f_hz ** 2)
        return self.alpha_es * cpu_cycles

    def calculate_transmission_time_to_cloud(self, data_size_mb):
        """计算到云服务器的传输时间"""
        data_size_bits = data_size_mb * 8 * 1e6  # MB转bits
        transmission_time = data_size_bits / self.transmission_rate_to_cloud
        return transmission_time
    
    def get_expected_completion_time(self, cpu_cycles):
        """
        获取新任务的预期完成时间
        包括等待时间和执行时间
        """
        execution_time = self.calculate_execution_time(cpu_cycles)
        current_load = self.calculate_task_load()
        return current_load + execution_time
    
    def get_state(self):
        """
        返回简化的服务器状态
        返回: [CPU频率(归一化), 任务负载(归一化)]
        """
        # 任务负载归一化（假设最大120秒的任务负载）
        task_load = self.calculate_task_load()
        task_load_norm = min(task_load / 120.0, 1.0)
        
        return [
            self.cpu_frequency / 12.0,  # CPU频率归一化（最大12GHz）
            task_load_norm              # 任务负载归一化
        ]


class CloudServer(_Phase2QueueMixin):
    """
    云服务器（Cloud Server, CS）

    Phase 1: 资源无限，无任务负载
    Phase 2: 有限资源，有任务队列
    """

    def __init__(self, server_id=0, config=None):
        self.server_id = server_id
        self.config = config or {}
        cfg_cloud = self.config.get('device_specs', {}).get('cloud_servers', {})
        self.cpu_frequency = float(cfg_cloud.get('cpu_frequency', 20.0))
        if self.cpu_frequency <= 0:
            raise ValueError("CS cpu_frequency must be positive")
        self.model_version = self.config.get('model_version', 1)
        self.physics_version = int(
            self.config.get('environment', {}).get('physics_version', 1)
        )

        # 能耗参数
        self.alpha_cs = float(cfg_cloud.get('alpha_cs', 3e-10))
        energy_cfg = self.config.get('energy', {})
        self.kappa_cs = float(energy_cfg.get('kappa_cs', 3e-28))

        # 并行处理能力
        self.parallel_factor = float(cfg_cloud.get('parallel_factor', 8.0))
        if self.kappa_cs < 0 or self.parallel_factor <= 0:
            raise ValueError(
                "CS kappa_cs must be non-negative and parallel_factor positive"
            )

        # Phase 2: 任务队列
        self._init_phase2_queue()

    def reset(self):
        """重置服务器状态"""
        if self.physics_version >= 2:
            self._reset_phase2_queue()

    def calculate_task_load(self):
        """计算任务负载 (Phase 2)。"""
        if self.physics_version < 2:
            return 0.0
        return max(self.get_available_time() - self.queue_time, 0.0)

    def add_task(self, task_id, cpu_cycles, current_time):
        """添加任务到队列 (Phase 2)。"""
        if self.physics_version < 2:
            return
        self._phase2_add_task(task_id, cpu_cycles, current_time)

    def update_tasks(self, time_elapsed):
        """更新任务执行状态 (Phase 2)。"""
        if self.physics_version < 2:
            return
        self._phase2_update_tasks(time_elapsed)

    def calculate_execution_time(self, cpu_cycles):
        """
        计算云服务器执行时间。
        云服务器具有强大的并行处理能力。
        """
        cpu_frequency_hz = self.cpu_frequency * 1e9  # GHz转Hz
        effective_frequency = cpu_frequency_hz * self.parallel_factor
        execution_time = cpu_cycles / effective_frequency
        return execution_time

    def calculate_energy_consumption(self, cpu_cycles):
        """计算云服务器能耗。

        Phase 1: E = alpha * cycles
        Phase 2: E = kappa * cycles * f^2
        """
        if self.physics_version >= 2:
            f_hz = self.cpu_frequency * 1e9
            return self.kappa_cs * cpu_cycles * (f_hz ** 2)
        return self.alpha_cs * cpu_cycles

    def get_expected_completion_time(self, cpu_cycles):
        """获取新任务的预期完成时间。"""
        exec_time = self.calculate_execution_time(cpu_cycles)
        if self.physics_version >= 2:
            return self.calculate_task_load() + exec_time
        return exec_time

    def get_state(self):
        """返回简化的服务器状态。"""
        return [
            self.cpu_frequency / 20.0  # CPU频率归一化（最大20GHz）
        ]


# 兼容性保持
Device = UserEquipment  # 别名，保持向后兼容
