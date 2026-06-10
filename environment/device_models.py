# environment/device_models.py
"""
简化的云边端设备模型
- UE: CPU频率 + 任务负载
- ES: CPU频率 + 任务负载  
- CS: CPU频率（资源无限）
- 考虑差异化的通信延迟
"""
import numpy as np
import random


class TaskExecution:
    """任务执行记录"""
    def __init__(self, task_id, task_workload, start_time, execution_time):
        self.task_id = task_id
        self.task_workload = task_workload
        self.start_time = start_time
        self.execution_time = execution_time
        self.remaining_time = execution_time
        self.completed = False


class UserEquipment:
    """
    端侧设备（User Equipment, UE）
    简化状态：[CPU频率, 任务负载]
    """
    
    def __init__(self, device_id, cpu_frequency=None, config=None):
        self.device_id = device_id
        cfg_env = (config or {}).get('environment', {})
        cfg_net = (config or {}).get('network', {})
        cfg_dev = (config or {}).get('device_specs', {}).get('user_equipment', {})
        # CPU频率：0.5-1.0 GHz异构配置
        if cpu_frequency is None:
            self.cpu_frequency = float(cfg_dev.get('cpu_frequency_range', [0.5, 0.8])[0])
            self.cpu_frequency = random.uniform(*cfg_dev.get('cpu_frequency_range', [0.5, 0.8]))
        else:
            self.cpu_frequency = cpu_frequency
            
        # # 计算限制：单任务最大可处理 CPU 周期数 500 Mcycles
        # self.max_cpu_cycles = 500e6  # 500 * 10^6 cycles
        
        # 能耗参数
        self.alpha_ue = float(cfg_dev.get('alpha_ue', 1e-10))
        self.transmission_power = float(cfg_dev.get('transmission_power', 0.5))
        
        # 网络参数（差异化通信延迟）
        self.transmission_rate_to_edge = float(cfg_dev.get('transmission_rate_to_edge', cfg_net.get('ue_to_edge_rate', 1e9)))
        self.transmission_rate_to_cloud = float(cfg_dev.get('transmission_rate_to_cloud', cfg_net.get('ue_to_cloud_total_rate', 100e6)))
        

        
        # 任务执行队列
        self.task_queue = []  # 当前执行和等待的任务
        self.current_execution = None  # 当前正在执行的任务
        
    def reset(self):
        """重置设备状态"""
        self.task_queue.clear()
        self.current_execution = None
        
    def calculate_task_load(self):
        """
        计算任务负载：当前执行任务剩余时间 + 队列中所有任务的处理时间总和
        
        返回: 任务负载（秒）
        """
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
        """计算计算能耗"""
        energy = self.alpha_ue * cpu_cycles
        return energy
    
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


class EdgeServer:
    """
    边缘服务器（Edge Server, ES）
    简化状态：[CPU频率, 任务负载]
    """
    
    def __init__(self, server_id, cpu_frequency, config=None):
        self.server_id = server_id
        self.cpu_frequency = cpu_frequency  # GHz
        cfg_env = (config or {}).get('environment', {})
        cfg_net = (config or {}).get('network', {})
        cfg_edge = (config or {}).get('device_specs', {}).get('edge_servers', {})
        # 能耗参数
        self.alpha_es = float(cfg_edge.get('alpha_es', 3e-10))
        
        # 网络参数
        self.transmission_rate_to_cloud = float(cfg_edge.get('transmission_rate_to_cloud', cfg_net.get('edge_to_cloud_rate', 1e9)))
        
        # 任务执行队列
        self.task_queue = []
        self.current_execution = None
        
    def reset(self):
        """重置服务器状态"""
        self.task_queue.clear()
        self.current_execution = None
        
    def calculate_task_load(self):
        """
        计算任务负载：当前执行任务剩余时间 + 队列中所有任务的处理时间总和
        """
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
        execution_time = self.calculate_execution_time(cpu_cycles)
        
        if self.current_execution is None or self.current_execution.completed:
            self.current_execution = TaskExecution(task_id, cpu_cycles, current_time, execution_time)
        else:
            task = TaskExecution(task_id, cpu_cycles, current_time, execution_time)
            self.task_queue.append(task)
    
    def update_tasks(self, time_elapsed):
        """更新任务执行状态"""
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
        """计算能耗"""
        energy = self.alpha_es * cpu_cycles
        return energy
    
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


class CloudServer:
    """
    云服务器（Cloud Server, CS）
    简化状态：[CPU频率]（资源无限，无任务负载）
    """
    
    def __init__(self, server_id=0, config=None):
        self.server_id = server_id
        cfg_cloud = (config or {}).get('device_specs', {}).get('cloud_servers', {})
        self.cpu_frequency = float(cfg_cloud.get('cpu_frequency', 20.0))
        
        # 能耗参数（极低）
        self.alpha_cs = float(cfg_cloud.get('alpha_cs', 3e-10))
        
        # 并行处理能力
        self.parallel_factor = float(cfg_cloud.get('parallel_factor', 8.0))
        
    def reset(self):
        """重置服务器状态（云服务器无需重置）"""
        pass
        
    def calculate_execution_time(self, cpu_cycles):
        """
        计算云服务器执行时间
        云服务器具有强大的并行处理能力，任务可以立即执行
        """
        cpu_frequency_hz = self.cpu_frequency * 1e9  # GHz转Hz
        # 考虑并行处理能力
        effective_frequency = cpu_frequency_hz * self.parallel_factor
        execution_time = cpu_cycles / effective_frequency
        return execution_time
    
    def calculate_energy_consumption(self, cpu_cycles):
        """计算云服务器能耗（极低）"""
        energy = self.alpha_cs * cpu_cycles
        return energy
    
    def get_expected_completion_time(self, cpu_cycles):
        """
        获取新任务的预期完成时间
        云服务器资源无限，无等待时间
        """
        return self.calculate_execution_time(cpu_cycles)
    
    def get_state(self):
        """
        返回简化的服务器状态
        返回: [CPU频率(归一化)]
        """
        return [
            self.cpu_frequency / 20.0  # CPU频率归一化（最大20GHz）
        ]


# 兼容性保持
Device = UserEquipment  # 别名，保持向后兼容