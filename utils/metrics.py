# utils/metrics.py
class MetricsTracker:
    """跟踪并记录训练过程中的指标"""
    def __init__(self):
        self.episode_rewards = []
        self.episode_delays = []
        self.episode_energy = []
        self.llm_used = []
        self.distillation_losses = []  # 蒸馏损失
        self.policy_similarity = []    # LLM与MADDPG行为相似度
        self.critic_losses = []        # Critic损失
        self.actor_losses = []         # Actor损失

    def add_episode(self, reward, delay, energy, llm_used=False):
        """添加单个episode的指标"""
        self.episode_rewards.append(reward)
        self.episode_delays.append(delay)
        self.episode_energy.append(energy)
        self.llm_used.append(1 if llm_used else 0)

    def add_training_metrics(self, critic_loss, actor_loss, distillation_loss, similarity=None):
        """添加训练过程的指标"""
        self.critic_losses.append(critic_loss)
        self.actor_losses.append(actor_loss)
        self.distillation_losses.append(distillation_loss)
        if similarity is not None:
            self.policy_similarity.append(similarity)

    def get_average_metrics(self, last_n=None):
        """获取平均指标
        
        Args:
            last_n: 只计算最后n个episode的平均值，如果为None则计算所有
        """
        if last_n is not None:
            rewards = self.episode_rewards[-last_n:]
            delays = self.episode_delays[-last_n:]
            energy = self.episode_energy[-last_n:]
            llm_used = self.llm_used[-last_n:]
            distillation = self.distillation_losses[-last_n:] if self.distillation_losses else []
            similarity = self.policy_similarity[-last_n:] if self.policy_similarity else []
            critic = self.critic_losses[-last_n:] if self.critic_losses else []
            actor = self.actor_losses[-last_n:] if self.actor_losses else []
        else:
            rewards = self.episode_rewards
            delays = self.episode_delays
            energy = self.episode_energy
            llm_used = self.llm_used
            distillation = self.distillation_losses
            similarity = self.policy_similarity
            critic = self.critic_losses
            actor = self.actor_losses

        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        avg_delay = sum(delays) / len(delays) if delays else 0
        avg_energy = sum(energy) / len(energy) if energy else 0
        llm_usage_ratio = sum(llm_used) / len(llm_used) if llm_used else 0
        avg_distillation = sum(distillation) / len(distillation) if distillation else 0
        avg_similarity = sum(similarity) / len(similarity) if similarity else 0
        avg_critic = sum(critic) / len(critic) if critic else 0
        avg_actor = sum(actor) / len(actor) if actor else 0

        return {
            'avg_reward': avg_reward,
            'avg_delay': avg_delay,
            'avg_energy': avg_energy,
            'llm_usage_ratio': llm_usage_ratio,
            'avg_distillation_loss': avg_distillation,
            'avg_policy_similarity': avg_similarity,
            'avg_critic_loss': avg_critic,
            'avg_actor_loss': avg_actor
        }


def calculate_episode_metrics(rewards, info):
    """计算单个episode的指标
    
    Args:
        rewards: 奖励列表
        info: 环境返回的信息字典，包含delays, energies, utilizations等
    
    Returns:
        dict: 包含计算后的指标
    """
    total_reward = sum(rewards) if rewards else 0
    avg_reward = total_reward / len(rewards) if rewards else 0
    
    # 从info中提取指标
    delays = info.get('delays', [])
    energies = info.get('energies', [])
    utilizations = info.get('utilizations', [])
    
    avg_delay = sum(delays) / len(delays) if delays else 0
    total_energy = sum(energies) if energies else 0
    avg_energy = total_energy / len(energies) if energies else 0
    avg_utilization = sum(utilizations) / len(utilizations) if utilizations else 0
    
    # 从任务完成率统计中提取指标
    completion_stats = info.get('task_completion_stats', {})
    completion_rate = completion_stats.get('on_time_completion_rate', 1.0)
    overall_completion_rate = completion_stats.get('overall_completion_rate', 1.0)
    timeout_rate = completion_stats.get('timeout_rate', 0.0)
    failure_rate = completion_stats.get('failure_rate', 0.0)
    avg_overtime = completion_stats.get('avg_overtime', 0.0)
    
    return {
        'total_reward': total_reward,
        'avg_reward': avg_reward,
        'total_energy': total_energy,
        'avg_energy': avg_energy,
        'avg_delay': avg_delay,
        'avg_utilization': avg_utilization,
        'on_time_completion_rate': completion_rate,              # 按时完成率
        'overall_completion_rate': overall_completion_rate,      # 总完成率
        'timeout_rate': timeout_rate,                            # 超时率
        'failure_rate': failure_rate,                            # 失败率
        'avg_overtime': avg_overtime,                            # 平均超时时间
        'num_steps': len(rewards)
    }

def calculate_completion_rate_statistics(completion_stats_list):
    """
    计算任务完成率的统计信息
    
    Args:
        completion_stats_list: 每个episode的任务完成率统计列表
    
    Returns:
        dict: 完成率统计摘要
    """
    if not completion_stats_list:
        return {}
    
    # 提取各项指标
    on_time_rates = [stats.get('on_time_completion_rate', 0) for stats in completion_stats_list]
    overall_rates = [stats.get('overall_completion_rate', 0) for stats in completion_stats_list]
    timeout_rates = [stats.get('timeout_rate', 0) for stats in completion_stats_list]
    failure_rates = [stats.get('failure_rate', 0) for stats in completion_stats_list]
    overtimes = [stats.get('avg_overtime', 0) for stats in completion_stats_list]
    
    # 计算统计指标
    import numpy as np
    
    return {
        'avg_on_time_completion_rate': np.mean(on_time_rates),
        'std_on_time_completion_rate': np.std(on_time_rates),
        'min_on_time_completion_rate': np.min(on_time_rates),
        'max_on_time_completion_rate': np.max(on_time_rates),
        
        'avg_overall_completion_rate': np.mean(overall_rates),
        'std_overall_completion_rate': np.std(overall_rates),
        
        'avg_timeout_rate': np.mean(timeout_rates),
        'std_timeout_rate': np.std(timeout_rates),
        
        'avg_failure_rate': np.mean(failure_rates),
        'std_failure_rate': np.std(failure_rates),
        
        'avg_overtime': np.mean([t for t in overtimes if t > 0]),  # 只计算有超时的情况
        'std_overtime': np.std([t for t in overtimes if t > 0]) if any(t > 0 for t in overtimes) else 0,
        
        'total_episodes': len(completion_stats_list)
    }

def analyze_deadline_violations(violation_records):
    """
    分析截止时间违反情况
    
    Args:
        violation_records: 截止时间违反记录列表
    
    Returns:
        dict: 违反情况分析
    """
    if not violation_records:
        return {
            'total_violations': 0,
            'violation_analysis': {}
        }
    
    import numpy as np
    from collections import defaultdict
    
    # 按任务类型分组分析
    violations_by_type = defaultdict(list)
    for record in violation_records:
        task_type = record.get('task_type', 'unknown')
        overtime = record.get('overtime', 0)
        violations_by_type[task_type].append(overtime)
    
    # 分析结果
    analysis = {}
    for task_type, overtimes in violations_by_type.items():
        analysis[task_type] = {
            'count': len(overtimes),
            'avg_overtime': np.mean(overtimes),
            'max_overtime': np.max(overtimes),
            'min_overtime': np.min(overtimes),
            'std_overtime': np.std(overtimes)
        }
    
    # 整体统计
    all_overtimes = [record.get('overtime', 0) for record in violation_records]
    
    return {
        'total_violations': len(violation_records),
        'avg_overtime': np.mean(all_overtimes),
        'max_overtime': np.max(all_overtimes),
        'min_overtime': np.min(all_overtimes),
        'std_overtime': np.std(all_overtimes),
        'violation_by_type': analysis
        }