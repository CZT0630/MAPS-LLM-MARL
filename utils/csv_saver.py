import pandas as pd
import os
from datetime import datetime
import numpy as np


class TrainingMetricsCSVSaver:
    """
    专门用于保存训练过程中核心指标到CSV文件的工具类
    """
    
    def __init__(self, save_dir="results/csv_data"):
        """
        初始化CSV保存器
        
        Args:
            save_dir: CSV文件保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def save_training_metrics_to_csv(self, episode_rewards, episode_latencies, 
                                   episode_energies, episode_completion_rates, 
                                   algorithm_name="LLM_MADDPG", timestamp=None):
        """
        保存训练过程中的核心指标到CSV表格
        
        Args:
            episode_rewards: 每轮奖励值列表
            episode_latencies: 每轮平均时延列表  
            episode_energies: 每轮能耗消耗列表
            episode_completion_rates: 每轮任务完成率列表
            algorithm_name: 算法名称
            timestamp: 时间戳，如果为None则自动生成
        
        Returns:
            str: 保存的CSV文件路径
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
        # 确保所有列表长度一致
        max_len = max(len(episode_rewards), len(episode_latencies), 
                     len(episode_energies), len(episode_completion_rates))
        
        # 补齐长度不足的列表
        def pad_list(lst, target_len):
            if len(lst) < target_len:
                # 用最后一个值填充或用0填充
                last_val = lst[-1] if lst else 0
                return lst + [last_val] * (target_len - len(lst))
            return lst[:target_len]
        
        episode_rewards = pad_list(episode_rewards, max_len)
        episode_latencies = pad_list(episode_latencies, max_len)
        episode_energies = pad_list(episode_energies, max_len)
        episode_completion_rates = pad_list(episode_completion_rates, max_len)
        
        # 创建DataFrame
        data = {
            'Episode': list(range(1, max_len + 1)),
            'Reward': episode_rewards,
            'Latency_s': episode_latencies,
            'Energy_J': episode_energies,
            'Completion_Rate': episode_completion_rates
        }
        
        # 添加一些统计指标列
        if len(episode_rewards) > 0:
            # 计算移动平均 (窗口大小为10)
            window_size = min(10, max_len)
            
            reward_ma = self._calculate_moving_average(episode_rewards, window_size)
            latency_ma = self._calculate_moving_average(episode_latencies, window_size)
            energy_ma = self._calculate_moving_average(episode_energies, window_size)
            completion_ma = self._calculate_moving_average(episode_completion_rates, window_size)
            
            data.update({
                'Reward_MA10': reward_ma,
                'Latency_MA10': latency_ma, 
                'Energy_MA10': energy_ma,
                'Completion_Rate_MA10': completion_ma
            })
        
        df = pd.DataFrame(data)
        
        # 保存到CSV
        filename = f"{algorithm_name}_training_metrics_{timestamp}.csv"
        filepath = os.path.join(self.save_dir, filename)
        
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        # 打印保存信息
        print(f"\n📊 训练指标已保存到CSV文件:")
        print(f"  文件路径: {filepath}")
        print(f"  数据行数: {len(df)}")
        print(f"  包含列: {list(df.columns)}")
        
        # 显示基本统计信息
        print(f"\n📈 训练指标统计摘要:")
        print(f"  平均奖励: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
        print(f"  平均时延: {np.mean(episode_latencies):.3f}s ± {np.std(episode_latencies):.3f}s")
        print(f"  平均能耗: {np.mean(episode_energies):.3f}J ± {np.std(episode_energies):.3f}J")
        print(f"  平均完成率: {np.mean(episode_completion_rates):.3f} ± {np.std(episode_completion_rates):.3f}")
        
        return filepath
    
    def save_comparison_metrics_to_csv(self, algorithms_results, timestamp=None):
        """
        保存多算法对比结果到CSV
        
        Args:
            algorithms_results: 字典格式 {algorithm_name: {metrics}}
            timestamp: 时间戳
            
        Returns:
            str: 保存的CSV文件路径
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
        comparison_data = []
        
        for algo_name, results in algorithms_results.items():
            if 'episode_rewards' in results:
                rewards = results['episode_rewards']
                latencies = results.get('episode_latencies', [])
                energies = results.get('episode_energies', [])
                completion_rates = results.get('episode_completion_rates', [])
                
                # 计算统计指标
                row = {
                    'Algorithm': algo_name,
                    'Total_Episodes': len(rewards),
                    'Avg_Reward': np.mean(rewards) if rewards else 0,
                    'Std_Reward': np.std(rewards) if rewards else 0,
                    'Best_Reward': np.max(rewards) if rewards else 0,
                    'Worst_Reward': np.min(rewards) if rewards else 0,
                    'Final_50_Avg_Reward': np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards),
                    'Avg_Latency': np.mean(latencies) if latencies else 0,
                    'Std_Latency': np.std(latencies) if latencies else 0,
                    'Avg_Energy': np.mean(energies) if energies else 0,
                    'Std_Energy': np.std(energies) if energies else 0,
                    'Avg_Completion_Rate': np.mean(completion_rates) if completion_rates else 0,
                    'Std_Completion_Rate': np.std(completion_rates) if completion_rates else 0
                }
                comparison_data.append(row)
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            filename = f"algorithms_comparison_{timestamp}.csv"
            filepath = os.path.join(self.save_dir, filename)
            
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            print(f"\n📊 算法对比结果已保存到CSV文件:")
            print(f"  文件路径: {filepath}")
            print(f"  对比算法数: {len(df)}")
            
            return filepath
        
        return None
    
    def _calculate_moving_average(self, data, window_size):
        """计算移动平均"""
        if len(data) == 0:
            return []
            
        moving_avg = []
        for i in range(len(data)):
            start_idx = max(0, i - window_size + 1)
            window_data = data[start_idx:i+1]
            moving_avg.append(np.mean(window_data))
        
        return moving_avg
    
    def load_training_metrics_from_csv(self, filepath):
        """
        从CSV文件加载训练指标
        
        Args:
            filepath: CSV文件路径
            
        Returns:
            dict: 包含训练指标的字典
        """
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig')
            
            result = {
                'episode_rewards': df['Reward'].tolist(),
                'episode_latencies': df['Latency_s'].tolist(),
                'episode_energies': df['Energy_J'].tolist(),
                'episode_completion_rates': df['Completion_Rate'].tolist()
            }
            
            print(f"✅ 成功从CSV加载训练指标: {filepath}")
            print(f"  Episode数: {len(result['episode_rewards'])}")
            
            return result
            
        except Exception as e:
            print(f"❌ 加载CSV文件失败: {e}")
            return None
    
    def get_latest_csv_files(self):
        """获取最新的CSV文件列表"""
        if not os.path.exists(self.save_dir):
            return []
            
        csv_files = [f for f in os.listdir(self.save_dir) if f.endswith('.csv')]
        csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.save_dir, x)), reverse=True)
        
        return [os.path.join(self.save_dir, f) for f in csv_files]


# 便捷函数
def save_training_metrics_csv(episode_rewards, episode_latencies, episode_energies, 
                            episode_completion_rates, algorithm_name="LLM_MADDPG", 
                            save_dir=None):
    """
    便捷函数：保存训练指标到CSV
    """
    if save_dir is None:
        # 如果没有指定保存目录，使用全局路径管理器
        from utils.path_manager import get_path_manager
        path_manager = get_path_manager()
        save_dir = path_manager.get_data_path("csv")
    
    saver = TrainingMetricsCSVSaver(save_dir)
    return saver.save_training_metrics_to_csv(
        episode_rewards, episode_latencies, episode_energies, 
        episode_completion_rates, algorithm_name
    )


def save_test_results_csv(test_results, algorithm_name="Test", save_dir="results/test_results"):
    """
    便捷函数：保存测试结果到CSV
    
    Args:
        test_results: 包含测试结果的字典
        algorithm_name: 算法名称
        save_dir: 保存目录
        
    Returns:
        str: 保存的CSV文件路径
    """
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建测试结果摘要数据
    summary_data = {
        'Algorithm': [algorithm_name],
        'Test_Episodes': [test_results.get('num_episodes', 0)],
        'Avg_Energy': [test_results.get('avg_energy', 0)],
        'Avg_Utilization': [test_results.get('avg_utilization', 0)],
        'Avg_Delay': [test_results.get('avg_delay', 0)],
        'Energy_Std': [test_results.get('energy_std', 0)],
        'Utilization_Std': [test_results.get('utilization_std', 0)],
        'Delay_Std': [test_results.get('delay_std', 0)]
    }
    
    # 保存摘要结果
    summary_df = pd.DataFrame(summary_data)
    summary_filename = f"{algorithm_name}_test_summary_{timestamp}.csv"
    summary_filepath = os.path.join(save_dir, summary_filename)
    summary_df.to_csv(summary_filepath, index=False, encoding='utf-8-sig')
    
    # 如果有详细的每轮数据，也保存
    if 'all_episode_energy' in test_results:
        episode_data = {
            'Episode': list(range(1, len(test_results['all_episode_energy']) + 1)),
            'Energy': test_results['all_episode_energy'],
            'Utilization': test_results.get('all_episode_utilization', []),
            'Delay': test_results.get('all_episode_delay', [])
        }
        
        detail_df = pd.DataFrame(episode_data)
        detail_filename = f"{algorithm_name}_test_details_{timestamp}.csv"
        detail_filepath = os.path.join(save_dir, detail_filename)
        detail_df.to_csv(detail_filepath, index=False, encoding='utf-8-sig')
        
        print(f"✅ 测试详细结果保存至: {detail_filepath}")
    
    print(f"✅ 测试结果摘要保存至: {summary_filepath}")
    return summary_filepath


if __name__ == "__main__":
    # 测试示例
    saver = TrainingMetricsCSVSaver()
    
    # 模拟训练数据
    episode_rewards = [i * 0.1 + np.random.normal(0, 0.05) for i in range(100)]
    episode_latencies = [2.0 + np.random.normal(0, 0.2) for _ in range(100)]
    episode_energies = [1.5 + np.random.normal(0, 0.1) for _ in range(100)]
    episode_completion_rates = [0.8 + np.random.normal(0, 0.05) for _ in range(100)]
    
    # 保存到CSV
    filepath = saver.save_training_metrics_to_csv(
        episode_rewards, episode_latencies, episode_energies, 
        episode_completion_rates, "TEST_ALGORITHM"
    )
    
    print(f"测试完成，文件保存至: {filepath}")