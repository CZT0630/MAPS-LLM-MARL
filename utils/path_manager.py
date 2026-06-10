"""
路径管理器 - 统一管理项目中所有文件保存路径
实现标准化的目录结构和文件命名规范
"""

import os
import time
from datetime import datetime
from typing import Optional


class PathManager:
    """统一路径管理器"""
    
    def __init__(self, base_dir: str = "results", experiment_timestamp: Optional[str] = None):
        """
        初始化路径管理器
        
        Args:
            base_dir: 基础目录，默认为 "results"
            experiment_timestamp: 实验时间戳，如果为None则自动生成
        """
        self.base_dir = base_dir
        if experiment_timestamp is None:
            experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_timestamp = experiment_timestamp
        self.experiment_dir = os.path.join(base_dir, f"experiment_{experiment_timestamp}")
        
        # 创建基础目录结构
        self._create_directories()
    
    def _create_directories(self):
        """创建标准目录结构"""
        dirs = [
            self.experiment_dir,
            # 算法结果目录
            os.path.join(self.experiment_dir, "maddpg", "models"),
            os.path.join(self.experiment_dir, "llm_maddpg", "models"),
            os.path.join(self.experiment_dir, "llm", "models"),
            os.path.join(self.experiment_dir, "mappo", "models"),
            os.path.join(self.experiment_dir, "happo", "models"),
            # 数据目录
            os.path.join(self.experiment_dir, "data", "csv"),
            os.path.join(self.experiment_dir, "data", "json"),
            os.path.join(self.experiment_dir, "data", "stats"),
            # 输出目录
            os.path.join(self.experiment_dir, "plots"),
            os.path.join(self.experiment_dir, "logs"),
            os.path.join(self.experiment_dir, "comparison"),
            os.path.join(self.experiment_dir, "test_results")
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_experiment_dir(self) -> str:
        """获取实验根目录"""
        return self.experiment_dir
    
    def get_model_path(self, algorithm: str) -> str:
        """
        获取模型保存目录
        
        Args:
            algorithm: 算法名称 ("maddpg", "llm_maddpg", "llm")
        """
        return os.path.join(self.experiment_dir, algorithm, "models")
    
    def get_model_file_path(self, algorithm: str, filename: str) -> str:
        """
        获取模型文件完整路径
        
        Args:
            algorithm: 算法名称
            filename: 文件名
        """
        return os.path.join(self.get_model_path(algorithm), filename)
    
    def get_data_path(self, data_type: str) -> str:
        """
        获取数据保存目录
        
        Args:
            data_type: 数据类型 ("csv", "json", "stats")
        """
        return os.path.join(self.experiment_dir, "data", data_type)
    
    def get_data_file_path(self, data_type: str, filename: str) -> str:
        """
        获取数据文件完整路径
        
        Args:
            data_type: 数据类型
            filename: 文件名
        """
        return os.path.join(self.get_data_path(data_type), filename)
    
    def get_plot_path(self) -> str:
        """获取图表保存目录"""
        return os.path.join(self.experiment_dir, "plots")
    
    def get_plot_file_path(self, filename: str) -> str:
        """获取图表文件完整路径"""
        return os.path.join(self.get_plot_path(), filename)
    
    def get_log_path(self) -> str:
        """获取日志保存目录"""
        return os.path.join(self.experiment_dir, "logs")
    
    def get_log_file_path(self, filename: str) -> str:
        """获取日志文件完整路径"""
        return os.path.join(self.get_log_path(), filename)
    
    def get_comparison_path(self) -> str:
        """获取对比结果保存目录"""
        return os.path.join(self.experiment_dir, "comparison")
    
    def get_comparison_file_path(self, filename: str) -> str:
        """获取对比结果文件完整路径"""
        return os.path.join(self.get_comparison_path(), filename)
    
    def get_test_results_path(self) -> str:
        """获取测试结果保存目录"""
        return os.path.join(self.experiment_dir, "test_results")
    
    def get_test_results_file_path(self, filename: str) -> str:
        """获取测试结果文件完整路径"""
        return os.path.join(self.get_test_results_path(), filename)
    
    def get_algorithm_result_path(self, algorithm: str) -> str:
        """获取算法结果目录（包含模型和结果文件）"""
        return os.path.join(self.experiment_dir, algorithm)
    
    def get_algorithm_result_file_path(self, algorithm: str, filename: str) -> str:
        """获取算法结果文件完整路径"""
        return os.path.join(self.get_algorithm_result_path(algorithm), filename)
    
    def generate_timestamped_filename(self, base_name: str, extension: str, include_experiment_timestamp: bool = False) -> str:
        """
        生成带时间戳的文件名
        
        Args:
            base_name: 基础文件名
            extension: 文件扩展名（包含点号）
            include_experiment_timestamp: 是否包含实验时间戳
        """
        if include_experiment_timestamp:
            timestamp = self.experiment_timestamp
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{base_name}_{timestamp}{extension}"
    
    def get_directory_info(self) -> dict:
        """获取目录结构信息"""
        return {
            "experiment_dir": self.experiment_dir,
            "experiment_timestamp": self.experiment_timestamp,
            "maddpg_models": self.get_model_path("maddpg"),
            "llm_maddpg_models": self.get_model_path("llm_maddpg"),
            "llm_models": self.get_model_path("llm"),
            "mappo_models": self.get_model_path("mappo"),
            "happo_models": self.get_model_path("happo"),
            "csv_data": self.get_data_path("csv"),
            "json_data": self.get_data_path("json"),
            "stats_data": self.get_data_path("stats"),
            "plots": self.get_plot_path(),
            "logs": self.get_log_path(),
            "comparison": self.get_comparison_path(),
            "test_results": self.get_test_results_path()
        }


# 全局路径管理器实例
_global_path_manager: Optional[PathManager] = None


def get_path_manager() -> PathManager:
    """
    获取全局路径管理器实例
    如果不存在则创建新实例
    """
    global _global_path_manager
    if _global_path_manager is None:
        _global_path_manager = PathManager()
    return _global_path_manager


def set_path_manager(path_manager: PathManager):
    """设置全局路径管理器实例"""
    global _global_path_manager
    _global_path_manager = path_manager


def create_new_experiment(base_dir: str = "results", experiment_name: Optional[str] = None) -> PathManager:
    """
    创建新的实验路径管理器
    
    Args:
        base_dir: 基础目录
        experiment_name: 实验名称，如果为None则使用时间戳
    """
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    path_manager = PathManager(base_dir, experiment_name)
    set_path_manager(path_manager)
    return path_manager


def print_directory_structure():
    """打印当前实验的目录结构"""
    path_manager = get_path_manager()
    info = path_manager.get_directory_info()
    
    print("\n📁 实验目录结构:")
    print(f"  实验根目录: {info['experiment_dir']}")
    print(f"  实验时间戳: {info['experiment_timestamp']}")
    print("\n📂 算法模型目录:")
    print(f"  纯MADDPG: {info['maddpg_models']}")
    print(f"  LLM+MADDPG: {info['llm_maddpg_models']}")
    print(f"  纯LLM: {info['llm_models']}")
    print(f"  MAPPO: {info['mappo_models']}")
    print(f"  HAPPO: {info['happo_models']}")
    print("\n📊 数据保存目录:")
    print(f"  CSV数据: {info['csv_data']}")
    print(f"  JSON数据: {info['json_data']}")
    print(f"  统计数据: {info['stats_data']}")
    print("\n📈 输出目录:")
    print(f"  图表: {info['plots']}")
    print(f"  日志: {info['logs']}")
    print(f"  对比结果: {info['comparison']}")
    print(f"  测试结果: {info['test_results']}")


if __name__ == "__main__":
    # 测试路径管理器
    print("测试路径管理器...")
    manager = create_new_experiment()
    print_directory_structure()
    
    # 测试路径生成
    print("\n📝 路径生成测试:")
    print(f"模型文件: {manager.get_model_file_path('maddpg', 'actor_agent_0_final.pth')}")
    print(f"CSV文件: {manager.get_data_file_path('csv', 'training_metrics.csv')}")
    print(f"图表文件: {manager.get_plot_file_path('training_curves.png')}")
    print(f"时间戳文件名: {manager.generate_timestamped_filename('training_log', '.log')}")