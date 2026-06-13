# utils/config.py
from pathlib import Path

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_project_env(env_path=None):
    """Load local secrets without overriding explicit process variables."""
    path = Path(env_path) if env_path is not None else PROJECT_ROOT / ".env"
    return load_dotenv(dotenv_path=path, override=False)


def load_config(config_path="config.yaml"):
    """加载配置文件"""
    load_project_env()
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f) or {}
        base = get_default_config()
        merged = deep_update(dict(base), loaded)
        if 'seed' not in merged:
            merged['seed'] = 42
        return merged
    except FileNotFoundError:
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        return get_default_config()

def get_default_config():
    """返回默认配置"""
    return {
        "environment": {
            "num_devices": 5,
            "num_edges": 2,
            "num_clouds": 1,
            "action_space": {"type": "continuous"},
            "device_config": {
                "cpu_capacity": 2.0,
                "memory_capacity": 4.0
            },
            "edge_config": {
                "cpu_capacity": 8.0,
                "memory_capacity": 16.0
            },
            "cloud_config": {
                "cpu_capacity": 32.0,
                "memory_capacity": 64.0
            },
            "task_config": {
                "types": ["computation_intensive", "data_intensive", "balanced"],
                "min_computation": 100,
                "max_computation": 1000,
                "min_data_size": 1,
                "max_data_size": 100,
                "min_deadline": 10,
                "max_deadline": 60
            }
        },
        "network": {
            "ue_to_edge_rate": 1e9,
            "ue_to_cloud_total_rate": 1e8,
            "edge_to_cloud_rate": 1e9
        },
        "tasks": {
            "simple_mode": True,
            "simple_data_choices": [100, 150, 200, 250],
            "simple_deadline_multiplier": 3.0,
            "generator_verbose": False,
            "fixed_deadline_seconds": None
        },
        "reward": {
            "type": "inverse",
            "latency_weight": 1.0,
            "energy_weight": 1.0,
            "deadline_penalty": 0.0
        },
        "llm": {
            "provider": "xiaomi_mimo",
            "display_name": "MiMo-V2.5",
            "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
            "endpoint": "/chat/completions",
            "model_name": "mimo-v2.5",
            "api_key_env": "MIMO_API_KEY",
            "temperature": 0.0,
            "max_completion_tokens": 4096,
            "enable_thinking": False,
            "query_frequency": 1  # 每多少个episode查询一次LLM
        },
        "maddpg": {
            "lr_actor": 1e-4,
            "lr_critic": 1e-3,
            "gamma": 0.99,
            "tau": 0.001,
            "buffer_size": int(1e6),
            "batch_size": 100,
            "max_episodes": 1000,
            "max_steps": 200
        },
        "happo": {
            "lr_actor": 3e-4,
            "lr_critic": 1e-3,
            "gamma": 0.99,
            "lam": 0.95,
            "clip_range": 0.2,
            "kl_coeff": 0.5,
            "entropy_coeff": 0.01,
            "value_coeff": 0.5,
            "update_epochs": 4,
            "batch_size": 64,
            "max_episodes": 200,
            "max_steps": 200
        },
        "mappo": {
            "lr_actor": 3e-4,
            "lr_critic": 1e-3,
            "gamma": 0.99,
            "lam": 0.95,
            "clip_range": 0.2,
            "entropy_coeff": 0.01,
            "value_coeff": 0.5,
            "update_epochs": 4,
            "batch_size": 64,
            "max_episodes": 200,
            "max_steps": 200
        },
        "seed": 42
    }

def deep_update(base, overrides):
    """深度合并字典"""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base
