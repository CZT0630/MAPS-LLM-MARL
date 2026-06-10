from pathlib import Path

import numpy as np

from LLM4RL.environment.cloud_edge_env import CloudEdgeDeviceEnv
from LLM4RL.experiments.runner import run_baseline
from LLM4RL.utils.config import load_config
from LLM4RL.utils.seed import set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_observation(seed):
    set_global_seed(seed)
    config = load_config(str(PROJECT_ROOT / "configs" / "smoke.yaml"))
    env = CloudEdgeDeviceEnv(config)
    observation, _ = env.reset(seed=seed)
    return observation


def test_same_seed_reproduces_initial_observation():
    assert np.array_equal(make_observation(42), make_observation(42))


def test_different_seed_changes_initial_observation():
    assert not np.array_equal(make_observation(42), make_observation(43))


def test_maddpg_smoke_run_writes_nonempty_result(tmp_path):
    config = load_config(str(PROJECT_ROOT / "configs" / "smoke.yaml"))
    config["maddpg"]["max_episodes"] = 1
    config["maddpg"]["max_steps"] = 3
    config["maddpg"]["batch_size"] = 2
    config["maddpg"]["train_frequency"] = 1

    result = run_baseline("maddpg", config, 42, tmp_path)

    assert result["status"] == "passed"
    assert result["episodes"] == 1
    assert result["updates"] > 0
    assert Path(result["run_dir"], "run_manifest.json").exists()
