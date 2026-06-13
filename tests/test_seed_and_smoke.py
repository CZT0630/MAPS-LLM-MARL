import json
import copy
from pathlib import Path

import numpy as np
import pytest

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


def test_same_seed_reproduces_short_training(tmp_path):
    config = load_config(str(PROJECT_ROOT / "configs" / "smoke.yaml"))
    config["maddpg"].update(
        {
            "max_episodes": 1,
            "max_steps": 3,
            "batch_size": 2,
            "train_frequency": 1,
        }
    )

    first = run_baseline("maddpg", config, 42, tmp_path / "first")
    second = run_baseline("maddpg", config, 42, tmp_path / "second")

    for filename in ("episode_metrics.json", "training_losses.json"):
        first_output = json.loads(
            Path(first["run_dir"], filename).read_text(encoding="utf-8")
        )
        second_output = json.loads(
            Path(second["run_dir"], filename).read_text(encoding="utf-8")
        )
        assert first_output == second_output


@pytest.mark.parametrize("num_edges", [3, 5, 10])
@pytest.mark.parametrize("algorithm", ["maddpg", "mappo"])
def test_phase2_hybrid_training_supports_dynamic_edge_counts(
    tmp_path, num_edges, algorithm
):
    config = load_config(str(PROJECT_ROOT / "configs" / "smoke_phase2.yaml"))
    config = copy.deepcopy(config)
    config["environment"]["num_edges"] = num_edges
    config["device_specs"]["edge_servers"]["count"] = num_edges
    config[algorithm]["max_episodes"] = 1
    config[algorithm]["max_steps"] = 3
    if algorithm == "maddpg":
        config[algorithm]["batch_size"] = 2
        config[algorithm]["train_frequency"] = 1

    result = run_baseline(
        algorithm,
        config,
        42,
        tmp_path / f"{algorithm}_{num_edges}",
    )

    assert result["status"] == "passed"
    assert result["updates"] > 0
    assert result["extra"]["action_dim"] == 3 + num_edges
