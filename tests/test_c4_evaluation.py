import csv
import json
from pathlib import Path

import numpy as np
import pytest

from LLM4RL.environment.cloud_edge_env import CloudEdgeDeviceEnv
from LLM4RL.experiments.runner import run_baseline, run_evaluation
from LLM4RL.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _phase2_config():
    config = load_config(str(PROJECT_ROOT / "configs" / "smoke_phase2.yaml"))
    config.setdefault("evaluation", {}).update(
        {
            "num_scenarios": 1,
            "window_steps": 2,
            "drain_horizon_steps": 1,
            "reward_threshold": None,
        }
    )
    return config


def test_environment_records_c4_task_ledger_fields():
    config = _phase2_config()
    env = CloudEdgeDeviceEnv(config)
    _obs, _ = env.reset(
        seed=42,
        options={"evaluation": True, "scenario_id": "unit-scenario"},
    )
    env.set_task_arrivals_enabled(False)
    actions = np.tile([1.0, 0.0, 0.0, 0.0], (env.num_devices, 1))

    _next, _rewards, _terminated, _truncated, info = env.step(
        actions,
        decision_latency_ms=12.5,
    )

    records = env.get_task_records()
    assert info["evaluation_mode"] is True
    assert info["task_arrivals_enabled"] is False
    assert records
    required = {
        "task_id",
        "scenario_id",
        "arrival_time",
        "deadline",
        "completion_time",
        "latency",
        "completed",
        "completed_on_time",
        "device_energy_j",
        "system_energy_j",
        "decision_latency_ms",
    }
    assert required.issubset(records[0])
    completed = [record for record in records if record["completed"]]
    assert completed
    assert completed[0]["scenario_id"] == "unit-scenario"
    assert completed[0]["decision_latency_ms"] == pytest.approx(12.5)
    assert completed[0]["system_energy_j"] >= completed[0]["device_energy_j"]


def test_run_evaluation_writes_c4_metrics_and_task_records(tmp_path):
    config = _phase2_config()

    result = run_evaluation("greedy_min_cost", config, 42, tmp_path)

    assert result["status"] == "passed"
    extra = result["extra"]
    assert extra["formal_evaluation"] is True
    metrics = extra["evaluation_metrics"]
    assert metrics["N_gen"] > 0
    assert 0.0 <= metrics["TCR"] <= 1.0
    assert 0.0 <= metrics["DVR"] <= 1.0
    assert "p95_task_latency" in metrics
    assert metrics["mean_decision_latency_ms"] is not None

    metrics_path = Path(extra["evaluation_metrics_path"])
    records_path = Path(extra["task_records_path"])
    assert metrics_path.exists()
    assert records_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "c4-evaluation-v1"
    assert payload["metrics"]["N_gen"] == metrics["N_gen"]
    with records_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == metrics["N_gen"]
    assert {
        "task_id",
        "scenario_id",
        "completion_time",
        "completed",
        "completed_on_time",
        "decision_latency_ms",
    }.issubset(rows[0])


def test_learning_evaluation_requires_checkpoint_by_default(tmp_path):
    config = _phase2_config()

    with pytest.raises(ValueError, match="checkpoint_dir is required"):
        run_evaluation("maddpg", config, 42, tmp_path)


def test_training_summary_includes_c4_convergence_metrics(tmp_path):
    config = _phase2_config()
    config["maddpg"].update(
        {
            "max_episodes": 1,
            "max_steps": 2,
            "batch_size": 2,
            "train_frequency": 1,
        }
    )

    result = run_baseline("maddpg", config, 42, tmp_path)

    convergence = result["convergence_metrics"]
    assert "reward_auc" in convergence
    assert "final_reward" in convergence
    assert convergence["threshold_status"] == "not_configured"
