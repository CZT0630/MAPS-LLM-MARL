import json
from pathlib import Path

from LLM4RL.experiments.ei.analyze import collect_evaluation_runs, write_outputs
from LLM4RL.experiments.ei.config import compose_ei_config
from LLM4RL.experiments.ei.scenario_bank import (
    build_scenario_bank,
    write_scenario_bank,
)
from LLM4RL.experiments.runner import run_baseline, run_evaluation
from LLM4RL.llm_assistant.ei_state import ei_environment_fingerprint
from LLM4RL.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_phase2_config():
    config = load_config(str(PROJECT_ROOT / "configs" / "smoke_phase2.yaml"))
    config.setdefault("training", {})["episodes"] = 2
    config["maddpg"].update(
        {
            "max_episodes": 2,
            "max_steps": 2,
            "batch_size": 2,
            "train_frequency": 1,
        }
    )
    config.setdefault("evaluation", {}).update(
        {
            "num_scenarios": 1,
            "window_steps": 2,
            "drain_horizon_steps": 1,
            "allow_untrained": True,
        }
    )
    return config


def test_compose_ei_config_merges_method_scale_and_deadline():
    config = compose_ei_config(
        "configs/ei/base.yaml",
        [
            "configs/ei/methods/maps.yaml",
            "configs/ei/scales/s2_u20.yaml",
            "configs/ei/deadlines/strict.yaml",
        ],
    )

    assert config["ei"]["method"] == "maps"
    assert config["ei"]["scenario_name"] == "s2_u20"
    assert config["ei"]["deadline_profile"] == "strict"
    assert config["environment"]["num_devices"] == 20
    assert config["device_specs"]["user_equipment"]["count"] == 20
    assert config["tasks"]["semantic_profiles"]["control"]["deadline_multiplier"] == 1.0
    assert len(config["ei"]["environment_fingerprint"]) == 64


def test_scenario_bank_builder_separates_train_and_test(tmp_path):
    config = _small_phase2_config()
    train_bank = build_scenario_bank(
        config=config,
        scenario_id="unit_s1",
        split="train",
        seeds=[100, 101],
        max_steps=2,
        construction_seed=42,
    )
    test_bank = build_scenario_bank(
        config=config,
        scenario_id="unit_s1",
        split="test",
        seeds=[200],
        max_steps=2,
        construction_seed=42,
    )
    train_path = write_scenario_bank(tmp_path / "train.json", train_bank)
    test_path = write_scenario_bank(tmp_path / "test.json", test_bank)

    assert train_bank["split"] == "train"
    assert test_bank["split"] == "test"
    assert train_bank["scenario_bank_id"] != test_bank["scenario_bank_id"]
    assert {item["seed"] for item in train_bank["scenarios"]}.isdisjoint(
        {item["seed"] for item in test_bank["scenarios"]}
    )
    assert train_bank["environment_fingerprint"] == ei_environment_fingerprint(config)
    assert json.loads(train_path.read_text(encoding="utf-8"))["scenario_bank_id"]
    assert json.loads(test_path.read_text(encoding="utf-8"))["scenario_bank_id"]


def test_training_run_uses_train_bank_and_writes_formal_artifacts(tmp_path):
    config = _small_phase2_config()
    bank = build_scenario_bank(
        config=config,
        scenario_id="train_unit",
        split="train",
        seeds=[123, 124],
        max_steps=2,
        construction_seed=42,
    )
    bank_path = write_scenario_bank(tmp_path / "train-bank.json", bank)
    config["training"]["scenario_bank"] = str(bank_path)

    result = run_baseline("maddpg", config, 42, tmp_path / "runs")

    assert result["status"] == "passed"
    assert result["episodes"] == 2
    assert result["extra"]["training_scenario_bank_id"] == bank["scenario_bank_id"]
    run_dir = Path(result["run_dir"])
    assert (run_dir / "resolved_config.yaml").exists()
    assert result["artifacts"]["resolved_config"] == str(run_dir / "resolved_config.yaml")
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["scenario_bank_id"] == bank["scenario_bank_id"]
    assert manifest["artifacts"]["checkpoints"]


def test_evaluation_analysis_collects_required_c5_artifacts(tmp_path):
    config = _small_phase2_config()
    bank = build_scenario_bank(
        config=config,
        scenario_id="eval_unit",
        split="test",
        seeds=[42],
        max_steps=2,
        construction_seed=42,
    )
    bank_path = write_scenario_bank(tmp_path / "test-bank.json", bank)
    config["evaluation"]["scenario_bank"] = str(bank_path)

    result = run_evaluation("greedy_min_cost", config, 42, tmp_path / "evals")
    rows = collect_evaluation_runs(tmp_path / "evals")
    outputs = write_outputs(rows, tmp_path / "analysis")

    assert result["status"] == "passed"
    assert rows
    assert rows[0]["scenario_bank_id"] == bank["scenario_bank_id"]
    assert rows[0]["required_artifacts_present"] is True
    assert Path(outputs["evaluation_summary_csv"]).exists()
    assert Path(outputs["analysis_summary_json"]).exists()


def test_committed_c5_formal_scenario_banks_are_valid():
    bank_dir = PROJECT_ROOT / "artifacts" / "ei" / "scenario_banks"
    scenario_ids = [
        "s1_u10_medium",
        "s2_u20_medium",
        "s2_u30_medium",
        "s2_u50_medium",
        "s3_u10_loose",
        "s3_u10_strict",
    ]

    for scenario_id in scenario_ids:
        train = json.loads(
            (bank_dir / f"{scenario_id}_train.json").read_text(encoding="utf-8")
        )
        test = json.loads(
            (bank_dir / f"{scenario_id}_test.json").read_text(encoding="utf-8")
        )
        assert train["schema_version"] == "ei-scenario-bank-v1"
        assert test["schema_version"] == "ei-scenario-bank-v1"
        assert train["split"] == "train"
        assert test["split"] == "test"
        assert train["scenario_bank_id"] != test["scenario_bank_id"]
        assert len(train["scenarios"]) == 5
        assert len(test["scenarios"]) == 20
        assert len(train["environment_fingerprint"]) == 64
        assert len(test["environment_fingerprint"]) == 64
        assert {item["seed"] for item in train["scenarios"]}.isdisjoint(
            {item["seed"] for item in test["scenarios"]}
        )
