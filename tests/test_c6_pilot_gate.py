import json
from pathlib import Path

import pytest

from LLM4RL.experiments.ei.config import compose_ei_config
from LLM4RL.experiments.ei.pilot_gate import (
    GateThresholds,
    build_pilot_scenario_banks,
    evaluate_pilot_gate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _result_record(
    algorithm,
    seed,
    *,
    reward_auc,
    final_reward=0.0,
    tcr=0.5,
    dvr=0.25,
    online_api_calls=0,
    runtime_miss_rate=0.0,
    runtime_fallback_rate=0.0,
):
    lookups = 10
    misses = int(round(lookups * runtime_miss_rate))
    fallback_actions = int(round(lookups * runtime_fallback_rate))
    train_extra = {
        "hybrid_action": True,
        "distillation": {
            "expert": {
                "runtime_stats": {
                    "lookups": lookups,
                    "hits": lookups - misses,
                    "misses": misses,
                    "hit_rate": (lookups - misses) / lookups,
                    "decision_actions": lookups,
                    "fallback_actions": fallback_actions,
                    "fallback_rate": runtime_fallback_rate,
                }
            }
        },
    }
    eval_extra = {
        "evaluation_metrics": {
            "TCR": tcr,
            "DVR": dvr,
            "p95_task_latency": 1.2,
        },
        "online_api_calls": online_api_calls,
    }
    if algorithm in {"maps", "maps_no_annealing"}:
        eval_extra["maps_eval_uses_llm"] = False
    return {
        "algorithm": algorithm,
        "seed": seed,
        "train_result": {
            "status": "passed",
            "convergence_metrics": {
                "reward_auc": reward_auc,
                "final_reward": final_reward,
            },
            "extra": train_extra,
        },
        "eval_result": {
            "status": "passed",
            "extra": eval_extra,
        },
    }


def _expert_cache_stats(
    *,
    parser_success_rate=1.0,
    valid_action_rate=1.0,
    fallback_rate=0.0,
):
    return {
        "available": True,
        "stats": {
            "parser_success_rate": parser_success_rate,
            "valid_action_rate": valid_action_rate,
            "fallback_rate": fallback_rate,
        },
    }


def test_pilot_bank_builder_creates_short_disjoint_banks(tmp_path):
    config = compose_ei_config(
        "configs/ei/base.yaml",
        [
            "configs/ei/scales/s1_u10.yaml",
            "configs/ei/deadlines/medium.yaml",
        ],
    )

    info = build_pilot_scenario_banks(
        config=config,
        output_dir=tmp_path,
        scenario_id="unit_c6_pilot",
        train_scenario_seeds=[1000, 1001],
        test_scenario_seeds=[2000],
        train_steps=3,
        eval_steps=4,
        construction_seed=42,
    )

    train = json.loads(Path(info["train_bank"]).read_text(encoding="utf-8"))
    test = json.loads(Path(info["test_bank"]).read_text(encoding="utf-8"))
    assert train["split"] == "train"
    assert test["split"] == "test"
    assert train["scenario_bank_id"] != test["scenario_bank_id"]
    assert [item["max_steps"] for item in train["scenarios"]] == [3, 3]
    assert [item["max_steps"] for item in test["scenarios"]] == [4]
    assert {item["seed"] for item in train["scenarios"]}.isdisjoint(
        {item["seed"] for item in test["scenarios"]}
    )


def test_pilot_bank_builder_rejects_train_test_seed_overlap(tmp_path):
    config = compose_ei_config(
        "configs/ei/base.yaml",
        ["configs/ei/scales/s1_u10.yaml"],
    )

    with pytest.raises(ValueError, match="disjoint"):
        build_pilot_scenario_banks(
            config=config,
            output_dir=tmp_path,
            scenario_id="overlap",
            train_scenario_seeds=[1000],
            test_scenario_seeds=[1000],
            train_steps=2,
            eval_steps=2,
            construction_seed=42,
        )


def test_c6_gate_passes_for_complete_synthetic_pilot():
    records = []
    for seed in (42, 43):
        records.extend(
            [
                _result_record("maddpg", seed, reward_auc=1.0 + seed * 0.001),
                _result_record("mappo", seed, reward_auc=0.9 + seed * 0.001),
                _result_record(
                    "maps_no_annealing",
                    seed,
                    reward_auc=1.05 + seed * 0.001,
                ),
                _result_record("maps", seed, reward_auc=1.1 + seed * 0.001),
            ]
        )

    report = evaluate_pilot_gate(
        records,
        thresholds=GateThresholds(),
        expert_cache_stats=_expert_cache_stats(),
    )

    assert report["gate_passed"] is True
    assert all(report["checks"].values())
    assert report["summaries"]["maps"]["seed_count"] == 2
    assert report["diagnostics"]["maps_mean_reward_auc"] > report["diagnostics"][
        "maddpg_mean_reward_auc"
    ]


def test_c6_gate_flags_auc_regression_cache_miss_and_metric_degeneracy():
    records = []
    for seed in (42, 43):
        records.extend(
            [
                _result_record("maddpg", seed, reward_auc=1.0, tcr=1.0, dvr=0.0),
                _result_record("mappo", seed, reward_auc=0.9, tcr=1.0, dvr=0.0),
                _result_record(
                    "maps_no_annealing",
                    seed,
                    reward_auc=0.95,
                    tcr=1.0,
                    dvr=0.0,
                    runtime_miss_rate=0.4,
                    runtime_fallback_rate=0.4,
                ),
                _result_record(
                    "maps",
                    seed,
                    reward_auc=0.8,
                    tcr=1.0,
                    dvr=0.0,
                    runtime_miss_rate=0.4,
                    runtime_fallback_rate=0.4,
                ),
            ]
        )

    report = evaluate_pilot_gate(
        records,
        thresholds=GateThresholds(),
        expert_cache_stats=_expert_cache_stats(fallback_rate=0.5),
    )

    assert report["gate_passed"] is False
    assert report["checks"]["maps_auc_not_below_maddpg"] is False
    assert report["checks"]["expert_cache_offline_quality_high"] is False
    assert report["checks"]["expert_cache_runtime_coverage_high"] is False
    assert report["checks"]["dvr_tcr_not_degenerate"] is False
