import csv
import json
from pathlib import Path

import pytest

from LLM4RL.experiments.ei.results_audit import METRIC_KEYS, run_results_audit


SUMMARY_FIELDS = [
    "run_dir",
    "algorithm",
    "seed",
    "scenario_bank_id",
    "scenario_count",
    *METRIC_KEYS,
    "required_artifacts_present",
    "evaluation_metrics_path",
    "task_records_path",
    "run_manifest_path",
    "resolved_config_path",
]


def _metrics(**overrides):
    base = {
        "N_gen": 100.0,
        "N_done": 70.0,
        "N_on_time": 60.0,
        "TCR": 0.70,
        "DVR": 0.40,
        "p95_task_latency": 70.0,
        "mean_latency": 30.0,
        "system_energy_per_completed_task": 200.0,
        "device_energy_per_completed_task": 4.0,
        "mean_decision_latency_ms": 5.0,
        "p95_decision_latency_ms": 8.0,
        "evaluation_reward": -50.0,
    }
    base.update(overrides)
    return base


def _scenario_rows(metrics):
    low = {key: value * 0.95 for key, value in metrics.items()}
    high = {key: value * 1.05 for key, value in metrics.items()}
    for index, payload in enumerate((low, high)):
        payload.update(
            {
                "scenario_id": f"unit-{index}",
                "scenario_index": index,
                "seed": 2000 + index,
                "window_steps": 10,
                "drain_horizon_steps": 2,
            }
        )
    return [low, high]


def _write_run(
    root: Path,
    *,
    experiment: str = "e2",
    scenario: str = "s1_u10_medium",
    algorithm: str,
    seed: int,
    metrics: dict,
    online_api_calls: int = 0,
) -> dict:
    run_dir = (
        root
        / "evaluation"
        / experiment
        / scenario
        / f"unit_{algorithm}_seed{seed}"
    )
    run_dir.mkdir(parents=True)
    metrics_path = run_dir / "evaluation_metrics.json"
    task_records_path = run_dir / "task_records.csv"
    manifest_path = run_dir / "run_manifest.json"
    config_path = run_dir / "resolved_config.yaml"
    metrics_path.write_text(
        json.dumps(
            {
                "schema_version": "c4-evaluation-v1",
                "algorithm": algorithm,
                "seed": seed,
                "metrics": metrics,
                "scenarios": _scenario_rows(metrics),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    task_records_path.write_text("task_id,completed\n1,True\n", encoding="utf-8")
    config_path.write_text("unit: true\n", encoding="utf-8")
    extra = {
        "formal_evaluation": True,
        "online_api_calls": online_api_calls,
        "llm_enabled": algorithm == "llm_only",
        "maps_eval_uses_llm": False if algorithm in {"maps", "maps_no_annealing"} else None,
    }
    if algorithm == "llm_only":
        extra.update(
            {
                "cache_coverage_complete": online_api_calls == 0,
                "cache_runtime_stats": {
                    "lookups": 10,
                    "hits": 10,
                    "misses": 0,
                    "online_api_calls": online_api_calls,
                    "fallback_actions": 0,
                    "fallback_rate": 0.0,
                },
            }
        )
    manifest_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "summary": {
                    "status": "passed",
                    "extra": extra,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    row = {
        "run_dir": str(run_dir),
        "algorithm": algorithm,
        "seed": seed,
        "scenario_bank_id": "unit-bank",
        "scenario_count": 2,
        "required_artifacts_present": True,
        "evaluation_metrics_path": str(metrics_path),
        "task_records_path": str(task_records_path),
        "run_manifest_path": str(manifest_path),
        "resolved_config_path": str(config_path),
    }
    row.update(metrics)
    return row


def _write_summary(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_results_audit_writes_bundle_and_paired_seed_contrasts(tmp_path):
    rows = [
        _write_run(
            tmp_path,
            algorithm="greedy_min_cost",
            seed=1,
            metrics=_metrics(TCR=0.40, DVR=0.80, p95_task_latency=90.0),
        ),
        _write_run(
            tmp_path,
            algorithm="llm_only",
            seed=1,
            metrics=_metrics(TCR=0.75, DVR=0.35, p95_task_latency=55.0),
        ),
    ]
    for seed, maps_tcr, maddpg_tcr, mappo_tcr in [
        (1, 0.80, 0.50, 0.68),
        (2, 0.86, 0.56, 0.70),
    ]:
        rows.extend(
            [
                _write_run(
                    tmp_path,
                    algorithm="maps",
                    seed=seed,
                    metrics=_metrics(TCR=maps_tcr, DVR=0.25, p95_task_latency=45.0),
                ),
                _write_run(
                    tmp_path,
                    algorithm="maddpg",
                    seed=seed,
                    metrics=_metrics(TCR=maddpg_tcr, DVR=0.50, p95_task_latency=72.0),
                ),
                _write_run(
                    tmp_path,
                    algorithm="mappo",
                    seed=seed,
                    metrics=_metrics(TCR=mappo_tcr, DVR=0.45, p95_task_latency=68.0),
                ),
            ]
        )

    input_csv = _write_summary(tmp_path / "evaluation_summary.csv", rows)
    output_dir = tmp_path / "audit"
    result = run_results_audit(
        input_csv,
        output_dir,
        expected_learning_seeds=2,
        expected_baseline_seeds=1,
    )

    summary = json.loads(Path(result["outputs"]["audit_summary_json"]).read_text())
    assert result["run_count"] == 8
    assert summary["gates"]["zero_online_api_calls_in_evaluation"] is True
    assert Path(result["outputs"]["analysis_report"]).exists()
    assert Path(result["outputs"]["figures"][0]["filename"]).read_text().startswith("<svg")

    descriptive = _read_csv(output_dir / "descriptive_stats.csv")
    maps_tcr = next(
        row
        for row in descriptive
        if row["algorithm"] == "maps" and row["metric"] == "TCR"
    )
    assert maps_tcr["n"] == "2"
    assert float(maps_tcr["mean"]) == pytest.approx(0.83)
    assert maps_tcr["inference_status"] == "ok"

    contrasts = _read_csv(output_dir / "paired_contrasts.csv")
    maps_vs_maddpg = next(
        row
        for row in contrasts
        if row["method_b"] == "maddpg" and row["metric"] == "TCR"
    )
    assert maps_vs_maddpg["status"] == "valid"
    assert maps_vs_maddpg["n_pairs"] == "2"
    assert float(maps_vs_maddpg["mean_improvement"]) == pytest.approx(0.30)
    assert maps_vs_maddpg["maps_better_count"] == "2"

    maps_vs_greedy = next(
        row
        for row in contrasts
        if row["method_b"] == "greedy_min_cost" and row["metric"] == "TCR"
    )
    assert maps_vs_greedy["status"] == "blocked_single_run_baseline"
    assert "single-seed baseline" in maps_vs_greedy["blocker"]


def test_results_audit_flags_online_api_calls_in_evaluation(tmp_path):
    row = _write_run(
        tmp_path,
        algorithm="llm_only",
        seed=1,
        metrics=_metrics(TCR=0.75),
        online_api_calls=1,
    )
    input_csv = _write_summary(tmp_path / "evaluation_summary.csv", [row])
    result = run_results_audit(
        input_csv,
        tmp_path / "audit",
        expected_learning_seeds=None,
        expected_baseline_seeds=1,
    )

    summary = json.loads(Path(result["outputs"]["audit_summary_json"]).read_text())
    assert summary["gates"]["zero_online_api_calls_in_evaluation"] is False
    assert any(
        issue["code"] == "online_api_calls_in_evaluation"
        for issue in summary["issues"]
    )
