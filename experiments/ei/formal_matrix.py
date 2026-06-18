"""C7 formal EI matrix orchestration.

This module expands the submission matrix into reproducible training and
evaluation jobs.  It deliberately reuses the C5 train/evaluate primitives and
adds a thin index/resume layer around them.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

PACKAGE_PARENT = Path(__file__).resolve().parents[3]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from LLM4RL.experiments.ei.analyze import (  # noqa: E402
    collect_evaluation_runs,
    write_outputs,
)
from LLM4RL.experiments.ei.config import (  # noqa: E402
    compose_ei_config,
    resolve_project_path,
    with_bank_paths,
)
from LLM4RL.experiments.runner import run_baseline, run_evaluation  # noqa: E402


DEFAULT_MATRIX_CONFIG = "configs/ei/formal_matrix.yaml"
INDEX_SCHEMA_VERSION = "ei-formal-matrix-index-v1"
LEARNING_METHODS = {"maddpg", "mappo", "maps_no_annealing", "maps"}
MAPS_METHODS = {"maps", "maps_no_annealing"}


@dataclass(frozen=True)
class MatrixJob:
    kind: str
    experiment: str
    scenario: str
    algorithm: str
    seed: int
    key: str
    scale_config: str
    deadline_config: str
    train_bank: str | None = None
    test_bank: str | None = None
    checkpoint_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_yaml(path_value: str | Path) -> dict[str, Any]:
    path = resolve_project_path(path_value)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_matrix_config(path_value: str | Path = DEFAULT_MATRIX_CONFIG) -> dict[str, Any]:
    data = _load_yaml(path_value)
    matrix = data.get("formal_matrix", {})
    if not matrix:
        raise ValueError("formal_matrix config section is required")
    return matrix


def method_fragment(algorithm: str) -> str:
    return f"configs/ei/methods/{algorithm}.yaml"


def _scenario(matrix: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    scenarios = matrix.get("scenarios", {})
    if scenario_id not in scenarios:
        raise KeyError(f"scenario {scenario_id!r} is not configured")
    return scenarios[scenario_id]


def _job_key(
    *,
    kind: str,
    experiment: str,
    scenario: str,
    algorithm: str,
    seed: int,
) -> str:
    return f"{kind}:{experiment}:{scenario}:{algorithm}:seed{seed}"


def _training_job(
    matrix: dict[str, Any],
    *,
    experiment: str,
    scenario_id: str,
    algorithm: str,
    seed: int,
) -> MatrixJob:
    scenario = _scenario(matrix, scenario_id)
    return MatrixJob(
        kind="train",
        experiment=experiment,
        scenario=scenario_id,
        algorithm=algorithm,
        seed=int(seed),
        key=_job_key(
            kind="train",
            experiment=experiment,
            scenario=scenario_id,
            algorithm=algorithm,
            seed=int(seed),
        ),
        scale_config=scenario["scale_config"],
        deadline_config=scenario["deadline_config"],
        train_bank=scenario["train_bank"],
        test_bank=scenario["test_bank"],
    )


def _evaluation_job(
    matrix: dict[str, Any],
    *,
    experiment: str,
    scenario_id: str,
    algorithm: str,
    seed: int,
    checkpoint_ref: str | None,
) -> MatrixJob:
    scenario = _scenario(matrix, scenario_id)
    return MatrixJob(
        kind="eval",
        experiment=experiment,
        scenario=scenario_id,
        algorithm=algorithm,
        seed=int(seed),
        key=_job_key(
            kind="eval",
            experiment=experiment,
            scenario=scenario_id,
            algorithm=algorithm,
            seed=int(seed),
        ),
        scale_config=scenario["scale_config"],
        deadline_config=scenario["deadline_config"],
        test_bank=scenario["test_bank"],
        checkpoint_ref=checkpoint_ref,
    )


def build_training_jobs(matrix: dict[str, Any]) -> list[MatrixJob]:
    seeds = [int(seed) for seed in matrix["seeds"]]
    experiments = matrix["experiments"]
    jobs: list[MatrixJob] = []

    e1 = experiments["e1"]
    for algorithm in e1["learning_methods"]:
        for seed in seeds:
            jobs.append(
                _training_job(
                    matrix,
                    experiment="e1",
                    scenario_id=e1["scenario"],
                    algorithm=algorithm,
                    seed=seed,
                )
            )

    e3 = experiments["e3"]
    for scenario_id in e3["scenarios"]:
        for algorithm in e3["learning_methods"]:
            for seed in seeds:
                jobs.append(
                    _training_job(
                        matrix,
                        experiment="e3",
                        scenario_id=scenario_id,
                        algorithm=algorithm,
                        seed=seed,
                    )
                )
    return jobs


def _checkpoint_ref(
    *,
    experiment: str,
    scenario: str,
    algorithm: str,
    seed: int,
) -> str:
    return _job_key(
        kind="train",
        experiment=experiment,
        scenario=scenario,
        algorithm=algorithm,
        seed=seed,
    )


def build_evaluation_jobs(matrix: dict[str, Any]) -> list[MatrixJob]:
    seeds = [int(seed) for seed in matrix["seeds"]]
    baseline_seed = int(matrix.get("baseline_eval_seed", seeds[0]))
    experiments = matrix["experiments"]
    jobs: list[MatrixJob] = []

    e1 = experiments["e1"]
    e2 = experiments["e2"]
    e2_scenario = e2["scenario"]
    for algorithm in e2["baseline_methods"]:
        jobs.append(
            _evaluation_job(
                matrix,
                experiment="e2",
                scenario_id=e2_scenario,
                algorithm=algorithm,
                seed=baseline_seed,
                checkpoint_ref=None,
            )
        )
    for algorithm in e1["learning_methods"]:
        for seed in seeds:
            jobs.append(
                _evaluation_job(
                    matrix,
                    experiment="e2",
                    scenario_id=e2_scenario,
                    algorithm=algorithm,
                    seed=seed,
                    checkpoint_ref=_checkpoint_ref(
                        experiment="e1",
                        scenario=e1["scenario"],
                        algorithm=algorithm,
                        seed=seed,
                    ),
                )
            )

    e3 = experiments["e3"]
    for scenario_id in e3["scenarios"]:
        for algorithm in e3["baseline_methods"]:
            jobs.append(
                _evaluation_job(
                    matrix,
                    experiment="e3",
                    scenario_id=scenario_id,
                    algorithm=algorithm,
                    seed=baseline_seed,
                    checkpoint_ref=None,
                )
            )
        for algorithm in e3["learning_methods"]:
            for seed in seeds:
                jobs.append(
                    _evaluation_job(
                        matrix,
                        experiment="e3",
                        scenario_id=scenario_id,
                        algorithm=algorithm,
                        seed=seed,
                        checkpoint_ref=_checkpoint_ref(
                            experiment="e3",
                            scenario=scenario_id,
                            algorithm=algorithm,
                            seed=seed,
                        ),
                    )
                )

    e4 = experiments["e4"]
    for scenario_id in e4["test_scenarios"]:
        for algorithm in e4["baseline_methods"]:
            jobs.append(
                _evaluation_job(
                    matrix,
                    experiment="e4",
                    scenario_id=scenario_id,
                    algorithm=algorithm,
                    seed=baseline_seed,
                    checkpoint_ref=None,
                )
            )
        for algorithm in e4["learning_methods"]:
            for seed in seeds:
                jobs.append(
                    _evaluation_job(
                        matrix,
                        experiment="e4",
                        scenario_id=scenario_id,
                        algorithm=algorithm,
                        seed=seed,
                        checkpoint_ref=_checkpoint_ref(
                            experiment="e1",
                            scenario=e1["scenario"],
                            algorithm=algorithm,
                            seed=seed,
                        ),
                    )
                )
    return jobs


def build_matrix_plan(matrix: dict[str, Any]) -> dict[str, Any]:
    training_jobs = build_training_jobs(matrix)
    evaluation_jobs = build_evaluation_jobs(matrix)
    return {
        "schema_version": "ei-formal-matrix-plan-v1",
        "training_job_count": len(training_jobs),
        "evaluation_job_count": len(evaluation_jobs),
        "training_jobs": [job.to_dict() for job in training_jobs],
        "evaluation_jobs": [job.to_dict() for job in evaluation_jobs],
    }


def _index_path(output_root: Path) -> Path:
    return output_root / "formal_matrix_index.json"


def load_index(output_root: str | Path) -> dict[str, Any]:
    path = _index_path(resolve_project_path(output_root))
    if not path.exists():
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training": {},
            "evaluation": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def write_index(output_root: str | Path, index: dict[str, Any]) -> Path:
    root = resolve_project_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    index["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = _index_path(root)
    path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def _config_for_job(
    matrix: dict[str, Any],
    job: MatrixJob,
    *,
    expert_cache: str | None,
    live_fill_cache_output: str | None,
) -> dict[str, Any]:
    fragments = [
        method_fragment(job.algorithm),
        job.scale_config,
        job.deadline_config,
    ]
    config = compose_ei_config(matrix["base_config"], fragments)
    config = with_bank_paths(
        config,
        train_bank=job.train_bank,
        test_bank=job.test_bank,
    )
    cache_path = expert_cache or matrix.get("expert_cache")
    if cache_path:
        config.setdefault("expert_cache", {})["path"] = str(
            resolve_project_path(cache_path)
        )
    if live_fill_cache_output:
        expert_cfg = config.setdefault("expert_cache", {})
        expert_cfg["live_fill_on_miss"] = True
        expert_cfg["live_fill_output"] = str(
            resolve_project_path(live_fill_cache_output)
        )
    return config


def _runtime_cache_rates(result: dict[str, Any]) -> dict[str, float | None]:
    stats = (
        result.get("extra", {})
        .get("distillation", {})
        .get("expert", {})
        .get("runtime_stats", {})
    )
    lookups = float(stats.get("lookups") or 0.0)
    miss_rate = (
        float(stats.get("misses", 0.0)) / lookups
        if lookups > 0.0
        else None
    )
    fallback_rate = (
        float(stats.get("fallback_rate"))
        if stats.get("fallback_rate") is not None
        else None
    )
    return {
        "miss_rate": miss_rate,
        "fallback_rate": fallback_rate,
    }


def maps_cache_coverage_ok(matrix: dict[str, Any], result: dict[str, Any]) -> bool:
    if not bool(matrix.get("require_maps_cache_coverage", True)):
        return True
    rates = _runtime_cache_rates(result)
    if rates["miss_rate"] is None:
        return False
    return (
        rates["miss_rate"] <= float(matrix.get("max_runtime_cache_miss_rate", 0.0))
        and (rates["fallback_rate"] or 0.0)
        <= float(matrix.get("max_runtime_fallback_rate", 0.0))
    )


def _expert_runtime_stats(result: dict[str, Any]) -> dict[str, Any]:
    return (
        result.get("extra", {})
        .get("distillation", {})
        .get("expert", {})
        .get("runtime_stats", {})
    )


def normalize_cache_prefill_result(
    job: MatrixJob,
    result: dict[str, Any],
) -> bool:
    """Mark MAPS live-fill runs as prefill artifacts, not formal passes.

    Returns True when the matrix runner should continue to the next job.
    Non-MAPS jobs and real failures keep the normal stop-on-failure behavior.
    """
    if job.algorithm not in MAPS_METHODS:
        return result.get("status") == "passed"

    status = result.get("status")
    if status == "passed":
        result["status"] = "cache_prefill_completed"
        result["cache_prefill_requires_frozen_replay"] = True
        return True

    if status != "cache_coverage_failed":
        return False

    stats = _expert_runtime_stats(result)
    if float(stats.get("fallback_rate") or 0.0) > 0.0:
        result["status"] = "cache_prefill_failed"
        result["cache_prefill_requires_frozen_replay"] = True
        return False

    result["status"] = "cache_prefill_completed"
    result["cache_prefill_requires_frozen_replay"] = True
    return True


def run_training_job(
    matrix: dict[str, Any],
    job: MatrixJob,
    *,
    output_root: Path,
    expert_cache: str | None = None,
    live_fill_cache_output: str | None = None,
) -> dict[str, Any]:
    config = _config_for_job(
        matrix,
        job,
        expert_cache=expert_cache,
        live_fill_cache_output=live_fill_cache_output,
    )
    result = run_baseline(
        job.algorithm,
        config,
        job.seed,
        output_root / "training" / job.experiment / job.scenario,
        command="experiments.ei.formal_matrix",
    )
    if job.algorithm in MAPS_METHODS:
        result["formal_cache_coverage"] = _runtime_cache_rates(result)
        if not maps_cache_coverage_ok(matrix, result):
            result["status"] = "cache_coverage_failed"
    return result


def _checkpoint_dir(index: dict[str, Any], checkpoint_ref: str | None) -> str | None:
    if checkpoint_ref is None:
        return None
    record = index.get("training", {}).get(checkpoint_ref)
    if not record:
        raise KeyError(f"missing checkpoint training job: {checkpoint_ref}")
    result = record.get("result", {})
    if result.get("status") != "passed":
        raise ValueError(f"checkpoint job did not pass: {checkpoint_ref}")
    return result.get("run_dir")


def run_eval_job(
    matrix: dict[str, Any],
    job: MatrixJob,
    *,
    output_root: Path,
    index: dict[str, Any],
    expert_cache: str | None = None,
) -> dict[str, Any]:
    config = _config_for_job(
        matrix,
        job,
        expert_cache=expert_cache,
        live_fill_cache_output=None,
    )
    checkpoint_dir = _checkpoint_dir(index, job.checkpoint_ref)
    if checkpoint_dir:
        config.setdefault("evaluation", {})["checkpoint_dir"] = checkpoint_dir
    return run_evaluation(
        job.algorithm,
        config,
        job.seed,
        output_root / "evaluation" / job.experiment / job.scenario,
        command="experiments.ei.formal_matrix",
    )


def _select_jobs(
    jobs: list[MatrixJob],
    experiments: set[str] | None,
    max_jobs: int | None,
) -> list[MatrixJob]:
    selected = [
        job for job in jobs if experiments is None or job.experiment in experiments
    ]
    return selected[:max_jobs] if max_jobs is not None else selected


def _already_passed(index: dict[str, Any], bucket: str, job: MatrixJob) -> bool:
    record = index.get(bucket, {}).get(job.key)
    return bool(record and record.get("result", {}).get("status") == "passed")


def execute_matrix(
    matrix: dict[str, Any],
    *,
    stage: str,
    experiments: set[str] | None,
    dry_run: bool,
    force: bool,
    max_jobs: int | None,
    expert_cache: str | None,
    live_fill_cache_output: str | None,
    cache_prefill: bool,
) -> dict[str, Any]:
    if cache_prefill and not live_fill_cache_output:
        raise ValueError("--cache-prefill requires --live-fill-cache-output")
    output_root = resolve_project_path(matrix["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    training_jobs = _select_jobs(build_training_jobs(matrix), experiments, max_jobs)
    eval_max = None if stage == "evaluation" else max_jobs
    evaluation_jobs = _select_jobs(build_evaluation_jobs(matrix), experiments, eval_max)
    plan = {
        "training_job_count": len(training_jobs),
        "evaluation_job_count": len(evaluation_jobs),
        "training_jobs": [job.to_dict() for job in training_jobs],
        "evaluation_jobs": [job.to_dict() for job in evaluation_jobs],
    }
    if dry_run:
        return {"dry_run": True, **plan}

    index = load_index(output_root)
    index.setdefault("matrix", copy.deepcopy(matrix))
    index.setdefault("training", {})
    index.setdefault("evaluation", {})

    executed = {"training": 0, "evaluation": 0, "skipped": 0}
    if stage in {"training", "all"}:
        for job in training_jobs:
            if not force and _already_passed(index, "training", job):
                executed["skipped"] += 1
                continue
            result = run_training_job(
                matrix,
                job,
                output_root=output_root,
                expert_cache=expert_cache,
                live_fill_cache_output=live_fill_cache_output,
            )
            index["training"][job.key] = {
                "job": job.to_dict(),
                "result": result,
            }
            executed["training"] += 1
            write_index(output_root, index)
            should_continue = (
                normalize_cache_prefill_result(job, result)
                if cache_prefill
                else result.get("status") == "passed"
            )
            if cache_prefill:
                index["training"][job.key]["result"] = result
                write_index(output_root, index)
            if not should_continue:
                break

    if stage in {"evaluation", "all"}:
        for job in evaluation_jobs:
            if not force and _already_passed(index, "evaluation", job):
                executed["skipped"] += 1
                continue
            result = run_eval_job(
                matrix,
                job,
                output_root=output_root,
                index=index,
                expert_cache=expert_cache,
            )
            index["evaluation"][job.key] = {
                "job": job.to_dict(),
                "result": result,
            }
            executed["evaluation"] += 1
            write_index(output_root, index)
            if result.get("status") != "passed":
                break

    analysis_outputs = {}
    if stage in {"analysis", "all", "evaluation"}:
        rows = collect_evaluation_runs(output_root / "evaluation")
        analysis_outputs = write_outputs(rows, output_root / "analysis")

    index_path = write_index(output_root, index)
    return {
        "dry_run": False,
        **plan,
        "executed": executed,
        "index_path": str(index_path),
        "analysis_outputs": analysis_outputs,
    }


def parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the C7 EI formal matrix")
    parser.add_argument("--config", default=DEFAULT_MATRIX_CONFIG)
    parser.add_argument(
        "--stage",
        choices=["training", "evaluation", "analysis", "all"],
        default="all",
    )
    parser.add_argument("--experiments", help="Comma list such as e1,e2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--expert-cache")
    parser.add_argument(
        "--live-fill-cache-output",
        help="Enable MiMo-on-miss during MAPS training for cache prefill only.",
    )
    parser.add_argument(
        "--cache-prefill",
        action="store_true",
        help=(
            "Continue MAPS training live-fill runs after cache misses and mark "
            "them as requiring frozen replay instead of formal passes."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    matrix = load_matrix_config(args.config)
    if args.output_root:
        matrix["output_root"] = args.output_root
    result = execute_matrix(
        matrix,
        stage=args.stage,
        experiments=parse_csv(args.experiments),
        dry_run=args.dry_run,
        force=args.force,
        max_jobs=args.max_jobs,
        expert_cache=args.expert_cache,
        live_fill_cache_output=args.live_fill_cache_output,
        cache_prefill=args.cache_prefill,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if result.get("dry_run"):
        return 0
    executed = result.get("executed", {})
    return 0 if executed.get("training", 0) or executed.get("evaluation", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
