"""C6 pilot-gate orchestration for EI formal experiments.

The pilot gate is intentionally a thin orchestration layer over the C5
training/evaluation entry points.  It creates a short frozen S1 bank, runs the
required pilot algorithms for the requested training seeds, and writes a
machine-readable gate report.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

PACKAGE_PARENT = Path(__file__).resolve().parents[3]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from LLM4RL.experiments.ei.config import (  # noqa: E402
    compose_ei_config,
    resolve_project_path,
    with_bank_paths,
)
from LLM4RL.experiments.ei.scenario_bank import (  # noqa: E402
    build_scenario_bank,
    write_scenario_bank,
)
from LLM4RL.experiments.runner import run_baseline, run_evaluation  # noqa: E402
from LLM4RL.llm_assistant.expert_cache import ExpertCache  # noqa: E402
from LLM4RL.utils.run_manifest import sanitize_config  # noqa: E402


REQUIRED_PILOT_ALGORITHMS = (
    "maddpg",
    "mappo",
    "maps_no_annealing",
    "maps",
)
DEFAULT_BASE_CONFIG = "configs/ei/base.yaml"
DEFAULT_SCALE_CONFIG = "configs/ei/scales/s1_u10.yaml"
DEFAULT_DEADLINE_CONFIG = "configs/ei/deadlines/medium.yaml"
DEFAULT_PILOT_CONFIG = "configs/ei/pilot.yaml"
REPORT_SCHEMA_VERSION = "ei-c6-pilot-gate-v1"


@dataclass(frozen=True)
class GateThresholds:
    """Numerical thresholds used by the pilot gate checks."""

    auc_tolerance: float = 0.0
    min_auc_delta: float = 1e-9
    min_parser_success_rate: float = 0.95
    min_valid_action_rate: float = 0.95
    max_cache_fallback_rate: float = 0.05
    max_runtime_cache_miss_rate: float = 0.05
    max_runtime_fallback_rate: float = 0.05

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "GateThresholds":
        if not data:
            return cls()
        allowed = {field for field in cls.__dataclass_fields__}
        values = {key: data[key] for key in allowed if key in data}
        return cls(**values)

    def to_dict(self) -> dict[str, float]:
        return {
            "auc_tolerance": self.auc_tolerance,
            "min_auc_delta": self.min_auc_delta,
            "min_parser_success_rate": self.min_parser_success_rate,
            "min_valid_action_rate": self.min_valid_action_rate,
            "max_cache_fallback_rate": self.max_cache_fallback_rate,
            "max_runtime_cache_miss_rate": self.max_runtime_cache_miss_rate,
            "max_runtime_fallback_rate": self.max_runtime_fallback_rate,
        }


def parse_csv_ints(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def parse_csv_strings(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _load_yaml(path_value: str | Path) -> dict[str, Any]:
    path = resolve_project_path(path_value)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _method_fragment(algorithm: str) -> str:
    return f"configs/ei/methods/{algorithm}.yaml"


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [number for value in values if (number := _numeric(value)) is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _all_present(mapping: dict[str, Any], keys: Iterable[str]) -> bool:
    return all(key in mapping and mapping[key] is not None for key in keys)


def _metric(result: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = result
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _load_expert_cache_stats(config: dict[str, Any]) -> dict[str, Any]:
    cache_path = config.get("expert_cache", {}).get("path")
    if not cache_path:
        return {"available": False, "reason": "expert_cache.path is not configured"}
    resolved = resolve_project_path(cache_path)
    if not resolved.exists():
        return {
            "available": False,
            "cache_path": str(resolved),
            "reason": "expert cache file does not exist",
        }
    cache = ExpertCache.load(resolved)
    return {
        "available": True,
        "cache_path": str(resolved),
        "file_sha256": cache.file_sha256,
        "provider": cache.provider,
        "requested_model": cache.requested_model,
        "prompt_version": cache.prompt_version,
        "state_key_version": cache.state_key_version,
        "stats": cache.stats(),
    }


def _runtime_cache_stats(record: dict[str, Any]) -> dict[str, Any] | None:
    train_extra = record.get("train_result", {}).get("extra", {})
    train_stats = _metric(train_extra, ("distillation", "expert", "runtime_stats"))
    if isinstance(train_stats, dict) and train_stats:
        return train_stats
    expert_stats = _metric(train_extra, ("expert", "runtime_stats"))
    if isinstance(expert_stats, dict) and expert_stats:
        return expert_stats
    eval_extra = record.get("eval_result", {}).get("extra", {})
    eval_stats = eval_extra.get("cache_runtime_stats")
    return eval_stats if isinstance(eval_stats, dict) and eval_stats else None


def summarize_pilot_records(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate per-seed train/eval records by algorithm."""

    summaries: dict[str, dict[str, Any]] = {}
    for record in records:
        algorithm = str(record["algorithm"])
        train_result = record.get("train_result", {})
        eval_result = record.get("eval_result", {})
        eval_metrics = eval_result.get("extra", {}).get("evaluation_metrics", {})
        summary = summaries.setdefault(
            algorithm,
            {
                "algorithm": algorithm,
                "seeds": [],
                "train_statuses": [],
                "eval_statuses": [],
                "reward_auc": [],
                "final_reward": [],
                "TCR": [],
                "DVR": [],
                "p95_task_latency": [],
                "online_api_calls": [],
                "hybrid_action": [],
                "maps_eval_uses_llm": [],
                "runtime_cache_stats": [],
            },
        )
        summary["seeds"].append(int(record["seed"]))
        summary["train_statuses"].append(train_result.get("status"))
        summary["eval_statuses"].append(eval_result.get("status"))
        summary["reward_auc"].append(
            _metric(train_result, ("convergence_metrics", "reward_auc"))
        )
        summary["final_reward"].append(
            _metric(train_result, ("convergence_metrics", "final_reward"))
        )
        summary["TCR"].append(eval_metrics.get("TCR"))
        summary["DVR"].append(eval_metrics.get("DVR"))
        summary["p95_task_latency"].append(eval_metrics.get("p95_task_latency"))
        summary["online_api_calls"].append(
            eval_result.get("extra", {}).get("online_api_calls")
        )
        summary["hybrid_action"].append(train_result.get("extra", {}).get("hybrid_action"))
        summary["maps_eval_uses_llm"].append(
            eval_result.get("extra", {}).get("maps_eval_uses_llm")
        )
        runtime_stats = _runtime_cache_stats(record)
        if runtime_stats is not None:
            summary["runtime_cache_stats"].append(runtime_stats)

    for summary in summaries.values():
        summary["seed_count"] = len(summary["seeds"])
        summary["mean_reward_auc"] = _mean(summary["reward_auc"])
        summary["mean_final_reward"] = _mean(summary["final_reward"])
        summary["mean_TCR"] = _mean(summary["TCR"])
        summary["mean_DVR"] = _mean(summary["DVR"])
        summary["mean_p95_task_latency"] = _mean(summary["p95_task_latency"])
        summary["total_online_api_calls"] = sum(
            int(value) for value in summary["online_api_calls"] if value is not None
        )
        summary["mean_runtime_cache_miss_rate"] = _mean(
            (
                stats.get("misses", 0) / stats["lookups"]
                if stats.get("lookups")
                else 0.0
            )
            for stats in summary["runtime_cache_stats"]
        )
        summary["mean_runtime_fallback_rate"] = _mean(
            stats.get("fallback_rate") for stats in summary["runtime_cache_stats"]
        )
    return summaries


def evaluate_pilot_gate(
    records: list[dict[str, Any]],
    *,
    required_algorithms: Iterable[str] = REQUIRED_PILOT_ALGORITHMS,
    thresholds: GateThresholds | None = None,
    expert_cache_stats: dict[str, Any] | None = None,
    allow_degenerate_metrics: bool = False,
) -> dict[str, Any]:
    """Evaluate the C6 pilot gate from collected train/eval records."""

    thresholds = thresholds or GateThresholds()
    required = list(required_algorithms)
    summaries = summarize_pilot_records(records)
    present_algorithms = set(summaries)
    required_present = all(algorithm in present_algorithms for algorithm in required)

    all_train_passed = all(
        record.get("train_result", {}).get("status") == "passed" for record in records
    )
    all_eval_passed = all(
        record.get("eval_result", {}).get("status") == "passed" for record in records
    )
    hybrid_action_consistent = all(
        record.get("train_result", {}).get("extra", {}).get("hybrid_action") is True
        for record in records
        if record["algorithm"] in {"maddpg", "maps_no_annealing", "maps"}
    )
    zero_online_calls = all(
        record.get("eval_result", {}).get("extra", {}).get("online_api_calls", 0) == 0
        for record in records
    )
    maps_eval_deployment_without_llm = all(
        record.get("eval_result", {}).get("extra", {}).get("maps_eval_uses_llm")
        is False
        for record in records
        if record["algorithm"] in {"maps", "maps_no_annealing"}
    )

    maps_auc = summaries.get("maps", {}).get("mean_reward_auc")
    maddpg_auc = summaries.get("maddpg", {}).get("mean_reward_auc")
    if maps_auc is None or maddpg_auc is None:
        maps_auc_not_below_maddpg = False
    else:
        maps_auc_not_below_maddpg = (
            maps_auc + thresholds.auc_tolerance >= maddpg_auc
        )

    fixed_auc = summaries.get("maps_no_annealing", {}).get("mean_reward_auc")
    if maps_auc is None or fixed_auc is None:
        fixed_and_annealed_differ = False
    else:
        fixed_and_annealed_differ = (
            abs(maps_auc - fixed_auc) >= thresholds.min_auc_delta
        )

    tcr_values = [
        number
        for summary in summaries.values()
        for value in summary["TCR"]
        if (number := _numeric(value)) is not None
    ]
    dvr_values = [
        number
        for summary in summaries.values()
        for value in summary["DVR"]
        if (number := _numeric(value)) is not None
    ]
    tcr_degenerate = bool(tcr_values) and (
        all(value <= 0.0 for value in tcr_values)
        or all(value >= 1.0 for value in tcr_values)
    )
    dvr_degenerate = bool(dvr_values) and (
        all(value <= 0.0 for value in dvr_values)
        or all(value >= 1.0 for value in dvr_values)
    )
    dvr_tcr_not_degenerate = (
        True if allow_degenerate_metrics else bool(tcr_values and dvr_values)
        and not tcr_degenerate
        and not dvr_degenerate
    )

    cache_stats = (expert_cache_stats or {}).get("stats", {})
    expert_cache_quality = (
        bool(expert_cache_stats and expert_cache_stats.get("available"))
        and _all_present(
            cache_stats,
            ("parser_success_rate", "valid_action_rate", "fallback_rate"),
        )
        and float(cache_stats["parser_success_rate"])
        >= thresholds.min_parser_success_rate
        and float(cache_stats["valid_action_rate"]) >= thresholds.min_valid_action_rate
        and float(cache_stats["fallback_rate"]) <= thresholds.max_cache_fallback_rate
    )

    maps_summaries = [
        summary
        for algorithm, summary in summaries.items()
        if algorithm in {"maps", "maps_no_annealing"}
    ]
    runtime_stats_available = any(
        summary["runtime_cache_stats"] for summary in maps_summaries
    )
    if runtime_stats_available:
        runtime_cache_quality = all(
            (
                summary["mean_runtime_cache_miss_rate"] is not None
                and summary["mean_runtime_cache_miss_rate"]
                <= thresholds.max_runtime_cache_miss_rate
                and summary["mean_runtime_fallback_rate"] is not None
                and summary["mean_runtime_fallback_rate"]
                <= thresholds.max_runtime_fallback_rate
            )
            for summary in maps_summaries
            if summary["runtime_cache_stats"]
        )
    else:
        runtime_cache_quality = False

    checks = {
        "required_algorithms_present": required_present,
        "all_training_runs_passed": all_train_passed,
        "all_evaluation_runs_passed": all_eval_passed,
        "hybrid_action_train_execute_consistent": hybrid_action_consistent,
        "zero_online_api_calls_in_eval": zero_online_calls,
        "maps_eval_deploys_actor_without_llm": maps_eval_deployment_without_llm,
        "expert_cache_offline_quality_high": expert_cache_quality,
        "expert_cache_runtime_coverage_high": runtime_cache_quality,
        "maps_auc_not_below_maddpg": maps_auc_not_below_maddpg,
        "fixed_and_annealed_measurably_differ": fixed_and_annealed_differ,
        "dvr_tcr_not_degenerate": dvr_tcr_not_degenerate,
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "gate_passed": all(checks.values()),
        "checks": checks,
        "thresholds": thresholds.to_dict(),
        "summaries": summaries,
        "expert_cache": expert_cache_stats or {"available": False},
        "diagnostics": {
            "required_algorithms": required,
            "present_algorithms": sorted(present_algorithms),
            "maps_mean_reward_auc": maps_auc,
            "maddpg_mean_reward_auc": maddpg_auc,
            "maps_no_annealing_mean_reward_auc": fixed_auc,
            "tcr_degenerate": tcr_degenerate,
            "dvr_degenerate": dvr_degenerate,
            "runtime_cache_stats_available": runtime_stats_available,
        },
    }


def _override_short_budget(
    config: dict[str, Any],
    algorithm: str,
    *,
    steps: int,
    batch_size: int,
    train_frequency: int,
    update_epochs: int | None,
    evaluation_steps: int,
    drain_steps: int,
    allow_untrained_eval: bool,
) -> dict[str, Any]:
    output = copy.deepcopy(config)
    output.setdefault("training", {})["max_steps_per_episode"] = steps
    output.setdefault("evaluation", {})["window_steps"] = evaluation_steps
    output["evaluation"]["drain_horizon_steps"] = drain_steps
    output["evaluation"]["allow_untrained"] = allow_untrained_eval
    output.setdefault("maddpg", {})["max_steps"] = steps
    output["maddpg"]["batch_size"] = batch_size
    output["maddpg"]["train_frequency"] = train_frequency
    output.setdefault(algorithm, {})["max_steps"] = steps
    output[algorithm]["batch_size"] = batch_size
    output[algorithm]["train_frequency"] = train_frequency
    output.setdefault("llm_maddpg", {})["max_steps"] = steps
    output["llm_maddpg"]["batch_size"] = batch_size
    output["llm_maddpg"]["train_frequency"] = train_frequency
    if update_epochs is not None and algorithm in {"mappo", "happo"}:
        output[algorithm]["update_epochs"] = update_epochs
    return output


def build_pilot_scenario_banks(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
    scenario_id: str,
    train_scenario_seeds: list[int],
    test_scenario_seeds: list[int],
    train_steps: int,
    eval_steps: int,
    construction_seed: int,
) -> dict[str, Any]:
    """Create short frozen train/test banks for the C6 pilot."""

    if set(train_scenario_seeds) & set(test_scenario_seeds):
        raise ValueError("pilot train/test scenario seeds must be disjoint")
    output_dir = resolve_project_path(output_dir)
    train_bank = build_scenario_bank(
        config=config,
        scenario_id=scenario_id,
        split="train",
        seeds=train_scenario_seeds,
        max_steps=train_steps,
        construction_seed=construction_seed,
        description="C6 short pilot training bank",
    )
    test_bank = build_scenario_bank(
        config=config,
        scenario_id=scenario_id,
        split="test",
        seeds=test_scenario_seeds,
        max_steps=eval_steps,
        construction_seed=construction_seed,
        description="C6 short pilot evaluation bank",
    )
    train_path = write_scenario_bank(output_dir / f"{scenario_id}_train.json", train_bank)
    test_path = write_scenario_bank(output_dir / f"{scenario_id}_test.json", test_bank)
    return {
        "train_bank": str(train_path),
        "test_bank": str(test_path),
        "train_bank_id": train_bank["scenario_bank_id"],
        "test_bank_id": test_bank["scenario_bank_id"],
        "train_scenario_count": len(train_bank["scenarios"]),
        "test_scenario_count": len(test_bank["scenarios"]),
    }


def run_pilot_gate(
    *,
    algorithms: list[str],
    seeds: list[int],
    base_config: str | Path,
    scale_config: str | Path,
    deadline_config: str | Path,
    pilot_config: str | Path | None,
    extra_fragments: list[str | Path],
    output_root: str | Path,
    scenario_id: str,
    train_scenario_seeds: list[int],
    test_scenario_seeds: list[int],
    train_steps: int,
    eval_steps: int,
    drain_steps: int,
    batch_size: int,
    train_frequency: int,
    update_epochs: int | None,
    thresholds: GateThresholds,
    allow_degenerate_metrics: bool = False,
    allow_untrained_eval: bool = False,
) -> dict[str, Any]:
    """Run C6 pilot training/evaluation and write the gate report."""

    output_root = resolve_project_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    bank_config = compose_ei_config(
        base_config,
        [scale_config, deadline_config, *extra_fragments],
    )
    bank_info = build_pilot_scenario_banks(
        config=bank_config,
        output_dir=output_root / "scenario_banks",
        scenario_id=scenario_id,
        train_scenario_seeds=train_scenario_seeds,
        test_scenario_seeds=test_scenario_seeds,
        train_steps=train_steps,
        eval_steps=eval_steps,
        construction_seed=min(seeds) if seeds else 0,
    )
    pilot_fragment = [pilot_config] if pilot_config else []
    records: list[dict[str, Any]] = []
    for algorithm in algorithms:
        config = compose_ei_config(
            base_config,
            [
                _method_fragment(algorithm),
                scale_config,
                deadline_config,
                *pilot_fragment,
                *extra_fragments,
            ],
        )
        config = with_bank_paths(
            config,
            train_bank=bank_info["train_bank"],
            test_bank=bank_info["test_bank"],
        )
        config = _override_short_budget(
            config,
            algorithm,
            steps=train_steps,
            batch_size=batch_size,
            train_frequency=train_frequency,
            update_epochs=update_epochs,
            evaluation_steps=eval_steps,
            drain_steps=drain_steps,
            allow_untrained_eval=allow_untrained_eval,
        )
        for seed in seeds:
            train_result = run_baseline(
                algorithm,
                config,
                seed,
                output_root / "training",
                command="experiments.ei.pilot_gate",
            )
            eval_config = copy.deepcopy(config)
            eval_config.setdefault("evaluation", {})["checkpoint_dir"] = train_result[
                "run_dir"
            ]
            eval_result = run_evaluation(
                algorithm,
                eval_config,
                seed,
                output_root / "evaluation",
                command="experiments.ei.pilot_gate",
            )
            records.append(
                {
                    "algorithm": algorithm,
                    "seed": int(seed),
                    "train_result": train_result,
                    "eval_result": eval_result,
                }
            )

    expert_cache_stats = _load_expert_cache_stats(bank_config)
    gate = evaluate_pilot_gate(
        records,
        required_algorithms=REQUIRED_PILOT_ALGORITHMS,
        thresholds=thresholds,
        expert_cache_stats=expert_cache_stats,
        allow_degenerate_metrics=allow_degenerate_metrics,
    )
    report = {
        **gate,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "scenario_banks": bank_info,
        "config": {
            "base_config": str(resolve_project_path(base_config)),
            "scale_config": str(resolve_project_path(scale_config)),
            "deadline_config": str(resolve_project_path(deadline_config)),
            "pilot_config": str(resolve_project_path(pilot_config))
            if pilot_config
            else None,
            "extra_fragments": [str(resolve_project_path(item)) for item in extra_fragments],
            "algorithms": algorithms,
            "seeds": seeds,
            "train_scenario_seeds": train_scenario_seeds,
            "test_scenario_seeds": test_scenario_seeds,
            "train_steps": train_steps,
            "eval_steps": eval_steps,
            "drain_steps": drain_steps,
            "batch_size": batch_size,
            "train_frequency": train_frequency,
            "update_epochs": update_epochs,
            "allow_degenerate_metrics": allow_degenerate_metrics,
            "allow_untrained_eval": allow_untrained_eval,
        },
        "records": records,
        "resolved_bank_config": sanitize_config(bank_config),
    }
    report_path = output_root / "pilot_gate.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    pilot_defaults = _load_yaml(DEFAULT_PILOT_CONFIG).get("pilot_gate", {})
    thresholds = GateThresholds.from_mapping(pilot_defaults.get("thresholds"))
    parser = argparse.ArgumentParser(description="Run C6 EI two-seed pilot gate")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--scale-config", default=DEFAULT_SCALE_CONFIG)
    parser.add_argument("--deadline-config", default=DEFAULT_DEADLINE_CONFIG)
    parser.add_argument("--pilot-config", default=DEFAULT_PILOT_CONFIG)
    parser.add_argument("--fragment", action="append", default=[])
    parser.add_argument(
        "--algorithms",
        default=",".join(pilot_defaults.get("algorithms", REQUIRED_PILOT_ALGORITHMS)),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(item) for item in pilot_defaults.get("seeds", [42, 43])),
    )
    parser.add_argument(
        "--train-scenario-seeds",
        default=",".join(
            str(item)
            for item in pilot_defaults.get("train_scenario_seeds", [1000, 1001])
        ),
    )
    parser.add_argument(
        "--test-scenario-seeds",
        default=",".join(
            str(item)
            for item in pilot_defaults.get("test_scenario_seeds", [2000, 2001])
        ),
    )
    parser.add_argument("--scenario-id", default=pilot_defaults.get("scenario_id", "c6_s1_u10_medium_pilot"))
    parser.add_argument("--train-steps", type=int, default=int(pilot_defaults.get("train_steps", 20)))
    parser.add_argument("--eval-steps", type=int, default=int(pilot_defaults.get("eval_steps", 20)))
    parser.add_argument("--drain-steps", type=int, default=int(pilot_defaults.get("drain_steps", 5)))
    parser.add_argument("--batch-size", type=int, default=int(pilot_defaults.get("batch_size", 4)))
    parser.add_argument(
        "--train-frequency",
        type=int,
        default=int(pilot_defaults.get("train_frequency", 1)),
    )
    parser.add_argument("--update-epochs", type=int, default=pilot_defaults.get("update_epochs", 1))
    parser.add_argument("--output-root", default=pilot_defaults.get("output_root", "artifacts/ei/pilot_gate"))
    parser.add_argument("--auc-tolerance", type=float, default=thresholds.auc_tolerance)
    parser.add_argument("--min-auc-delta", type=float, default=thresholds.min_auc_delta)
    parser.add_argument(
        "--min-parser-success-rate",
        type=float,
        default=thresholds.min_parser_success_rate,
    )
    parser.add_argument(
        "--min-valid-action-rate",
        type=float,
        default=thresholds.min_valid_action_rate,
    )
    parser.add_argument(
        "--max-cache-fallback-rate",
        type=float,
        default=thresholds.max_cache_fallback_rate,
    )
    parser.add_argument(
        "--max-runtime-cache-miss-rate",
        type=float,
        default=thresholds.max_runtime_cache_miss_rate,
    )
    parser.add_argument(
        "--max-runtime-fallback-rate",
        type=float,
        default=thresholds.max_runtime_fallback_rate,
    )
    parser.add_argument("--allow-degenerate-metrics", action="store_true")
    parser.add_argument("--allow-untrained-eval", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    thresholds = GateThresholds(
        auc_tolerance=args.auc_tolerance,
        min_auc_delta=args.min_auc_delta,
        min_parser_success_rate=args.min_parser_success_rate,
        min_valid_action_rate=args.min_valid_action_rate,
        max_cache_fallback_rate=args.max_cache_fallback_rate,
        max_runtime_cache_miss_rate=args.max_runtime_cache_miss_rate,
        max_runtime_fallback_rate=args.max_runtime_fallback_rate,
    )
    report = run_pilot_gate(
        algorithms=parse_csv_strings(args.algorithms),
        seeds=parse_csv_ints(args.seeds),
        base_config=args.base_config,
        scale_config=args.scale_config,
        deadline_config=args.deadline_config,
        pilot_config=args.pilot_config,
        extra_fragments=args.fragment,
        output_root=args.output_root,
        scenario_id=args.scenario_id,
        train_scenario_seeds=parse_csv_ints(args.train_scenario_seeds),
        test_scenario_seeds=parse_csv_ints(args.test_scenario_seeds),
        train_steps=args.train_steps,
        eval_steps=args.eval_steps,
        drain_steps=args.drain_steps,
        batch_size=args.batch_size,
        train_frequency=args.train_frequency,
        update_epochs=args.update_epochs,
        thresholds=thresholds,
        allow_degenerate_metrics=args.allow_degenerate_metrics,
        allow_untrained_eval=args.allow_untrained_eval,
    )
    print(
        json.dumps(
            {
                "gate_passed": report["gate_passed"],
                "checks": report["checks"],
                "report_path": report["report_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if args.fail_on_gate and not report["gate_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
