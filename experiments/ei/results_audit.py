"""C8 result audit and statistical analysis for the EI formal matrix.

The C7 analysis step aggregates one row per evaluation run.  This module builds
the next layer: artifact checks, seed-level descriptive statistics, paired
MAPS contrasts, conservative claim candidates, and real SVG figures.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PACKAGE_PARENT = Path(__file__).resolve().parents[3]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from LLM4RL.experiments.ei.config import resolve_project_path  # noqa: E402


SCHEMA_VERSION = "ei-results-audit-v1"

METRIC_KEYS = [
    "N_gen",
    "N_done",
    "N_on_time",
    "TCR",
    "DVR",
    "p95_task_latency",
    "mean_latency",
    "system_energy_per_completed_task",
    "device_energy_per_completed_task",
    "mean_decision_latency_ms",
    "p95_decision_latency_ms",
    "evaluation_reward",
]

COMPARISON_METRICS = [
    "TCR",
    "DVR",
    "p95_task_latency",
    "mean_latency",
    "system_energy_per_completed_task",
    "device_energy_per_completed_task",
    "mean_decision_latency_ms",
    "p95_decision_latency_ms",
    "evaluation_reward",
]

PRIMARY_METRICS = [
    "TCR",
    "DVR",
    "p95_task_latency",
    "system_energy_per_completed_task",
]

METRIC_DIRECTIONS = {
    "N_gen": "higher",
    "N_done": "higher",
    "N_on_time": "higher",
    "TCR": "higher",
    "DVR": "lower",
    "p95_task_latency": "lower",
    "mean_latency": "lower",
    "system_energy_per_completed_task": "lower",
    "device_energy_per_completed_task": "lower",
    "mean_decision_latency_ms": "lower",
    "p95_decision_latency_ms": "lower",
    "evaluation_reward": "higher",
}

LEARNING_METHODS = {"maddpg", "mappo", "maps_no_annealing", "maps"}
MAPS_METHODS = {"maps", "maps_no_annealing"}
BASELINE_METHODS = {"greedy_min_cost", "llm_only"}

METHOD_ORDER = [
    "greedy_min_cost",
    "llm_only",
    "maddpg",
    "mappo",
    "maps_no_annealing",
    "maps",
]

METHOD_LABELS = {
    "greedy_min_cost": "Greedy-MinCost",
    "llm_only": "LLM-only",
    "maddpg": "MADDPG",
    "mappo": "MAPPO",
    "maps_no_annealing": "MAPS-w/o-Annealing",
    "maps": "MAPS",
}

METHOD_COLORS = {
    "greedy_min_cost": "#000000",
    "llm_only": "#E69F00",
    "maddpg": "#56B4E9",
    "mappo": "#009E73",
    "maps_no_annealing": "#CC79A7",
    "maps": "#0072B2",
}

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    experiment: str
    scenario: str
    algorithm: str
    seed: int | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "experiment": self.experiment,
            "scenario": self.scenario,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "message": self.message,
        }


def _path_parts(path_value: str) -> list[str]:
    normalized = str(path_value).replace("\\", "/")
    return [part for part in normalized.split("/") if part]


def infer_experiment_and_scenario(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("experiment") and row.get("scenario"):
        return str(row["experiment"]), str(row["scenario"])
    parts = _path_parts(str(row.get("run_dir", "")))
    if "evaluation" in parts:
        index = parts.index("evaluation")
        if index + 2 < len(parts):
            return parts[index + 1], parts[index + 2]
    return "unknown", "unknown"


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _read_json(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _nested(payload: dict[str, Any] | None, keys: Iterable[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    return T_CRITICAL_95.get(df, 1.96)


def _ci95(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    mean = _mean(values)
    std = _sample_std(values)
    if mean is None or std is None:
        return None, None
    margin = _t_critical_95(len(values) - 1) * std / math.sqrt(len(values))
    return mean - margin, mean + margin


def _exact_two_sided_sign_p(positive: int, negative: int) -> float | None:
    n = positive + negative
    if n == 0:
        return None
    tail = min(positive, negative)
    probability = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, probability)


def _directional_improvement(metric: str, maps_value: float, other_value: float) -> float:
    direction = METRIC_DIRECTIONS.get(metric, "higher")
    if direction == "higher":
        return maps_value - other_value
    return other_value - maps_value


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_float(row.get(field)) for field in fieldnames})


def load_evaluation_rows(input_csv: str | Path) -> list[dict[str, Any]]:
    path = resolve_project_path(input_csv)
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        experiment, scenario = infer_experiment_and_scenario(raw)
        metrics_payload = _read_json(raw.get("evaluation_metrics_path"))
        manifest = _read_json(raw.get("run_manifest_path"))
        scenarios = []
        if isinstance(metrics_payload, dict):
            scenarios = list(metrics_payload.get("scenarios") or [])
        row: dict[str, Any] = {
            **raw,
            "experiment": experiment,
            "scenario": scenario,
            "algorithm": str(raw.get("algorithm", "")),
            "seed": _int_or_none(raw.get("seed")),
            "required_artifacts_present": _bool_value(
                raw.get("required_artifacts_present")
            ),
            "_metrics_payload": metrics_payload,
            "_manifest": manifest,
            "_scenarios": scenarios,
        }
        for metric in METRIC_KEYS:
            row[metric] = _float_or_none(raw.get(metric))
        rows.append(row)
    return rows


def _group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["experiment"]),
        str(row["scenario"]),
        str(row["algorithm"]),
    )


def _metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    return [value for value in (_float_or_none(row.get(metric)) for row in rows) if value is not None]


def build_descriptive_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)

    output: list[dict[str, Any]] = []
    for (experiment, scenario, algorithm), items in sorted(grouped.items()):
        seeds = sorted({item["seed"] for item in items if item.get("seed") is not None})
        unit = "training_seed" if algorithm in LEARNING_METHODS else "single_eval_run"
        for metric in METRIC_KEYS:
            values = _metric_values(items, metric)
            mean = _mean(values)
            std = _sample_std(values)
            ci_low, ci_high = _ci95(values)
            output.append(
                {
                    "experiment": experiment,
                    "scenario": scenario,
                    "algorithm": algorithm,
                    "method_label": METHOD_LABELS.get(algorithm, algorithm),
                    "metric": metric,
                    "direction": METRIC_DIRECTIONS.get(metric, "higher"),
                    "unit_of_analysis": unit,
                    "n": len(values),
                    "seeds": ",".join(str(seed) for seed in seeds),
                    "mean": mean,
                    "std": std,
                    "se": std / math.sqrt(len(values)) if std is not None else None,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "inference_status": "ok" if len(values) >= 2 else "blocked_single_run",
                }
            )
    return output


def build_scenario_descriptive_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        scenarios = row.get("_scenarios") or []
        if not scenarios:
            continue
        for metric in COMPARISON_METRICS:
            values = _metric_values(scenarios, metric)
            mean = _mean(values)
            std = _sample_std(values)
            ci_low, ci_high = _ci95(values)
            output.append(
                {
                    "experiment": row["experiment"],
                    "scenario": row["scenario"],
                    "algorithm": row["algorithm"],
                    "seed": row["seed"],
                    "metric": metric,
                    "direction": METRIC_DIRECTIONS.get(metric, "higher"),
                    "unit_of_analysis": "test_scenario_within_run",
                    "n": len(values),
                    "mean": mean,
                    "std": std,
                    "se": std / math.sqrt(len(values)) if std is not None else None,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "inference_status": (
                        "descriptive_scenario_ci_not_training_seed_inference"
                        if len(values) >= 2
                        else "blocked_single_scenario"
                    ),
                }
            )
    return output


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("sign_test_p") is not None:
            grouped[str(row["multiple_comparison_family"])].append(row)

    for items in grouped.values():
        ordered = sorted(items, key=lambda item: float(item["sign_test_p"]))
        m = len(ordered)
        running = 0.0
        for index, item in enumerate(ordered):
            adjusted = min(1.0, (m - index) * float(item["sign_test_p"]))
            running = max(running, adjusted)
            item["holm_p"] = running


def build_paired_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, str], dict[str, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    counts_by_algorithm: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (str(row["experiment"]), str(row["scenario"]))
        algorithm = str(row["algorithm"])
        seed = row.get("seed")
        counts_by_algorithm[key][algorithm] += 1
        if seed is not None:
            by_group[key][algorithm][int(seed)] = row

    output: list[dict[str, Any]] = []
    for (experiment, scenario), algorithms in sorted(by_group.items()):
        if "maps" not in algorithms:
            continue
        comparators = [
            algorithm
            for algorithm in METHOD_ORDER
            if algorithm != "maps" and algorithm in algorithms
        ]
        for comparator in comparators:
            map_seeds = set(algorithms["maps"])
            comparator_seeds = set(algorithms[comparator])
            paired_seeds = sorted(map_seeds & comparator_seeds)
            baseline_blocked = comparator in BASELINE_METHODS
            for metric in COMPARISON_METRICS:
                family = f"{experiment}:{scenario}:{metric}:maps-vs-comparators"
                row: dict[str, Any] = {
                    "experiment": experiment,
                    "scenario": scenario,
                    "method_a": "maps",
                    "method_b": comparator,
                    "contrast_label": f"MAPS vs {METHOD_LABELS.get(comparator, comparator)}",
                    "metric": metric,
                    "direction": METRIC_DIRECTIONS.get(metric, "higher"),
                    "unit_of_analysis": "paired_training_seed",
                    "n_pairs": len(paired_seeds),
                    "paired_seeds": ",".join(str(seed) for seed in paired_seeds),
                    "method_a_n": counts_by_algorithm[(experiment, scenario)]["maps"],
                    "method_b_n": counts_by_algorithm[(experiment, scenario)][comparator],
                    "multiple_comparison_family": family,
                    "test_name": "paired seed delta + exact two-sided sign test",
                    "status": "valid",
                    "blocker": "",
                    "mean_delta_raw": None,
                    "ci95_delta_raw_low": None,
                    "ci95_delta_raw_high": None,
                    "mean_improvement": None,
                    "ci95_improvement_low": None,
                    "ci95_improvement_high": None,
                    "cohens_dz_improvement": None,
                    "maps_better_count": None,
                    "ties": None,
                    "sign_test_p": None,
                    "holm_p": None,
                }
                if baseline_blocked:
                    row["status"] = "blocked_single_run_baseline"
                    row["unit_of_analysis"] = "not_tested"
                    row["blocker"] = (
                        "Comparator is a single-seed baseline; report descriptive "
                        "difference only, not seed-level significance."
                    )
                    output.append(row)
                    continue
                if len(paired_seeds) < 2:
                    row["status"] = "blocked_insufficient_paired_seeds"
                    row["blocker"] = "At least two paired seeds are required."
                    output.append(row)
                    continue

                raw_deltas: list[float] = []
                improvements: list[float] = []
                for seed in paired_seeds:
                    maps_value = _float_or_none(algorithms["maps"][seed].get(metric))
                    other_value = _float_or_none(algorithms[comparator][seed].get(metric))
                    if maps_value is None or other_value is None:
                        continue
                    raw_deltas.append(maps_value - other_value)
                    improvements.append(
                        _directional_improvement(metric, maps_value, other_value)
                    )
                if len(improvements) < 2:
                    row["status"] = "blocked_missing_metric_values"
                    row["blocker"] = "Fewer than two paired finite metric values."
                    output.append(row)
                    continue

                delta_ci = _ci95(raw_deltas)
                improvement_ci = _ci95(improvements)
                improvement_std = _sample_std(improvements)
                positives = sum(1 for value in improvements if value > 0.0)
                negatives = sum(1 for value in improvements if value < 0.0)
                row.update(
                    {
                        "n_pairs": len(improvements),
                        "mean_delta_raw": _mean(raw_deltas),
                        "ci95_delta_raw_low": delta_ci[0],
                        "ci95_delta_raw_high": delta_ci[1],
                        "mean_improvement": _mean(improvements),
                        "ci95_improvement_low": improvement_ci[0],
                        "ci95_improvement_high": improvement_ci[1],
                        "cohens_dz_improvement": (
                            _mean(improvements) / improvement_std
                            if improvement_std and improvement_std > 0.0
                            else None
                        ),
                        "maps_better_count": positives,
                        "ties": len(improvements) - positives - negatives,
                        "sign_test_p": _exact_two_sided_sign_p(positives, negatives),
                    }
                )
                output.append(row)
    _holm_adjust(output)
    return output


def audit_artifacts(
    rows: list[dict[str, Any]],
    *,
    expected_learning_seeds: int | None = 5,
    expected_baseline_seeds: int | None = 1,
) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    path_fields = [
        "evaluation_metrics_path",
        "task_records_path",
        "run_manifest_path",
        "resolved_config_path",
    ]
    for row in rows:
        experiment = str(row["experiment"])
        scenario = str(row["scenario"])
        algorithm = str(row["algorithm"])
        seed = row.get("seed")
        if not row.get("required_artifacts_present"):
            issues.append(
                AuditIssue(
                    "error",
                    "required_artifacts_missing",
                    experiment,
                    scenario,
                    algorithm,
                    seed,
                    "C7 summary reports missing required artifacts.",
                )
            )
        for field in path_fields:
            path_value = row.get(field)
            if not path_value or not Path(path_value).exists():
                issues.append(
                    AuditIssue(
                        "error",
                        "artifact_path_missing",
                        experiment,
                        scenario,
                        algorithm,
                        seed,
                        f"{field} does not exist: {path_value}",
                    )
                )

        manifest = row.get("_manifest")
        if isinstance(manifest, dict):
            status = manifest.get("status") or _nested(manifest, ("summary", "status"))
            if status != "passed":
                issues.append(
                    AuditIssue(
                        "error",
                        "manifest_not_passed",
                        experiment,
                        scenario,
                        algorithm,
                        seed,
                        f"run_manifest status is {status!r}, expected 'passed'.",
                    )
                )
            extra = _nested(manifest, ("summary", "extra")) or {}
            online_calls = _int_or_none(extra.get("online_api_calls"))
            if online_calls not in (None, 0):
                issues.append(
                    AuditIssue(
                        "error",
                        "online_api_calls_in_evaluation",
                        experiment,
                        scenario,
                        algorithm,
                        seed,
                        f"evaluation recorded online_api_calls={online_calls}.",
                    )
                )
            if algorithm == "llm_only":
                cache_stats = extra.get("cache_runtime_stats") or _nested(
                    extra, ("expert", "runtime_stats")
                )
                if not cache_stats:
                    issues.append(
                        AuditIssue(
                            "warning",
                            "llm_cache_runtime_stats_missing",
                            experiment,
                            scenario,
                            algorithm,
                            seed,
                            "LLM-only run has no cache runtime stats in manifest.",
                        )
                    )
                else:
                    misses = _int_or_none(cache_stats.get("misses"))
                    fallback_actions = _int_or_none(cache_stats.get("fallback_actions"))
                    if misses not in (None, 0) or fallback_actions not in (None, 0):
                        issues.append(
                            AuditIssue(
                                "error",
                                "llm_cache_coverage_incomplete",
                                experiment,
                                scenario,
                                algorithm,
                                seed,
                                (
                                    "LLM-only evaluation had cache misses or fallback "
                                    f"actions: misses={misses}, fallback={fallback_actions}."
                                ),
                            )
                        )
                if extra.get("cache_coverage_complete") is False:
                    issues.append(
                        AuditIssue(
                            "error",
                            "llm_cache_coverage_flag_false",
                            experiment,
                            scenario,
                            algorithm,
                            seed,
                            "LLM-only manifest reports cache_coverage_complete=false.",
                        )
                    )
            elif extra.get("llm_enabled") is True:
                issues.append(
                    AuditIssue(
                        "error",
                        "non_llm_evaluation_enabled_llm",
                        experiment,
                        scenario,
                        algorithm,
                        seed,
                        "A non-LLM-only evaluation reported llm_enabled=true.",
                    )
                )
            if algorithm in MAPS_METHODS and extra.get("maps_eval_uses_llm") is True:
                issues.append(
                    AuditIssue(
                        "error",
                        "maps_eval_uses_llm",
                        experiment,
                        scenario,
                        algorithm,
                        seed,
                        "MAPS evaluation must deploy actor-only without LLM calls.",
                    )
                )

        metrics_payload = row.get("_metrics_payload")
        if isinstance(metrics_payload, dict):
            scenarios = metrics_payload.get("scenarios") or []
            expected_scenario_count = _int_or_none(row.get("scenario_count"))
            if expected_scenario_count is not None and len(scenarios) != expected_scenario_count:
                issues.append(
                    AuditIssue(
                        "error",
                        "scenario_count_mismatch",
                        experiment,
                        scenario,
                        algorithm,
                        seed,
                        (
                            f"evaluation_metrics has {len(scenarios)} scenarios, "
                            f"summary says {expected_scenario_count}."
                        ),
                    )
                )
            metrics = metrics_payload.get("metrics") or {}
            for metric in METRIC_KEYS:
                summary_value = _float_or_none(row.get(metric))
                payload_value = _float_or_none(metrics.get(metric))
                if summary_value is None:
                    issues.append(
                        AuditIssue(
                            "error",
                            "summary_metric_missing",
                            experiment,
                            scenario,
                            algorithm,
                            seed,
                            f"{metric} is missing or non-finite in evaluation summary.",
                        )
                    )
                elif payload_value is not None and abs(summary_value - payload_value) > 1e-9:
                    issues.append(
                        AuditIssue(
                            "error",
                            "summary_metric_mismatch",
                            experiment,
                            scenario,
                            algorithm,
                            seed,
                            (
                                f"{metric} differs between summary "
                                f"({summary_value}) and metrics payload ({payload_value})."
                            ),
                        )
                    )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)
    for (experiment, scenario, algorithm), items in grouped.items():
        seeds = {item.get("seed") for item in items}
        expected = None
        if algorithm in LEARNING_METHODS:
            expected = expected_learning_seeds
        elif algorithm in BASELINE_METHODS:
            expected = expected_baseline_seeds
        if expected is not None and len(seeds) != expected:
            issues.append(
                AuditIssue(
                    "warning",
                    "unexpected_seed_count",
                    experiment,
                    scenario,
                    algorithm,
                    None,
                    (
                        f"Observed {len(seeds)} seed(s), expected {expected}. "
                        "This affects inference scope."
                    ),
                )
            )
    return issues


def _lookup_descriptive(
    descriptive_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (
            str(row["experiment"]),
            str(row["scenario"]),
            str(row["algorithm"]),
            str(row["metric"]),
        ): row
        for row in descriptive_rows
    }


def _available_methods(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    experiment: str,
    scenario: str,
    metric: str,
) -> list[str]:
    return [
        algorithm
        for algorithm in METHOD_ORDER
        if (experiment, scenario, algorithm, metric) in lookup
    ]


def _metric_extent(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    keys: list[tuple[str, str, str, str]],
) -> tuple[float, float]:
    values: list[float] = []
    for key in keys:
        row = lookup.get(key)
        if not row:
            continue
        for field in ("mean", "ci95_low", "ci95_high"):
            value = _float_or_none(row.get(field))
            if value is not None:
                values.append(value)
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if low >= 0.0:
        low = 0.0
    if math.isclose(low, high):
        high = low + 1.0
    padding = (high - low) * 0.08
    return low, high + padding


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 11,
    anchor: str = "middle",
    rotate: int | None = None,
    weight: str | None = None,
) -> str:
    attrs = [
        f'x="{x:.1f}"',
        f'y="{y:.1f}"',
        f'font-size="{size}"',
        f'text-anchor="{anchor}"',
        'font-family="Arial, sans-serif"',
    ]
    if rotate is not None:
        attrs.append(f'transform="rotate({rotate} {x:.1f} {y:.1f})"')
    if weight:
        attrs.append(f'font-weight="{weight}"')
    return f"<text {' '.join(attrs)}>{html.escape(text)}</text>"


def _draw_axes(x: float, y: float, width: float, height: float) -> list[str]:
    return [
        f'<line x1="{x:.1f}" y1="{y + height:.1f}" x2="{x + width:.1f}" y2="{y + height:.1f}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y + height:.1f}" stroke="#333" stroke-width="1"/>',
    ]


def _bar_panel(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    experiment: str,
    scenario: str,
    metric: str,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
) -> list[str]:
    methods = _available_methods(lookup, experiment, scenario, metric)
    keys = [(experiment, scenario, method, metric) for method in methods]
    low, high = _metric_extent(lookup, keys)

    def y_coord(value: float) -> float:
        return y + height - ((value - low) / (high - low)) * height

    elements = [_svg_text(x + width / 2, y - 18, title, size=13, weight="bold")]
    elements.extend(_draw_axes(x, y, width, height))
    tick_count = 4
    for tick in range(tick_count + 1):
        value = low + (high - low) * tick / tick_count
        yy = y_coord(value)
        elements.append(
            f'<line x1="{x - 4:.1f}" y1="{yy:.1f}" x2="{x + width:.1f}" y2="{yy:.1f}" stroke="#dddddd" stroke-width="1"/>'
        )
        elements.append(_svg_text(x - 8, yy + 4, f"{value:.2g}", size=9, anchor="end"))

    if not methods:
        elements.append(_svg_text(x + width / 2, y + height / 2, "No data", size=12))
        return elements

    slot = width / len(methods)
    bar_width = min(36.0, slot * 0.62)
    zero = y_coord(0.0)
    for index, method in enumerate(methods):
        row = lookup[(experiment, scenario, method, metric)]
        mean = _float_or_none(row.get("mean"))
        if mean is None:
            continue
        cx = x + slot * index + slot / 2
        top = y_coord(mean)
        bar_y = min(top, zero)
        bar_h = abs(zero - top)
        color = METHOD_COLORS.get(method, "#999999")
        elements.append(
            f'<rect x="{cx - bar_width / 2:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" fill="{color}" opacity="0.9"/>'
        )
        ci_low = _float_or_none(row.get("ci95_low"))
        ci_high = _float_or_none(row.get("ci95_high"))
        if ci_low is not None and ci_high is not None:
            y_low = y_coord(ci_low)
            y_high = y_coord(ci_high)
            elements.extend(
                [
                    f'<line x1="{cx:.1f}" y1="{y_low:.1f}" x2="{cx:.1f}" y2="{y_high:.1f}" stroke="#222" stroke-width="1.2"/>',
                    f'<line x1="{cx - 5:.1f}" y1="{y_low:.1f}" x2="{cx + 5:.1f}" y2="{y_low:.1f}" stroke="#222" stroke-width="1.2"/>',
                    f'<line x1="{cx - 5:.1f}" y1="{y_high:.1f}" x2="{cx + 5:.1f}" y2="{y_high:.1f}" stroke="#222" stroke-width="1.2"/>',
                ]
            )
        elements.append(
            _svg_text(
                cx,
                y + height + 16,
                METHOD_LABELS.get(method, method),
                size=8,
                rotate=-35,
            )
        )
    return elements


def _best_metric_label(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    experiment: str,
    scenario: str,
    metric: str,
) -> str:
    candidates = []
    for algorithm in _available_methods(lookup, experiment, scenario, metric):
        row = lookup.get((experiment, scenario, algorithm, metric))
        value = _float_or_none(row.get("mean")) if row else None
        if value is not None:
            candidates.append((algorithm, value))
    if not candidates:
        return "No data available."
    reverse = METRIC_DIRECTIONS.get(metric, "higher") == "higher"
    algorithm, value = sorted(candidates, key=lambda item: item[1], reverse=reverse)[0]
    return f"{METHOD_LABELS.get(algorithm, algorithm)} ({value:.4g})"


def _mean_lookup(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    experiment: str,
    scenario: str,
    algorithm: str,
    metric: str,
) -> float | None:
    row = lookup.get((experiment, scenario, algorithm, metric))
    return _float_or_none(row.get("mean")) if row else None


def _scenario_sort_key(scenario: str) -> tuple[int, str]:
    ue_match = re.search(r"u(\d+)", scenario)
    if ue_match:
        return int(ue_match.group(1)), scenario
    deadline_order = {"loose": 0, "medium": 1, "strict": 2}
    for label, order in deadline_order.items():
        if label in scenario:
            return order, scenario
    return 999, scenario


def _deadline_sort_key(scenario: str) -> tuple[int, str]:
    deadline_order = {"loose": 0, "medium": 1, "strict": 2}
    for label, order in deadline_order.items():
        if label in scenario:
            return order, scenario
    return 999, scenario


def _line_panel(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    experiment: str,
    scenarios: list[str],
    algorithms: list[str],
    metric: str,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    xlabel: str,
    label_mode: str = "auto",
) -> list[str]:
    keys = [
        (experiment, scenario, algorithm, metric)
        for scenario in scenarios
        for algorithm in algorithms
    ]
    low, high = _metric_extent(lookup, keys)

    def x_coord(index: int) -> float:
        if len(scenarios) == 1:
            return x + width / 2
        return x + width * index / (len(scenarios) - 1)

    def y_coord(value: float) -> float:
        return y + height - ((value - low) / (high - low)) * height

    elements = [_svg_text(x + width / 2, y - 18, title, size=13, weight="bold")]
    elements.extend(_draw_axes(x, y, width, height))
    for tick in range(5):
        value = low + (high - low) * tick / 4
        yy = y_coord(value)
        elements.append(
            f'<line x1="{x - 4:.1f}" y1="{yy:.1f}" x2="{x + width:.1f}" y2="{yy:.1f}" stroke="#dddddd" stroke-width="1"/>'
        )
        elements.append(_svg_text(x - 8, yy + 4, f"{value:.2g}", size=9, anchor="end"))

    for index, scenario in enumerate(scenarios):
        xx = x_coord(index)
        elements.append(
            f'<line x1="{xx:.1f}" y1="{y + height:.1f}" x2="{xx:.1f}" y2="{y + height + 4:.1f}" stroke="#333"/>'
        )
        ue_label = re.search(r"u(\d+)", scenario)
        if label_mode == "deadline":
            text = next(
                (label for label in ("loose", "medium", "strict") if label in scenario),
                scenario,
            )
        elif label_mode == "ue" and ue_label:
            text = f"U={ue_label.group(1)}"
        elif ue_label:
            text = f"U={ue_label.group(1)}"
        else:
            text = scenario.replace("s3_u10_", "")
        elements.append(_svg_text(xx, y + height + 18, text, size=10))

    for algorithm in algorithms:
        points: list[tuple[float, float]] = []
        for index, scenario in enumerate(scenarios):
            row = lookup.get((experiment, scenario, algorithm, metric))
            value = _float_or_none(row.get("mean")) if row else None
            if value is not None:
                points.append((x_coord(index), y_coord(value)))
        if not points:
            continue
        color = METHOD_COLORS.get(algorithm, "#999999")
        if len(points) >= 2:
            path_data = " ".join(
                f"{'M' if idx == 0 else 'L'} {px:.1f} {py:.1f}"
                for idx, (px, py) in enumerate(points)
            )
            elements.append(
                f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
        for px, py in points:
            elements.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{color}"/>')

    legend_x = x + width + 18
    legend_y = y + 8
    for idx, algorithm in enumerate(algorithms):
        yy = legend_y + idx * 18
        color = METHOD_COLORS.get(algorithm, "#999999")
        elements.append(
            f'<line x1="{legend_x:.1f}" y1="{yy:.1f}" x2="{legend_x + 16:.1f}" y2="{yy:.1f}" stroke="{color}" stroke-width="2"/>'
        )
        elements.append(
            _svg_text(
                legend_x + 22,
                yy + 4,
                METHOD_LABELS.get(algorithm, algorithm),
                size=10,
                anchor="start",
            )
        )
    elements.append(_svg_text(x + width / 2, y + height + 40, xlabel, size=11))
    return elements


def write_figures(
    output_dir: Path,
    descriptive_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    lookup = _lookup_descriptive(descriptive_rows)
    figures: list[dict[str, str]] = []

    width = 980
    height = 430
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 28, "E2 independent-test main metrics", size=16, weight="bold"),
    ]
    elements.extend(
        _bar_panel(
            lookup,
            experiment="e2",
            scenario="s1_u10_medium",
            metric="p95_task_latency",
            x=70,
            y=70,
            width=380,
            height=245,
            title="P95 task latency (lower is better)",
        )
    )
    elements.extend(
        _bar_panel(
            lookup,
            experiment="e2",
            scenario="s1_u10_medium",
            metric="DVR",
            x=545,
            y=70,
            width=380,
            height=245,
            title="Deadline violation ratio (lower is better)",
        )
    )
    elements.append(
        _svg_text(
            width / 2,
            400,
            "Bars show run means; error bars show 95% CI when repeated seeds exist.",
            size=11,
        )
    )
    elements.append("</svg>")
    path = figures_dir / "figure-01-e2-main-comparison.svg"
    path.write_text("\n".join(elements), encoding="utf-8")
    figures.append(
        {
            "filename": str(path),
            "purpose": "Compare E2 independent-test latency and deadline reliability.",
            "metric": "p95_task_latency,DVR",
            "key_observation": (
                "Lowest E2 P95 latency: "
                + _best_metric_label(
                    lookup,
                    experiment="e2",
                    scenario="s1_u10_medium",
                    metric="p95_task_latency",
                )
                + "; lowest E2 DVR: "
                + _best_metric_label(
                    lookup,
                    experiment="e2",
                    scenario="s1_u10_medium",
                    metric="DVR",
                )
                + "."
            ),
            "interpretation": (
                "Use the paired seed table before turning the visual ranking into a "
                "performance claim; single-seed baselines are descriptive only."
            ),
        }
    )

    e3_scenarios = sorted(
        {
            scenario
            for experiment, scenario, _algorithm, metric in lookup
            if experiment == "e3" and metric == "TCR"
        },
        key=_scenario_sort_key,
    )
    e3_algorithms = [
        algorithm
        for algorithm in ["greedy_min_cost", "maddpg", "mappo", "maps"]
        if any(("e3", scenario, algorithm, "TCR") in lookup for scenario in e3_scenarios)
    ]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 28, "E3 UE-scale stress trend", size=16, weight="bold"),
    ]
    elements.extend(
        _line_panel(
            lookup,
            experiment="e3",
            scenarios=e3_scenarios,
            algorithms=e3_algorithms,
            metric="TCR",
            x=80,
            y=70,
            width=690,
            height=250,
            title="Task completion ratio across UE counts",
            xlabel="UE count",
            label_mode="ue",
        )
    )
    elements.append(
        _svg_text(
            width / 2,
            400,
            "Lines connect per-method seed means; baseline is descriptive because it has one evaluation seed.",
            size=11,
        )
    )
    elements.append("</svg>")
    path = figures_dir / "figure-02-e3-scale-trend.svg"
    path.write_text("\n".join(elements), encoding="utf-8")
    figures.append(
        {
            "filename": str(path),
            "purpose": "Show how completion ratio changes as UE count grows.",
            "metric": "TCR",
            "key_observation": (
                "MAPS TCR changes from "
                + _format_float(
                    _mean_lookup(
                        lookup,
                        "e3",
                        e3_scenarios[0],
                        "maps",
                        "TCR",
                    )
                    if e3_scenarios
                    else None
                )
                + " at the smallest plotted UE count to "
                + _format_float(
                    _mean_lookup(
                        lookup,
                        "e3",
                        e3_scenarios[-1],
                        "maps",
                        "TCR",
                    )
                    if e3_scenarios
                    else None
                )
                + " at the largest plotted UE count."
            ),
            "interpretation": (
                "This figure supports stress-test trend discussion; it does not by "
                "itself establish cross-scale generalization."
            ),
        }
    )

    e4_scenarios = sorted(
        {
            scenario
            for experiment, scenario, _algorithm, metric in lookup
            if experiment == "e4" and metric == "DVR"
        },
        key=_deadline_sort_key,
    )
    e4_algorithms = [
        algorithm
        for algorithm in ["greedy_min_cost", "maddpg", "mappo", "maps"]
        if any(("e4", scenario, algorithm, "DVR") in lookup for scenario in e4_scenarios)
    ]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 28, "E4 deadline sensitivity", size=16, weight="bold"),
    ]
    elements.extend(
        _line_panel(
            lookup,
            experiment="e4",
            scenarios=e4_scenarios,
            algorithms=e4_algorithms,
            metric="DVR",
            x=80,
            y=70,
            width=690,
            height=250,
            title="Deadline violation ratio across deadline profiles",
            xlabel="Deadline profile",
            label_mode="deadline",
        )
    )
    elements.append(
        _svg_text(
            width / 2,
            400,
            "Lower curves indicate fewer violations; interpretation must account for CIs in the numeric tables.",
            size=11,
        )
    )
    elements.append("</svg>")
    path = figures_dir / "figure-03-e4-deadline-sensitivity.svg"
    path.write_text("\n".join(elements), encoding="utf-8")
    figures.append(
        {
            "filename": str(path),
            "purpose": "Assess reliability sensitivity under loose/medium/strict deadlines.",
            "metric": "DVR",
            "key_observation": (
                "MAPS DVR by deadline profile: "
                + ", ".join(
                    f"{scenario.replace('s1_u10_', '').replace('s3_u10_', '')}="
                    f"{_format_float(_mean_lookup(lookup, 'e4', scenario, 'maps', 'DVR'))}"
                    for scenario in e4_scenarios
                )
                + "."
            ),
            "interpretation": (
                "This figure shows deadline sensitivity under the evaluated profiles; "
                "claims should stay within loose/medium/strict S3 settings."
            ),
        }
    )
    return figures


def _markdown_table(rows: list[dict[str, Any]], fields: list[str], limit: int = 12) -> str:
    if not rows:
        return "_No rows._"
    clipped = rows[:limit]
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = []
    for row in clipped:
        body.append("| " + " | ".join(_format_float(row.get(field)) for field in fields) + " |")
    if len(rows) > limit:
        body.append(f"| ... | {' | '.join('' for _ in fields[1:])} |")
    return "\n".join([header, sep, *body])


def build_claim_candidates(
    contrasts: list[dict[str, Any]],
    descriptive_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    valid_primary = [
        row
        for row in contrasts
        if row.get("status") == "valid"
        and row.get("method_a") == "maps"
        and row.get("metric") in PRIMARY_METRICS
    ]
    for row in valid_primary:
        mean_improvement = _float_or_none(row.get("mean_improvement"))
        ci_low = _float_or_none(row.get("ci95_improvement_low"))
        better_count = _int_or_none(row.get("maps_better_count"))
        n_pairs = _int_or_none(row.get("n_pairs")) or 0
        if mean_improvement is None:
            decision = "discard"
            allowed = "No claim; metric values are incomplete."
            uncertainty = "Missing metric values."
        elif mean_improvement > 0.0 and ci_low is not None and ci_low > 0.0:
            decision = "keep"
            allowed = (
                "MAPS showed a seed-paired improvement under the evaluated setting; "
                "report the mean delta, 95% CI, and effect size."
            )
            uncertainty = (
                f"n={n_pairs} paired seeds; exact sign-test p is coarse at this size."
            )
        elif mean_improvement > 0.0:
            decision = "weaken"
            allowed = (
                "MAPS had a favorable mean in this metric, but the CI crosses zero; "
                "word as a trend, not a confirmed advantage."
            )
            uncertainty = f"Only {better_count}/{n_pairs} paired seeds favor MAPS or CI crosses zero."
        else:
            decision = "discard"
            allowed = "Do not claim MAPS improves this metric in this comparison."
            uncertainty = "Mean directional improvement is non-positive."
        candidates.append(
            {
                "claim": (
                    f"{row['experiment']}/{row['scenario']} {row['contrast_label']} "
                    f"on {row['metric']}"
                ),
                "source_evidence": (
                    "paired_contrasts.csv: "
                    f"mean_improvement={_format_float(mean_improvement)}, "
                    f"ci95=[{_format_float(row.get('ci95_improvement_low'))}, "
                    f"{_format_float(row.get('ci95_improvement_high'))}], "
                    f"cohens_dz={_format_float(row.get('cohens_dz_improvement'))}"
                ),
                "allowed_wording": allowed,
                "forbidden_stronger_wording": (
                    "Do not claim general superiority, convergence guarantees, or "
                    "significance without the reported seed-level evidence."
                ),
                "uncertainty": uncertainty,
                "next_check": "Inspect raw task records and rerun if any scenario-level anomaly appears.",
                "decision": decision,
            }
        )

    blocked_baselines = [
        row
        for row in contrasts
        if row.get("status") == "blocked_single_run_baseline"
        and row.get("metric") in {"TCR", "DVR", "p95_task_latency"}
    ]
    seen: set[tuple[str, str, str]] = set()
    for row in blocked_baselines:
        key = (row["experiment"], row["scenario"], row["method_b"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "claim": (
                    f"{row['experiment']}/{row['scenario']} MAPS vs "
                    f"{METHOD_LABELS.get(row['method_b'], row['method_b'])}"
                ),
                "source_evidence": "descriptive_stats.csv and scenario_descriptive_stats.csv",
                "allowed_wording": (
                    "Report as a descriptive comparison only because the baseline has "
                    "one evaluation seed."
                ),
                "forbidden_stronger_wording": (
                    "Do not state seed-level statistical significance against this baseline."
                ),
                "uncertainty": "Baseline run count is one; scenario-level CIs are not training-seed CIs.",
                "next_check": "Add repeated baseline seeds or use a pre-registered scenario bootstrap protocol.",
                "decision": "descriptive_only",
            }
        )
    if not candidates:
        candidates.append(
            {
                "claim": "No C8 performance claim is currently supported.",
                "source_evidence": "No valid paired primary-metric contrast was available.",
                "allowed_wording": "State that C8 found insufficient evidence for a claim.",
                "forbidden_stronger_wording": "Do not infer superiority from incomplete data.",
                "uncertainty": "Analysis inputs may be incomplete.",
                "next_check": "Complete the missing formal evaluation matrix.",
                "decision": "discard",
            }
        )
    return candidates


def _top_e2_rows(descriptive_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in descriptive_rows
        if row["experiment"] == "e2"
        and row["scenario"] == "s1_u10_medium"
        and row["metric"] in {"TCR", "DVR", "p95_task_latency", "system_energy_per_completed_task"}
    ]
    order = {metric: index for index, metric in enumerate(PRIMARY_METRICS)}
    return sorted(rows, key=lambda row: (order.get(row["metric"], 99), METHOD_ORDER.index(row["algorithm"]) if row["algorithm"] in METHOD_ORDER else 99))


def write_reports(
    output_dir: Path,
    *,
    input_csv: Path,
    rows: list[dict[str, Any]],
    issues: list[AuditIssue],
    descriptive_rows: list[dict[str, Any]],
    scenario_rows: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    claims: list[dict[str, str]],
    figures: list[dict[str, str]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    issue_counts = Counter(issue.severity for issue in issues)
    valid_contrasts = [row for row in contrasts if row.get("status") == "valid"]
    blocked_contrasts = [row for row in contrasts if row.get("status") != "valid"]

    report_path = output_dir / "analysis-report.md"
    report = [
        "# C8 EI Results Analysis Report",
        "",
        f"- Input summary: `{input_csv}`",
        f"- Evaluation runs audited: {len(rows)}",
        f"- Artifact issues: {len(issues)} ({dict(issue_counts)})",
        f"- Valid paired MAPS contrasts: {len(valid_contrasts)}",
        f"- Blocked/non-inferential contrasts: {len(blocked_contrasts)}",
        "",
        "## Analysis Questions",
        "",
        "- E2: How does MAPS compare with baselines and learning methods on the frozen S1 test bank?",
        "- E3: How do methods change as UE count grows?",
        "- E4: How sensitive are the methods to loose/medium/strict deadline profiles?",
        "",
        "## Key Evidence Boundary",
        "",
        (
            "Learning-method comparisons use paired training seeds as the independent "
            "unit. Greedy-MinCost and LLM-only currently have one evaluation seed, so "
            "their comparisons are descriptive unless a separate bootstrap protocol is "
            "pre-registered."
        ),
        "",
        "## E2 Main Numeric Table",
        "",
        _markdown_table(
            _top_e2_rows(descriptive_rows),
            [
                "metric",
                "algorithm",
                "n",
                "mean",
                "std",
                "ci95_low",
                "ci95_high",
                "inference_status",
            ],
            limit=32,
        ),
        "",
        "## Claim Candidates",
        "",
    ]
    for claim in claims:
        report.extend(
            [
                f"- Claim: {claim['claim']}",
                f"  - Source evidence: {claim['source_evidence']}",
                f"  - Allowed wording: {claim['allowed_wording']}",
                f"  - Forbidden stronger wording: {claim['forbidden_stronger_wording']}",
                f"  - Uncertainty: {claim['uncertainty']}",
                f"  - Next check: {claim['next_check']}",
                f"  - Decision: {claim['decision']}",
            ]
        )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    stats_path = output_dir / "stats-appendix.md"
    issue_rows = [issue.to_dict() for issue in issues]
    stats = [
        "# C8 Statistical Appendix",
        "",
        "## Unit Of Analysis",
        "",
        "- Learning methods: one formal evaluation run per training seed.",
        "- Baselines: one evaluation seed in the current C7 matrix; not used for seed-level inferential tests.",
        "- Scenario-level rows are reported separately as within-run descriptive uncertainty.",
        "",
        "## Metric Directions",
        "",
        _markdown_table(
            [
                {"metric": metric, "direction": METRIC_DIRECTIONS.get(metric, "higher")}
                for metric in COMPARISON_METRICS
            ],
            ["metric", "direction"],
            limit=20,
        ),
        "",
        "## Paired Contrast Method",
        "",
        (
            "For MAPS vs learning-method comparators, C8 matches runs by experiment, "
            "scenario, and seed. It reports raw MAPS-comparator deltas, directional "
            "improvements, 95% t confidence intervals for the paired deltas, Cohen's "
            "dz, exact two-sided sign-test p-values, and Holm-adjusted p-values within "
            "each experiment/scenario/metric comparison family."
        ),
        "",
        "## Valid Contrast Preview",
        "",
        _markdown_table(
            valid_contrasts,
            [
                "experiment",
                "scenario",
                "contrast_label",
                "metric",
                "n_pairs",
                "mean_improvement",
                "ci95_improvement_low",
                "ci95_improvement_high",
                "cohens_dz_improvement",
                "sign_test_p",
                "holm_p",
            ],
            limit=24,
        ),
        "",
        "## Blockers And Audit Issues",
        "",
        _markdown_table(
            issue_rows,
            ["severity", "code", "experiment", "scenario", "algorithm", "seed", "message"],
            limit=30,
        ),
        "",
        "## Output Tables",
        "",
        "- `descriptive_stats.csv`: run/seed-level descriptive statistics.",
        "- `scenario_descriptive_stats.csv`: within-run scenario-level descriptive statistics.",
        "- `paired_contrasts.csv`: MAPS paired contrasts and blocked baseline contrasts.",
    ]
    stats_path.write_text("\n".join(stats) + "\n", encoding="utf-8")

    figure_catalog_path = output_dir / "figure-catalog.md"
    catalog = [
        "# C8 Figure Catalog",
        "",
        "All figures are generated from `descriptive_stats.csv`; error bars represent 95% CI when repeated seeds exist.",
        "",
    ]
    for index, figure in enumerate(figures, start=1):
        name = Path(figure["filename"]).name
        catalog.extend(
            [
                f"## Figure {index}: `{name}`",
                "",
                f"- Purpose: {figure['purpose']}",
                f"- Data source: `{input_csv}` via C8 descriptive statistics.",
                f"- Plotted metric(s): {figure['metric']}",
                "- Caption requirements: state metric direction, n per method, and CI/error-bar meaning.",
                f"- Key observation: {figure.get('key_observation', 'Inspect the plotted means together with descriptive_stats.csv.')}",
                f"- Interpretation: {figure.get('interpretation', 'Do not infer significance from the figure alone.')}",
                "- Interpretation checklist: Why this plot exists; what changed numerically; what claim is allowed by the seed-level table.",
                "- Known caveat: single-seed baselines have no seed-level CI in the bar/line figure.",
                "",
            ]
        )
    figure_catalog_path.write_text("\n".join(catalog), encoding="utf-8")

    return {
        "analysis_report": str(report_path),
        "stats_appendix": str(stats_path),
        "figure_catalog": str(figure_catalog_path),
    }


def write_audit_summary(
    output_dir: Path,
    *,
    input_csv: Path,
    rows: list[dict[str, Any]],
    issues: list[AuditIssue],
    descriptive_path: Path,
    scenario_descriptive_path: Path,
    contrasts_path: Path,
    report_paths: dict[str, str],
    figures: list[dict[str, str]],
    claims: list[dict[str, str]],
) -> Path:
    issue_counts = Counter(issue.severity for issue in issues)
    group_counts = Counter(
        f"{row['experiment']}:{row['scenario']}:{row['algorithm']}" for row in rows
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "input_csv": str(input_csv),
        "run_count": len(rows),
        "groups": dict(sorted(group_counts.items())),
        "issue_count": len(issues),
        "issue_counts_by_severity": dict(issue_counts),
        "gates": {
            "all_required_artifacts_present": not any(
                issue.code in {"required_artifacts_missing", "artifact_path_missing"}
                for issue in issues
            ),
            "all_manifests_passed": not any(
                issue.code == "manifest_not_passed" for issue in issues
            ),
            "zero_online_api_calls_in_evaluation": not any(
                issue.code == "online_api_calls_in_evaluation" for issue in issues
            ),
            "maps_actor_only_evaluation": not any(
                issue.code == "maps_eval_uses_llm" for issue in issues
            ),
            "llm_only_cache_coverage_complete": not any(
                issue.code.startswith("llm_cache_coverage") for issue in issues
            ),
        },
        "metric_directions": METRIC_DIRECTIONS,
        "claim_decisions": dict(Counter(claim["decision"] for claim in claims)),
        "issues": [issue.to_dict() for issue in issues],
        "outputs": {
            "descriptive_stats_csv": str(descriptive_path),
            "scenario_descriptive_stats_csv": str(scenario_descriptive_path),
            "paired_contrasts_csv": str(contrasts_path),
            **report_paths,
            "figures": figures,
        },
    }
    path = output_dir / "audit_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_results_audit(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    expected_learning_seeds: int | None = 5,
    expected_baseline_seeds: int | None = 1,
) -> dict[str, Any]:
    input_path = resolve_project_path(input_csv)
    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = load_evaluation_rows(input_path)
    issues = audit_artifacts(
        rows,
        expected_learning_seeds=expected_learning_seeds,
        expected_baseline_seeds=expected_baseline_seeds,
    )
    descriptive_rows = build_descriptive_stats(rows)
    scenario_rows = build_scenario_descriptive_stats(rows)
    contrasts = build_paired_contrasts(rows)
    claims = build_claim_candidates(contrasts, descriptive_rows)
    figures = write_figures(output_path, descriptive_rows)

    descriptive_path = output_path / "descriptive_stats.csv"
    scenario_descriptive_path = output_path / "scenario_descriptive_stats.csv"
    contrasts_path = output_path / "paired_contrasts.csv"
    _write_csv(
        descriptive_path,
        descriptive_rows,
        [
            "experiment",
            "scenario",
            "algorithm",
            "method_label",
            "metric",
            "direction",
            "unit_of_analysis",
            "n",
            "seeds",
            "mean",
            "std",
            "se",
            "ci95_low",
            "ci95_high",
            "inference_status",
        ],
    )
    _write_csv(
        scenario_descriptive_path,
        scenario_rows,
        [
            "experiment",
            "scenario",
            "algorithm",
            "seed",
            "metric",
            "direction",
            "unit_of_analysis",
            "n",
            "mean",
            "std",
            "se",
            "ci95_low",
            "ci95_high",
            "inference_status",
        ],
    )
    _write_csv(
        contrasts_path,
        contrasts,
        [
            "experiment",
            "scenario",
            "method_a",
            "method_b",
            "contrast_label",
            "metric",
            "direction",
            "unit_of_analysis",
            "n_pairs",
            "paired_seeds",
            "method_a_n",
            "method_b_n",
            "multiple_comparison_family",
            "test_name",
            "status",
            "blocker",
            "mean_delta_raw",
            "ci95_delta_raw_low",
            "ci95_delta_raw_high",
            "mean_improvement",
            "ci95_improvement_low",
            "ci95_improvement_high",
            "cohens_dz_improvement",
            "maps_better_count",
            "ties",
            "sign_test_p",
            "holm_p",
        ],
    )
    report_paths = write_reports(
        output_path,
        input_csv=input_path,
        rows=rows,
        issues=issues,
        descriptive_rows=descriptive_rows,
        scenario_rows=scenario_rows,
        contrasts=contrasts,
        claims=claims,
        figures=figures,
    )
    summary_path = write_audit_summary(
        output_path,
        input_csv=input_path,
        rows=rows,
        issues=issues,
        descriptive_path=descriptive_path,
        scenario_descriptive_path=scenario_descriptive_path,
        contrasts_path=contrasts_path,
        report_paths=report_paths,
        figures=figures,
        claims=claims,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_count": len(rows),
        "issue_count": len(issues),
        "outputs": {
            "audit_summary_json": str(summary_path),
            "descriptive_stats_csv": str(descriptive_path),
            "scenario_descriptive_stats_csv": str(scenario_descriptive_path),
            "paired_contrasts_csv": str(contrasts_path),
            **report_paths,
            "figures": figures,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the C8 EI result audit and statistical analysis."
    )
    parser.add_argument(
        "--input-csv",
        default="artifacts/ei/formal_matrix/analysis/evaluation_summary.csv",
        help="C7 evaluation_summary.csv path.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/ei/formal_matrix/results_audit",
        help="Directory for C8 analysis bundle.",
    )
    parser.add_argument(
        "--expected-learning-seeds",
        type=int,
        default=5,
        help="Expected seed count for learning methods; use 0 to disable.",
    )
    parser.add_argument(
        "--expected-baseline-seeds",
        type=int,
        default=1,
        help="Expected seed count for baseline methods; use 0 to disable.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_results_audit(
        args.input_csv,
        args.output_dir,
        expected_learning_seeds=(
            args.expected_learning_seeds if args.expected_learning_seeds > 0 else None
        ),
        expected_baseline_seeds=(
            args.expected_baseline_seeds if args.expected_baseline_seeds > 0 else None
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
