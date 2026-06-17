"""Aggregate EI evaluation artifacts into machine-readable summaries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PACKAGE_PARENT = Path(__file__).resolve().parents[3]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from LLM4RL.experiments.ei.config import resolve_project_path  # noqa: E402


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


def _artifact_exists(path_value: str | None) -> bool:
    return bool(path_value) and Path(path_value).exists()


def collect_evaluation_runs(input_root: str | Path) -> list[dict[str, Any]]:
    root = resolve_project_path(input_root)
    rows = []
    for metrics_path in sorted(root.rglob("evaluation_metrics.json")):
        run_dir = metrics_path.parent
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest_path = run_dir / "run_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        metrics = metrics_payload.get("metrics", {})
        row = {
            "run_dir": str(run_dir),
            "algorithm": metrics_payload.get(
                "algorithm",
                manifest.get("algorithm"),
            ),
            "seed": metrics_payload.get("seed", manifest.get("seed")),
            "scenario_bank_id": metrics_payload.get("scenario_bank_id"),
            "scenario_count": metrics_payload.get("scenario_count"),
            "evaluation_metrics_path": str(metrics_path),
            "task_records_path": metrics_payload.get("task_records_path"),
            "run_manifest_path": str(manifest_path),
            "resolved_config_path": str(run_dir / "resolved_config.yaml"),
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(key)
        row["required_artifacts_present"] = all(
            [
                _artifact_exists(row["evaluation_metrics_path"]),
                _artifact_exists(row["task_records_path"]),
                _artifact_exists(row["run_manifest_path"]),
                _artifact_exists(row["resolved_config_path"]),
            ]
        )
        rows.append(row)
    return rows


def summarize_by_algorithm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["algorithm"])].append(row)
    summary = {}
    for algorithm, items in grouped.items():
        metric_summary = {}
        for key in METRIC_KEYS:
            values = [
                float(item[key])
                for item in items
                if item.get(key) is not None
            ]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = (
                sum((value - mean) ** 2 for value in values) / len(values)
            )
            metric_summary[key] = {
                "mean": mean,
                "std": variance ** 0.5,
                "count": len(values),
            }
        summary[algorithm] = {
            "runs": len(items),
            "all_required_artifacts_present": all(
                item["required_artifacts_present"] for item in items
            ),
            "metrics": metric_summary,
        }
    return summary


def write_outputs(rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, str]:
    output = resolve_project_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "evaluation_summary.csv"
    json_path = output / "analysis_summary.json"
    fieldnames = [
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
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": "ei-analysis-v1",
        "run_count": len(rows),
        "summary_by_algorithm": summarize_by_algorithm(rows),
        "runs": rows,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return {
        "evaluation_summary_csv": str(csv_path),
        "analysis_summary_json": str(json_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate EI evaluation artifacts")
    parser.add_argument("--input-root", default="artifacts/ei/evaluations")
    parser.add_argument("--output-dir", default="artifacts/ei/analysis")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = collect_evaluation_runs(args.input_root)
    outputs = write_outputs(rows, args.output_dir)
    print(
        json.dumps(
            {"run_count": len(rows), **outputs},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

