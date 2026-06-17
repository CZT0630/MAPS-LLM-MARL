#!/usr/bin/env python3
"""Unified command-line entry point for Phase 1 baselines."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from LLM4RL.experiments.runner import (  # noqa: E402
    SUPPORTED_ALGORITHMS,
    run_evaluation,
    run_baseline,
    run_phase1_audit,
)
from LLM4RL.utils.config import load_config  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent


def _parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _resolve_config(path: str | None, mode: str) -> Path:
    if path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    if mode in ("smoke", "audit"):
        return PROJECT_ROOT / "configs" / "smoke.yaml"
    return PROJECT_ROOT / "config.yaml"


def _apply_overrides(
    config: dict,
    episodes: int | None,
    steps: int | None,
    drain_steps: int | None = None,
) -> dict:
    config = copy.deepcopy(config)
    if episodes is not None:
        config.setdefault("training", {})["episodes"] = episodes
        config.setdefault("evaluation", {})["num_scenarios"] = episodes
        config.setdefault("evaluation", {})["max_scenarios"] = episodes
        for key in SUPPORTED_ALGORITHMS:
            config.setdefault(key, {})["max_episodes"] = episodes
    if steps is not None:
        config.setdefault("training", {})["max_steps_per_episode"] = steps
        config.setdefault("evaluation", {})["window_steps"] = steps
        for key in SUPPORTED_ALGORITHMS:
            config.setdefault(key, {})["max_steps"] = steps
        config.setdefault("maddpg", {})["max_steps"] = steps
    if drain_steps is not None:
        config.setdefault("evaluation", {})["drain_horizon_steps"] = drain_steps
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MAPS reproducible baseline and environment runner"
    )
    parser.add_argument(
        "--mode",
        choices=("train", "smoke", "audit", "eval"),
        default="smoke",
        help=(
            "train one baseline, run a short smoke test, execute Gate 1 audit, "
            "or run formal C4 evaluation"
        ),
    )
    parser.add_argument(
        "--algorithm",
        choices=SUPPORTED_ALGORITHMS + ("all",),
        default="all",
    )
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("--seeds", default="42", help="comma-separated seeds")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--drain-steps",
        type=int,
        help="C4 eval-only drain horizon after stopping new task arrivals",
    )
    parser.add_argument("--output-root", default="results/phase1")
    parser.add_argument(
        "--audit-path",
        default="artifacts/phase1/baseline_audit.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = _resolve_config(args.config, args.mode)
    config = _apply_overrides(
        load_config(str(config_path)),
        args.episodes,
        args.steps,
        args.drain_steps,
    )
    seeds = _parse_seeds(args.seeds)

    if args.mode == "audit":
        audit = run_phase1_audit(
            config=config,
            seeds=seeds,
            output_root=args.output_root,
            audit_path=args.audit_path,
        )
        print(json.dumps(audit["checks"], ensure_ascii=False, indent=2))
        return 0 if audit["gate_1_passed"] else 1

    if args.mode == "eval":
        algorithms = (
            list(SUPPORTED_ALGORITHMS)
            if args.algorithm == "all"
            else [args.algorithm]
        )
        failed = False
        for seed in seeds:
            for algorithm in algorithms:
                result = run_evaluation(
                    algorithm=algorithm,
                    config=config,
                    seed=seed,
                    output_root=args.output_root,
                )
                metrics = result["extra"]["evaluation_metrics"]
                print(
                    f"{algorithm} eval seed={seed}: {result['status']} "
                    f"TCR={metrics['TCR']:.6f} DVR={metrics['DVR']:.6f}"
                )
                failed = failed or result["status"] != "passed"
        return 1 if failed else 0

    algorithms = (
        list(SUPPORTED_ALGORITHMS)
        if args.algorithm == "all"
        else [args.algorithm]
    )
    if args.mode == "smoke":
        config = _apply_overrides(
            config,
            episodes=args.episodes if args.episodes is not None else 2,
            steps=args.steps if args.steps is not None else 5,
            drain_steps=args.drain_steps,
        )

    failed = False
    for seed in seeds:
        for algorithm in algorithms:
            result = run_baseline(
                algorithm=algorithm,
                config=config,
                seed=seed,
                output_root=args.output_root,
            )
            print(
                f"{algorithm} seed={seed}: {result['status']} "
                f"reward={result['metrics']['reward']:.6f}"
            )
            failed = failed or result["status"] != "passed"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
