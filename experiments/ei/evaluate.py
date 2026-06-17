"""EI evaluation entry point using frozen test scenario banks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[3]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from LLM4RL.experiments.ei.config import compose_ei_config, with_bank_paths  # noqa: E402
from LLM4RL.experiments.runner import SUPPORTED_ALGORITHMS, run_evaluation  # noqa: E402


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def default_method_fragment(algorithm: str) -> str:
    return f"configs/ei/methods/{algorithm}.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EI formal evaluation")
    parser.add_argument("--algorithm", choices=SUPPORTED_ALGORITHMS, required=True)
    parser.add_argument("--base-config", default="configs/ei/base.yaml")
    parser.add_argument("--method-config")
    parser.add_argument("--scale-config", default="configs/ei/scales/s1_u10.yaml")
    parser.add_argument("--deadline-config", default="configs/ei/deadlines/medium.yaml")
    parser.add_argument("--fragment", action="append", default=[])
    parser.add_argument("--test-bank")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--drain-steps", type=int)
    parser.add_argument("--output-root", default="artifacts/ei/evaluations")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    method_config = args.method_config or default_method_fragment(args.algorithm)
    fragments = [
        method_config,
        args.scale_config,
        args.deadline_config,
        *args.fragment,
    ]
    config = compose_ei_config(args.base_config, fragments)
    config = with_bank_paths(config, test_bank=args.test_bank)
    if args.checkpoint_dir:
        config.setdefault("evaluation", {})["checkpoint_dir"] = args.checkpoint_dir
    if args.allow_untrained:
        config.setdefault("evaluation", {})["allow_untrained"] = True
    if args.steps is not None:
        config.setdefault("evaluation", {})["window_steps"] = args.steps
    if args.drain_steps is not None:
        config.setdefault("evaluation", {})["drain_horizon_steps"] = args.drain_steps
    results = []
    for seed in parse_seeds(args.seeds):
        results.append(
            run_evaluation(
                algorithm=args.algorithm,
                config=config,
                seed=seed,
                output_root=args.output_root,
            )
        )
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    return 0 if all(item["status"] == "passed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

