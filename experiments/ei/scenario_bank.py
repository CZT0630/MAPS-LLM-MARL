"""Build deterministic train/test scenario banks for EI experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PACKAGE_PARENT = Path(__file__).resolve().parents[3]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from LLM4RL.experiments.ei.config import (  # noqa: E402
    compose_ei_config,
    resolve_project_path,
)
from LLM4RL.llm_assistant.ei_state import ei_environment_fingerprint  # noqa: E402
from LLM4RL.utils.run_manifest import stable_config_hash  # noqa: E402


def parse_seed_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def seed_range(start: int, count: int) -> list[int]:
    if count < 0:
        raise ValueError("seed count must be non-negative")
    return [start + index for index in range(count)]


def build_scenario_bank(
    *,
    config: dict[str, Any],
    scenario_id: str,
    split: str,
    seeds: list[int],
    max_steps: int,
    construction_seed: int,
    description: str = "",
) -> dict[str, Any]:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if not seeds:
        raise ValueError(f"{split} scenario bank requires at least one seed")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    scenarios = [
        {
            "scenario_id": f"{scenario_id}-{split}-{index:04d}",
            "seed": int(seed),
            "max_steps": int(max_steps),
        }
        for index, seed in enumerate(seeds)
    ]
    core = {
        "schema_version": "ei-scenario-bank-v1",
        "scenario_id": scenario_id,
        "split": split,
        "scenarios": scenarios,
        "environment_fingerprint": ei_environment_fingerprint(config),
        "resolved_config_hash": stable_config_hash(config),
        "construction_seed": int(construction_seed),
        "determinism_contract": {
            "task_arrivals_and_attributes": "regenerated from scenario seed and resolved config",
            "ue_es_distances_and_channel": "shared through resolved config and scenario seed",
            "initial_queues": "reset by scenario seed before each run",
            "deadline_distribution": "shared through resolved config",
        },
    }
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return {
        **core,
        "scenario_bank_id": f"ei-{scenario_id}-{split}-{digest}",
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_scenario_bank(path: str | Path, bank: dict[str, Any]) -> Path:
    output = resolve_project_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        bank,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create separated EI train/test scenario banks"
    )
    parser.add_argument("--base-config", default="configs/ei/base.yaml")
    parser.add_argument("--fragment", action="append", default=[])
    parser.add_argument("--scenario-id", default="s1_medium")
    parser.add_argument("--output-dir", default="artifacts/ei/scenario_banks")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--construction-seed", type=int, default=42)
    parser.add_argument("--train-seeds")
    parser.add_argument("--test-seeds")
    parser.add_argument("--train-start", type=int, default=1000)
    parser.add_argument("--test-start", type=int, default=2000)
    parser.add_argument("--train-count", type=int, default=5)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--description", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = compose_ei_config(args.base_config, args.fragment)
    train_seeds = parse_seed_list(args.train_seeds) or seed_range(
        args.train_start,
        args.train_count,
    )
    test_seeds = parse_seed_list(args.test_seeds) or seed_range(
        args.test_start,
        args.test_count,
    )
    if set(train_seeds) & set(test_seeds):
        raise ValueError("train and test scenario banks must use disjoint seeds")
    output_dir = resolve_project_path(args.output_dir)
    train_bank = build_scenario_bank(
        config=config,
        scenario_id=args.scenario_id,
        split="train",
        seeds=train_seeds,
        max_steps=args.max_steps,
        construction_seed=args.construction_seed,
        description=args.description,
    )
    test_bank = build_scenario_bank(
        config=config,
        scenario_id=args.scenario_id,
        split="test",
        seeds=test_seeds,
        max_steps=args.max_steps,
        construction_seed=args.construction_seed,
        description=args.description,
    )
    train_path = write_scenario_bank(
        output_dir / f"{args.scenario_id}_train.json",
        train_bank,
    )
    test_path = write_scenario_bank(
        output_dir / f"{args.scenario_id}_test.json",
        test_bank,
    )
    print(
        json.dumps(
            {
                "train_bank": str(train_path),
                "train_bank_id": train_bank["scenario_bank_id"],
                "test_bank": str(test_path),
                "test_bank_id": test_bank["scenario_bank_id"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

