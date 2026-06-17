"""Expand the EI expert cache from recorded MAPS cache-miss states."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_package_root = Path(__file__).resolve().parents[1]
_workspace_root = _package_root.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from LLM4RL.llm_assistant.ei_prompt_builder import EIPromptBuilder
from LLM4RL.llm_assistant.expert_cache import ExpertCache, hash_state
from LLM4RL.llm_assistant.llm_client import LLMClient
from LLM4RL.utils.config import load_config


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else _package_root / path


def expand_input_paths(paths: list[str | Path]) -> list[Path]:
    """Resolve files or directories containing missed_expert_states artifacts."""
    output: list[Path] = []
    for path_value in paths:
        path = _resolve_project_path(path_value)
        if path.is_dir():
            output.extend(sorted(path.rglob("missed_expert_states.json")))
        elif path.is_file():
            output.append(path)
        else:
            raise FileNotFoundError(f"missing state input not found: {path}")
    if not output:
        raise ValueError("no missed_expert_states.json files found")
    return output


def load_missing_state_records(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load and de-duplicate missed states by canonical state hash."""
    unique: dict[str, dict[str, Any]] = {}
    for path in expand_input_paths(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "ei-missed-expert-states-v1":
            raise ValueError(f"unsupported missing-state schema in {path}")
        for record in payload.get("records", []):
            state = record.get("state")
            if state is None:
                raise ValueError(f"missing state payload in {path}")
            state_hash = str(record.get("state_hash") or hash_state(state))
            if state_hash != hash_state(state):
                raise ValueError(f"state_hash does not match state payload in {path}")
            unique.setdefault(state_hash, {"state_hash": state_hash, "state": state})
    return list(unique.values())


def build_prompt(builder: EIPromptBuilder, state: dict[str, Any]) -> str:
    return builder.build(
        ue_info=state["ue_states"],
        es_info=state["es_states"],
        cs_info=state["cs_states"],
        tasks_info=state["tasks"],
        backhaul_info=state["backhaul"],
        num_edges=state["num_edges"],
    )


def expand_cache_from_records(
    *,
    records: list[dict[str, Any]],
    cache_path: str | Path,
    llm_config_path: str | Path,
    output_path: str | Path | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    cache_path = _resolve_project_path(cache_path)
    output_path = _resolve_project_path(output_path or cache_path)
    cache = ExpertCache.load(cache_path)
    builder = EIPromptBuilder()
    all_missing = [
        record
        for record in records
        if not cache.has(str(record["state_hash"]))
    ]
    missing = list(all_missing)
    if limit is not None:
        missing = missing[: int(limit)]
    summary: dict[str, Any] = {
        "schema_version": "ei-cache-expansion-summary-v1",
        "cache_path": str(cache_path),
        "output_path": str(output_path),
        "input_state_count": len(records),
        "states_missing_from_cache": len(all_missing),
        "states_selected_for_query": len(missing),
        "limit": limit,
        "dry_run": dry_run,
        "added": 0,
        "skipped_existing": len(records) - len(all_missing),
    }
    if dry_run or not missing:
        summary["cache_stats_before"] = cache.stats()
        summary["cache_stats_after"] = cache.stats()
        return summary

    llm_config = load_config(str(_resolve_project_path(llm_config_path)))
    client = LLMClient(llm_config)
    if client.provider != cache.provider:
        raise ValueError("LLM provider does not match cache provider")
    if client.model_name != cache.requested_model:
        raise ValueError("LLM model does not match cache requested_model")
    if client.temperature != cache.temperature:
        raise ValueError("LLM temperature does not match cache")

    summary["cache_stats_before"] = cache.stats()
    for record in missing:
        state = record["state"]
        state_hash = str(record["state_hash"])
        prompt = build_prompt(builder, state)
        started = time.perf_counter()
        raw_response = client.query(prompt)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        cache.add_from_api_response(
            state_hash=state_hash,
            raw_response=raw_response,
            num_devices=len(state["ue_states"]),
            num_edges=int(state["num_edges"]),
            llm_metadata=client.last_metadata,
            query_latency_ms=elapsed_ms,
            decision_mask=[task is not None for task in state["tasks"]],
        )
        cache.save(output_path)
        summary["added"] += 1
        print(
            f"[{summary['added']}/{len(missing)}] "
            f"hash={state_hash[:12]} cache_size={len(cache)}"
        )
    summary["cache_stats_after"] = cache.stats()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query MiMo for MAPS training states missing from the EI cache."
    )
    parser.add_argument(
        "--states",
        action="append",
        required=True,
        help="missed_expert_states.json file or directory containing such files.",
    )
    parser.add_argument("--cache", default="artifacts/ei/expert_cache.json")
    parser.add_argument("--output", help="Output cache path; defaults to --cache.")
    parser.add_argument("--config", default="configs/ei/llm_mimo.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summary-output")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = load_missing_state_records(args.states)
    summary = expand_cache_from_records(
        records=records,
        cache_path=args.cache,
        llm_config_path=args.config,
        output_path=args.output,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    if args.summary_output:
        summary_path = _resolve_project_path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
