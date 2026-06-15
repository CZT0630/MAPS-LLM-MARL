"""Generate or resume the formal joint-state MiMo expert cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_package_root = Path(__file__).resolve().parents[1]
_workspace_root = _package_root.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from LLM4RL.environment.cloud_edge_env import CloudEdgeDeviceEnv
from LLM4RL.llm_assistant.ei_prompt_builder import EIPromptBuilder
from LLM4RL.llm_assistant.ei_state import (
    build_ei_state_payload,
    ei_environment_fingerprint,
)
from LLM4RL.llm_assistant.expert_cache import ExpertCache, hash_state
from LLM4RL.llm_assistant.llm_client import LLMClient
from LLM4RL.utils.config import load_config
from LLM4RL.utils.seed import set_global_seed


def _entry_env_actions(entry, num_devices: int) -> list[list[float]]:
    if len(entry.parsed_action) != num_devices:
        raise ValueError("cached action count does not match environment")
    actions = []
    for action, valid in zip(entry.parsed_action, entry.valid_mask):
        if not valid:
            actions.append([1.0, 0.0, 0.0, 0.0])
            continue
        actions.append(
            [
                action["partition"]["local"],
                action["partition"]["edge"],
                action["partition"]["cloud"],
                float(action["edge_id"]),
            ]
        )
    return actions


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else _package_root / path


def _write_scenario_bank(
    path: Path,
    *,
    scenarios: list[dict[str, int]],
    cache: ExpertCache,
    cache_path: Path,
    env_config_path: Path,
    environment_fingerprint: str,
    construction_seed: int,
) -> None:
    core = {
        "schema_version": "ei-expert-scenario-bank-v1",
        "scenarios": scenarios,
        "cache_sha256": cache.file_sha256,
        "environment_fingerprint": environment_fingerprint,
        "construction_seed": construction_seed,
        "prompt_version": cache.prompt_version,
        "state_key_version": cache.state_key_version,
    }
    scenario_bank_id = "ei-expert-" + hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    payload = {
        **core,
        "scenario_bank_id": scenario_bank_id,
        "cache_path": str(cache_path),
        "env_config_path": str(env_config_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def _build_prompt(
    builder: EIPromptBuilder, state: dict
) -> str:
    return builder.build(
        ue_info=state["ue_states"],
        es_info=state["es_states"],
        cs_info=state["cs_states"],
        tasks_info=state["tasks"],
        backhaul_info=state["backhaul"],
        num_edges=state["num_edges"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate state-keyed expert cache from MiMo API."
    )
    parser.add_argument(
        "--config",
        default="configs/ei/llm_mimo.yaml",
        help="LLM config YAML.",
    )
    parser.add_argument(
        "--env-config",
        required=True,
        help="Frozen Phase-2 environment config YAML.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Target number of unique joint states.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="artifacts/ei/expert_cache.json",
    )
    parser.add_argument(
        "--scenario-bank-output",
        default="artifacts/ei/expert_scenario_bank.json",
        help="Frozen LLM-only scenario bank generated with the cache.",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing cache instead of resuming it.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    llm_config_path = _resolve_project_path(args.config)
    env_config_path = _resolve_project_path(args.env_config)
    if not llm_config_path.is_file():
        raise FileNotFoundError(f"LLM config not found: {llm_config_path}")
    if not env_config_path.is_file():
        raise FileNotFoundError(
            f"environment config not found: {env_config_path}"
        )
    llm_config = load_config(str(llm_config_path))
    env_config = load_config(str(env_config_path))
    if int(env_config.get("environment", {}).get("physics_version", 1)) < 2:
        raise ValueError("--env-config must enable environment.physics_version >= 2")
    set_global_seed(args.seed)
    client = LLMClient(llm_config)
    builder = EIPromptBuilder()
    if client.provider != "xiaomi_mimo":
        raise ValueError("formal C3 cache provider must be xiaomi_mimo")
    if client.model_name != "mimo-v2.5":
        raise ValueError("formal C3 cache model must be mimo-v2.5")
    if client.temperature != 0.0:
        raise ValueError("formal C3 cache temperature must be 0.0")

    output_path = _resolve_project_path(args.output)
    scenario_bank_path = _resolve_project_path(args.scenario_bank_output)
    if output_path.exists() and not args.overwrite:
        cache = ExpertCache.load(output_path)
        if (
            cache.prompt_version != builder.VERSION
            or cache.provider != client.provider
            or cache.requested_model != client.model_name
            or cache.temperature != client.temperature
        ):
            raise ValueError(
                "existing cache metadata does not match current LLM config"
            )
    else:
        cache = ExpertCache(
            prompt_version=builder.VERSION,
            provider=client.provider,
            requested_model=client.model_name,
            temperature=client.temperature,
        )

    env = CloudEdgeDeviceEnv(env_config)
    collected = len(cache)
    uncovered_hashes = set(cache.state_hashes)
    episode = 0
    api_calls = 0
    cache_hits = 0
    scenario_steps: dict[int, int] = {}

    print(f"Target: {args.num_samples} unique joint states")
    print(f"Existing entries: {collected}")
    print(f"Provider: {client.provider}, Model: {client.model_name}")

    while collected < args.num_samples or uncovered_hashes:
        scenario_seed = args.seed + episode
        _global_state, _ = env.reset(seed=scenario_seed)
        env.max_steps = args.max_steps
        executed_steps = 0
        for _step in range(args.max_steps):
            if collected >= args.num_samples and not uncovered_hashes:
                break
            state = build_ei_state_payload(env)
            if not any(task is not None for task in state["tasks"]):
                dummy = [[1.0, 0.0, 0.0, 0.0]] * env.num_devices
                _, _, terminated, truncated, _ = env.step(dummy)
                executed_steps += 1
                if terminated or truncated:
                    break
                continue

            state_hash = hash_state(state)
            entry = cache.get(state_hash)
            if entry is not None:
                cache_hits += 1
                uncovered_hashes.discard(state_hash)
            else:
                prompt = _build_prompt(builder, state)
                started = time.perf_counter()
                raw_response = client.query(prompt)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                entry = cache.add_from_api_response(
                    state_hash=state_hash,
                    raw_response=raw_response,
                    num_devices=env.num_devices,
                    num_edges=env.num_edges,
                    llm_metadata=client.last_metadata,
                    query_latency_ms=elapsed_ms,
                    decision_mask=[
                        task is not None for task in state["tasks"]
                    ],
                )
                cache.save(output_path)
                api_calls += 1
                collected = len(cache)
                print(
                    f"[{collected}/{args.num_samples}] "
                    f"hash={state_hash[:12]} "
                    f"parser={entry.parser_status} "
                    f"latency={elapsed_ms:.0f}ms"
                )

            env_actions = _entry_env_actions(entry, env.num_devices)
            _, _, terminated, truncated, _ = env.step(env_actions)
            executed_steps += 1
            if terminated or truncated:
                break
        if executed_steps:
            scenario_steps[scenario_seed] = max(
                scenario_steps.get(scenario_seed, 0),
                executed_steps,
            )
        episode += 1

    cache.save(output_path)
    scenarios = [
        {"seed": seed, "max_steps": steps}
        for seed, steps in sorted(scenario_steps.items())
    ]
    _write_scenario_bank(
        scenario_bank_path,
        scenarios=scenarios,
        cache=cache,
        cache_path=output_path,
        env_config_path=env_config_path,
        environment_fingerprint=ei_environment_fingerprint(env_config),
        construction_seed=args.seed,
    )
    stats = cache.stats()
    print(f"Cache saved to {output_path}")
    print(f"Scenario bank saved to {scenario_bank_path}")
    print(f"Total entries: {stats['total']}")
    print(f"Parser success rate: {stats['parser_success_rate']:.2%}")
    print(f"Valid action rate: {stats['valid_action_rate']:.2%}")
    print(f"Fallback rate: {stats['fallback_rate']:.2%}")
    print(f"API calls this run: {api_calls}")
    print(f"Cache hits this run: {cache_hits}")


if __name__ == "__main__":
    main()
