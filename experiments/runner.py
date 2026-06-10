"""Unified Phase 1 baseline runners."""

from __future__ import annotations

import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from LLM4RL.algos.common.trajectory_buffer import TrajectoryBuffer
from LLM4RL.algos.happo.happo_agent import HAPPOAgent
from LLM4RL.algos.maddpg.maddpg_agent import MADDPGAgent
from LLM4RL.algos.maddpg.replay_buffer import JointReplayBuffer
from LLM4RL.algos.mappo.mappo_agent import MAPPOAgent
from LLM4RL.environment.cloud_edge_env import CloudEdgeDeviceEnv
from LLM4RL.llm_assistant.expert_provider import FixedCacheExpertProvider
from LLM4RL.utils.run_manifest import build_manifest, write_manifest
from LLM4RL.utils.seed import set_global_seed


SUPPORTED_ALGORITHMS = ("maddpg", "legacy_maps", "mappo", "happo")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_run_dir(output_root: Path, algorithm: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"{timestamp}_{algorithm}_seed{seed}"
    (run_dir / "models").mkdir(parents=True, exist_ok=False)
    return run_dir


def _local_states(env: CloudEdgeDeviceEnv, global_state) -> np.ndarray:
    return np.asarray(
        [env.extract_agent_state(global_state, i) for i in range(env.num_devices)],
        dtype=np.float32,
    )


def _decode_continuous_actions(policy_actions: np.ndarray, num_edges: int) -> np.ndarray:
    actions = np.asarray(policy_actions, dtype=np.float32).copy()
    edge_ids = np.floor(actions[:, 3] * num_edges).astype(np.int64)
    actions[:, 3] = np.clip(edge_ids, 0, num_edges - 1)
    return actions


def _episode_metrics(
    reward_steps: list[float],
    latencies: list[float],
    energies: list[float],
    completion_stats: dict[str, Any],
) -> dict[str, float]:
    return {
        "reward": float(np.mean(reward_steps)) if reward_steps else 0.0,
        "latency": float(np.mean(latencies)) if latencies else 0.0,
        "energy": float(np.mean(energies)) if energies else 0.0,
        "completion_rate": float(
            completion_stats.get("on_time_completion_rate", 0.0)
        ),
    }


def _finalize_run(
    run_dir: Path,
    manifest: dict[str, Any],
    episodes: list[dict[str, float]],
    losses: list[dict[str, float]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = {
        key: float(np.mean([episode[key] for episode in episodes]))
        for key in ("reward", "latency", "energy", "completion_rate")
    }
    finite = all(math.isfinite(value) for value in metrics.values())
    result = {
        "status": "passed" if finite and episodes else "failed",
        "run_dir": str(run_dir),
        "episodes": len(episodes),
        "updates": len(losses),
        "metrics": metrics,
        "finite_metrics": finite,
        "extra": extra or {},
    }
    (run_dir / "episode_metrics.json").write_text(
        json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "training_losses.json").write_text(
        json.dumps(losses, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest.update(
        {
            "status": result["status"],
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "summary": result,
        }
    )
    write_manifest(run_dir / "run_manifest.json", manifest)
    return result


def _run_maddpg_family(
    algorithm: str,
    config: dict[str, Any],
    seed: int,
    output_root: Path,
    command: str,
) -> dict[str, Any]:
    set_global_seed(seed)
    run_dir = _make_run_dir(output_root, algorithm, seed)
    manifest = build_manifest(PROJECT_ROOT, algorithm, seed, config, command)
    write_manifest(run_dir / "run_manifest.json", manifest)

    env = CloudEdgeDeviceEnv(config)
    state_dim = env.get_agent_state_dim()
    action_dim = env.action_space.shape[0]
    num_agents = env.num_devices
    base_cfg = copy.deepcopy(config.get("maddpg", {}))
    algorithm_cfg = copy.deepcopy(base_cfg)
    expert_provider = None
    expert_metadata: dict[str, Any] = {}

    if algorithm == "legacy_maps":
        legacy_cfg = config.get("legacy_maps", {})
        algorithm_cfg.update(legacy_cfg)
        algorithm_cfg["distill_weight"] = float(legacy_cfg.get("distill_weight", 0.2))
        cache_path = Path(legacy_cfg["expert_cache"])
        if not cache_path.is_absolute():
            cache_path = PROJECT_ROOT / cache_path
        expert_provider = FixedCacheExpertProvider(
            cache_path=cache_path,
            num_agents=num_agents,
            num_edges=env.num_edges,
        )

    agents = [
        MADDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            agent_idx=index,
            num_edges=env.num_edges,
            config=algorithm_cfg,
        )
        for index in range(num_agents)
    ]
    replay = JointReplayBuffer(int(algorithm_cfg.get("buffer_size", 100000)))
    episodes_count = int(
        algorithm_cfg.get(
            "max_episodes", config.get("training", {}).get("episodes", 20)
        )
    )
    max_steps = int(algorithm_cfg.get("max_steps", 10))
    train_frequency = int(algorithm_cfg.get("train_frequency", 1))
    batch_size = int(algorithm_cfg.get("batch_size", 64))

    episodes: list[dict[str, float]] = []
    losses: list[dict[str, float]] = []
    global_step = 0

    for episode in range(episodes_count):
        global_state, _ = env.reset(seed=seed + episode)
        for agent in agents:
            agent.reset_noise()
        reward_steps: list[float] = []
        latencies: list[float] = []
        energies: list[float] = []
        info: dict[str, Any] = {}

        for step in range(max_steps):
            global_step += 1
            states = _local_states(env, global_state)
            policy_actions = np.asarray(
                [
                    agent.select_action(
                        states[index],
                        add_noise=episode < max(1, int(episodes_count * 0.7)),
                    )
                    for index, agent in enumerate(agents)
                ],
                dtype=np.float32,
            )
            env_actions = np.asarray(
                [
                    agent.policy_to_env_action(policy_actions[index])
                    for index, agent in enumerate(agents)
                ],
                dtype=np.float32,
            )

            expert_actions = None
            expert_mask = None
            if expert_provider is not None:
                expert_batch = expert_provider.get_actions(episode, step)
                expert_actions = expert_batch.policy_actions
                expert_mask = expert_batch.valid_mask
                expert_metadata = expert_batch.metadata

            next_global_state, rewards, terminated, truncated, info = env.step(
                env_actions,
                llm_actions=(
                    expert_actions.tolist() if expert_actions is not None else None
                ),
            )
            next_states = _local_states(env, next_global_state)
            done = bool(terminated or truncated)
            replay.add(
                states=states,
                actions=policy_actions,
                rewards=rewards,
                next_states=next_states,
                done=done,
                expert_actions=expert_actions,
                expert_mask=expert_mask,
            )

            has_task = info.get("has_task_list", [True] * num_agents)
            valid_rewards = [
                float(reward)
                for reward, present in zip(rewards, has_task)
                if present
            ]
            reward_steps.append(
                float(np.mean(valid_rewards))
                if valid_rewards
                else float(np.mean(rewards))
            )
            latencies.extend(
                float(value)
                for value, present in zip(info.get("total_latencies", []), has_task)
                if present
            )
            energies.extend(
                float(value)
                for value, present in zip(info.get("total_energies", []), has_task)
                if present
            )

            if len(replay) >= batch_size and global_step % train_frequency == 0:
                batch = replay.sample(batch_size)
                step_losses = [agent.update(batch, agents) for agent in agents]
                losses.append(
                    {
                        "global_step": global_step,
                        "critic_loss": float(
                            np.mean([item["critic_loss"] for item in step_losses])
                        ),
                        "actor_loss": float(
                            np.mean([item["actor_loss"] for item in step_losses])
                        ),
                        "distill_loss": float(
                            np.mean([item["distill_loss"] for item in step_losses])
                        ),
                    }
                )

            global_state = next_global_state
            if done:
                break

        episodes.append(
            _episode_metrics(
                reward_steps,
                latencies,
                energies,
                info.get("task_completion_stats", {}),
            )
        )

    for index, agent in enumerate(agents):
        agent.save_model(run_dir / "models" / f"agent_{index}_final.pt")

    return _finalize_run(
        run_dir,
        manifest,
        episodes,
        losses,
        extra={
            "buffer_size": len(replay),
            "joint_replay": True,
            "centralized_critic": True,
            "expert": expert_metadata,
        },
    )


def _run_on_policy(
    algorithm: str,
    config: dict[str, Any],
    seed: int,
    output_root: Path,
    command: str,
) -> dict[str, Any]:
    set_global_seed(seed)
    run_dir = _make_run_dir(output_root, algorithm, seed)
    manifest = build_manifest(PROJECT_ROOT, algorithm, seed, config, command)
    write_manifest(run_dir / "run_manifest.json", manifest)

    env = CloudEdgeDeviceEnv(config)
    state_dim = env.get_agent_state_dim()
    global_state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    num_agents = env.num_devices
    algorithm_cfg = copy.deepcopy(config.get(algorithm, {}))
    agent_class = MAPPOAgent if algorithm == "mappo" else HAPPOAgent
    agents = [
        agent_class(
            state_dim=state_dim,
            action_dim=action_dim,
            global_state_dim=global_state_dim,
            agent_idx=index,
            config=algorithm_cfg,
        )
        for index in range(num_agents)
    ]
    episodes_count = int(algorithm_cfg.get("max_episodes", 20))
    max_steps = int(algorithm_cfg.get("max_steps", 10))
    episodes: list[dict[str, float]] = []
    losses: list[dict[str, float]] = []

    for episode in range(episodes_count):
        global_state, _ = env.reset(seed=seed + episode)
        buffer = TrajectoryBuffer(num_agents)
        reward_steps: list[float] = []
        latencies: list[float] = []
        energies: list[float] = []
        info: dict[str, Any] = {}

        for _ in range(max_steps):
            policy_actions = []
            pending = []
            for index, agent in enumerate(agents):
                state = env.extract_agent_state(global_state, index)
                action, action_info = agent.select_action(
                    state, global_state, deterministic=False
                )
                policy_actions.append(action)
                pending.append((index, state, action, action_info))
            policy_actions_array = np.asarray(policy_actions, dtype=np.float32)
            env_actions = _decode_continuous_actions(
                policy_actions_array, env.num_edges
            )
            next_global_state, rewards, terminated, truncated, info = env.step(
                env_actions
            )
            done = bool(terminated or truncated)

            for index, state, action, action_info in pending:
                buffer.add(
                    index,
                    state,
                    global_state,
                    action,
                    rewards[index],
                    float(done),
                    action_info["log_prob"],
                    action_info["value"],
                    action_info["alpha"],
                    action_info["beta"],
                )

            has_task = info.get("has_task_list", [True] * num_agents)
            valid_rewards = [
                float(reward)
                for reward, present in zip(rewards, has_task)
                if present
            ]
            reward_steps.append(
                float(np.mean(valid_rewards))
                if valid_rewards
                else float(np.mean(rewards))
            )
            latencies.extend(
                float(value)
                for value, present in zip(info.get("total_latencies", []), has_task)
                if present
            )
            energies.extend(
                float(value)
                for value, present in zip(info.get("total_energies", []), has_task)
                if present
            )
            global_state = next_global_state
            if done:
                break

        buffer.compute_advantages(
            gamma=float(algorithm_cfg.get("gamma", 0.99)),
            lam=float(algorithm_cfg.get("lam", 0.95)),
        )
        update_stats = []
        for index, agent in enumerate(agents):
            update_stats.append(agent.update(buffer.get_batch(index)))
        losses.append(
            {
                "episode": episode,
                **{
                    key: float(np.mean([item[key] for item in update_stats]))
                    for key in update_stats[0]
                },
            }
        )
        episodes.append(
            _episode_metrics(
                reward_steps,
                latencies,
                energies,
                info.get("task_completion_stats", {}),
            )
        )

    for index, agent in enumerate(agents):
        torch.save(
            {
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
            },
            run_dir / "models" / f"agent_{index}_final.pt",
        )
    return _finalize_run(
        run_dir,
        manifest,
        episodes,
        losses,
        extra={"continuous_edge_selector": True, "phase3_replacement_required": True},
    )


def run_baseline(
    algorithm: str,
    config: dict[str, Any],
    seed: int,
    output_root: str | Path,
    command: str | None = None,
) -> dict[str, Any]:
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"unsupported algorithm {algorithm!r}; choose from {SUPPORTED_ALGORITHMS}"
        )
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    command = command or " ".join(sys.argv)
    if algorithm in ("maddpg", "legacy_maps"):
        return _run_maddpg_family(algorithm, config, seed, output_root, command)
    return _run_on_policy(algorithm, config, seed, output_root, command)


def _compare_run_outputs(
    left_run_dir: str | Path,
    right_run_dir: str | Path,
    atol: float = 1e-7,
) -> dict[str, Any]:
    filenames = ("episode_metrics.json", "training_losses.json")
    max_abs_difference = 0.0

    def compare_values(left: Any, right: Any) -> bool:
        nonlocal max_abs_difference
        if isinstance(left, bool) or isinstance(right, bool):
            return left is right
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            difference = abs(float(left) - float(right))
            max_abs_difference = max(max_abs_difference, difference)
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(
                compare_values(left_item, right_item)
                for left_item, right_item in zip(left, right)
            )
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(
                compare_values(left[key], right[key]) for key in left
            )
        return left == right

    files_match = {}
    for filename in filenames:
        left = json.loads((Path(left_run_dir) / filename).read_text(encoding="utf-8"))
        right = json.loads((Path(right_run_dir) / filename).read_text(encoding="utf-8"))
        files_match[filename] = compare_values(left, right)

    return {
        "passed": all(files_match.values()),
        "absolute_tolerance": atol,
        "max_abs_difference": max_abs_difference,
        "files_match": files_match,
    }


def _run_reproducibility_probe(
    config: dict[str, Any],
    seed: int,
    output_root: str | Path,
    command: str | None,
) -> dict[str, Any]:
    probe_config = copy.deepcopy(config)
    probe_config["maddpg"].update(
        {
            "max_episodes": 2,
            "max_steps": 5,
            "batch_size": 2,
            "train_frequency": 1,
        }
    )
    probe_root = Path(output_root) / "_reproducibility"
    first = run_baseline("maddpg", probe_config, seed, probe_root, command)
    second = run_baseline("maddpg", probe_config, seed, probe_root, command)
    comparison = _compare_run_outputs(first["run_dir"], second["run_dir"])
    return {
        **comparison,
        "algorithm": "maddpg",
        "seed": seed,
        "episodes": 2,
        "steps_per_episode": 5,
        "run_dirs": [first["run_dir"], second["run_dir"]],
    }


def run_phase1_audit(
    config: dict[str, Any],
    seeds: list[int],
    output_root: str | Path,
    audit_path: str | Path,
    command: str | None = None,
) -> dict[str, Any]:
    results = []
    for seed in seeds:
        for algorithm in SUPPORTED_ALGORITHMS:
            try:
                result = run_baseline(
                    algorithm=algorithm,
                    config=config,
                    seed=seed,
                    output_root=output_root,
                    command=command,
                )
            except Exception as exc:
                result = {
                    "status": "failed",
                    "algorithm": algorithm,
                    "seed": seed,
                    "error": repr(exc),
                }
            result["algorithm"] = algorithm
            result["seed"] = seed
            results.append(result)

    by_algorithm = {
        algorithm: [
            result
            for result in results
            if result["algorithm"] == algorithm
        ]
        for algorithm in SUPPORTED_ALGORITHMS
    }
    try:
        reproducibility = _run_reproducibility_probe(
            config=config,
            seed=seeds[0],
            output_root=output_root,
            command=command,
        )
    except Exception as exc:
        reproducibility = {
            "passed": False,
            "algorithm": "maddpg",
            "seed": seeds[0],
            "error": repr(exc),
        }
    checks = {
        "all_algorithms_started": all(by_algorithm.values()),
        "all_runs_passed": all(result.get("status") == "passed" for result in results),
        "all_metrics_finite": all(
            result.get("finite_metrics", False)
            for result in results
            if result.get("status") == "passed"
        ),
        "joint_replay_enabled": all(
            result.get("extra", {}).get("joint_replay", False)
            for result in by_algorithm["maddpg"] + by_algorithm["legacy_maps"]
        ),
        "centralized_critic_enabled": all(
            result.get("extra", {}).get("centralized_critic", False)
            for result in by_algorithm["maddpg"] + by_algorithm["legacy_maps"]
        ),
        "legacy_expert_cache_present": all(
            result.get("extra", {}).get("expert", {}).get("cache_path")
            for result in by_algorithm["legacy_maps"]
        ),
        "same_seed_training_reproducible": reproducibility["passed"],
    }
    audit = {
        "schema_version": "1.0",
        "phase": "0-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "algorithms": list(SUPPORTED_ALGORITHMS),
        "checks": checks,
        "gate_1_passed": all(checks.values()),
        "runs": results,
        "reproducibility_probe": reproducibility,
        "limitations": [
            "The fixed expert cache is synthetic and is only for engineering validation.",
            "The Phase 1 environment still uses the legacy simplified physical model.",
            "MAPPO and HAPPO still use the legacy continuous edge selector until Phase 3.",
        ],
    }
    audit_path = Path(audit_path)
    if not audit_path.is_absolute():
        audit_path = PROJECT_ROOT / audit_path
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return audit
