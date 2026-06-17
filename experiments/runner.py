"""Unified Phase 1 baseline runners."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from LLM4RL.algos.common.annealing import AnnealingSchedule, FixedSchedule
from LLM4RL.algos.common.hybrid_action import HybridActionCodec
from LLM4RL.algos.common.trajectory_buffer import TrajectoryBuffer
from LLM4RL.algos.happo.happo_agent import HAPPOAgent
from LLM4RL.algos.maddpg.maddpg_agent import MADDPGAgent
from LLM4RL.algos.maddpg.replay_buffer import JointReplayBuffer
from LLM4RL.algos.mappo.mappo_agent import MAPPOAgent
from LLM4RL.baselines.greedy_min_cost import GreedyMinCostAgent
from LLM4RL.baselines.llm_only import LLMOnlyAgent
from LLM4RL.environment.cloud_edge_env import CloudEdgeDeviceEnv
from LLM4RL.llm_assistant.cached_expert_provider import CachedExpertProvider
from LLM4RL.llm_assistant.ei_prompt_builder import EIPromptBuilder
from LLM4RL.llm_assistant.ei_state import (
    build_ei_state_payload,
    ei_environment_fingerprint,
)
from LLM4RL.llm_assistant.expert_cache import ExpertCache, hash_state
from LLM4RL.llm_assistant.expert_provider import FixedCacheExpertProvider
from LLM4RL.utils.run_manifest import build_manifest, write_manifest
from LLM4RL.utils.seed import set_global_seed


SUPPORTED_ALGORITHMS = (
    "maddpg",
    "legacy_maps",
    "maps",
    "maps_no_annealing",
    "mappo",
    "happo",
    "greedy_min_cost",
    "llm_only",
)
PHASE1_ALGORITHMS = (
    "maddpg",
    "legacy_maps",
    "mappo",
    "happo",
    "greedy_min_cost",
)
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


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _make_maps_expert_provider(
    config: dict[str, Any],
    num_agents: int,
    num_edges: int,
):
    expert_cfg = copy.deepcopy(config.get("expert_cache", {}))
    llm_cfg = config.get("llm_maddpg", {})
    cache_format = str(
        expert_cfg.get(
            "format",
            llm_cfg.get("expert_cache_format", "fixed"),
        )
    )
    cache_path = _resolve_project_path(
        expert_cfg.get(
            "path",
            llm_cfg.get(
                "expert_cache", "fixtures/legacy_expert_cache.json"
            ),
        )
    )
    fallback_policy = str(
        expert_cfg.get("fallback_policy", "all_local")
    )
    if cache_format == "fixed":
        return FixedCacheExpertProvider(
            cache_path=cache_path,
            num_agents=num_agents,
            num_edges=num_edges,
        )
    if cache_format != "state_keyed":
        raise ValueError(
            "expert_cache.format must be 'fixed' or 'state_keyed'"
        )
    cache = ExpertCache.load(cache_path)
    if len(cache) == 0:
        raise ValueError("formal state-keyed cache must not be empty")
    if cache.provider != "xiaomi_mimo":
        raise ValueError("formal cache provider must be xiaomi_mimo")
    if cache.requested_model != "mimo-v2.5":
        raise ValueError("formal cache model must be mimo-v2.5")
    if cache.prompt_version != EIPromptBuilder.VERSION:
        raise ValueError("formal cache prompt_version must be ei-v1")
    if cache.state_key_version != ExpertCache.STATE_KEY_VERSION:
        raise ValueError("formal cache state_key_version is unsupported")
    if cache.temperature != 0.0:
        raise ValueError("formal cache temperature must be 0.0")
    return CachedExpertProvider(
        cache=cache,
        num_agents=num_agents,
        num_edges=num_edges,
        fallback_policy=fallback_policy,
    )


def _episode_metrics(
    reward_steps: list[float],
    latencies: list[float],
    energies: list[float],
    completion_stats: dict[str, Any],
    *,
    environment_steps: int | None = None,
    episode_index: int | None = None,
) -> dict[str, float]:
    metrics = {
        "reward": float(np.mean(reward_steps)) if reward_steps else 0.0,
        "latency": float(np.mean(latencies)) if latencies else 0.0,
        "energy": float(np.mean(energies)) if energies else 0.0,
        "completion_rate": float(
            completion_stats.get("on_time_completion_rate", 0.0)
        ),
    }
    if environment_steps is not None:
        metrics["environment_steps"] = int(environment_steps)
    if episode_index is not None:
        metrics["episode"] = int(episode_index)
    return metrics


def _convergence_metrics(
    episodes: list[dict[str, float]],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not episodes:
        return {
            "reward_auc": None,
            "final_reward": None,
            "steps_to_threshold": None,
            "threshold_status": "no_episodes",
        }
    rewards = np.asarray([float(item["reward"]) for item in episodes], dtype=np.float64)
    steps = np.asarray(
        [
            float(item.get("environment_steps", index + 1))
            for index, item in enumerate(episodes)
        ],
        dtype=np.float64,
    )
    if len(rewards) == 1 or steps[-1] <= steps[0]:
        reward_auc = float(rewards[-1])
    else:
        reward_auc = float(np.trapz(rewards, steps) / (steps[-1] - steps[0]))

    tail_count = max(1, int(math.ceil(0.1 * len(rewards))))
    final_reward = float(np.mean(rewards[-tail_count:]))

    evaluation_cfg = config.get("evaluation", {})
    threshold = evaluation_cfg.get("reward_threshold")
    patience = int(evaluation_cfg.get("threshold_patience", 5))
    if threshold is None:
        return {
            "reward_auc": reward_auc,
            "final_reward": final_reward,
            "steps_to_threshold": None,
            "threshold_status": "not_configured",
            "threshold": None,
            "threshold_patience": patience,
        }
    threshold = float(threshold)
    patience = max(1, patience)
    steps_to_threshold = None
    for index in range(0, len(rewards) - patience + 1):
        if np.all(rewards[index : index + patience] >= threshold):
            steps_to_threshold = int(steps[index])
            break
    return {
        "reward_auc": reward_auc,
        "final_reward": final_reward,
        "steps_to_threshold": steps_to_threshold,
        "threshold_status": (
            "reached" if steps_to_threshold is not None else "not_reached"
        ),
        "threshold": threshold,
        "threshold_patience": patience,
    }


C4_TASK_RECORD_FIELDS = [
    "task_id",
    "scenario_id",
    "arrival_time",
    "deadline",
    "completion_time",
    "latency",
    "completed",
    "completed_on_time",
    "device_energy_j",
    "system_energy_j",
    "decision_latency_ms",
    "ue_id",
    "arrival_slot",
    "task_type",
    "semantic_type",
    "priority",
    "data_size_mb",
    "cpu_cycles",
    "predicted_completion_time",
    "predicted_latency",
    "failure_reason",
]


def _finalize_c4_task_records(
    records: list[dict[str, Any]],
    *,
    evaluation_end_time: float,
    drain_end_time: float,
) -> list[dict[str, Any]]:
    finalized = []
    for record in records:
        if float(record["arrival_time"]) >= evaluation_end_time:
            continue
        output = copy.deepcopy(record)
        predicted_completion = output.get("completion_time")
        predicted_latency = output.get("latency")
        output["predicted_completion_time"] = predicted_completion
        output["predicted_latency"] = predicted_latency
        completed = (
            predicted_completion is not None
            and float(predicted_completion) <= drain_end_time + 1e-9
        )
        output["completed"] = bool(completed)
        if completed:
            latency = float(predicted_latency)
            output["completion_time"] = float(predicted_completion)
            output["latency"] = latency
            output["completed_on_time"] = (
                output["completion_time"]
                <= float(output["arrival_time"]) + float(output["deadline"]) + 1e-9
            )
        else:
            output["completion_time"] = None
            output["latency"] = None
            output["completed_on_time"] = False
            output["failure_reason"] = output.get("failure_reason") or (
                "not_completed_before_drain_horizon"
            )
        finalized.append(output)
    return finalized


def _percentile_or_none(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _c4_metrics(
    records: list[dict[str, Any]],
    reward_steps: list[float] | None = None,
) -> dict[str, Any]:
    n_gen = len(records)
    completed = [record for record in records if record["completed"]]
    on_time = [record for record in records if record["completed_on_time"]]
    latencies = [float(record["latency"]) for record in completed]
    system_energies = [float(record["system_energy_j"]) for record in completed]
    device_energies = [float(record["device_energy_j"]) for record in completed]
    decision_latencies = [
        float(record["decision_latency_ms"])
        for record in records
        if record.get("decision_latency_ms") is not None
    ]
    n_done = len(completed)
    n_on_time = len(on_time)
    metrics = {
        "N_gen": n_gen,
        "N_done": n_done,
        "N_on_time": n_on_time,
        "TCR": float(n_done / n_gen) if n_gen else 1.0,
        "DVR": float((n_gen - n_on_time) / n_gen) if n_gen else 0.0,
        "p95_task_latency": _percentile_or_none(latencies, 95),
        "mean_latency": _mean_or_none(latencies),
        "system_energy_per_completed_task": (
            float(np.sum(system_energies) / n_done) if n_done else None
        ),
        "device_energy_per_completed_task": (
            float(np.sum(device_energies) / n_done) if n_done else None
        ),
        "mean_decision_latency_ms": _mean_or_none(decision_latencies),
        "p95_decision_latency_ms": _percentile_or_none(decision_latencies, 95),
    }
    if reward_steps is not None:
        metrics["evaluation_reward"] = (
            float(np.mean(reward_steps)) if reward_steps else 0.0
        )
    return metrics


def _write_task_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted(
        {
            key
            for record in records
            for key in record
            if key not in C4_TASK_RECORD_FIELDS
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=C4_TASK_RECORD_FIELDS + extra_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)


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
    convergence = _convergence_metrics(episodes, manifest.get("config", {}))
    finite = all(math.isfinite(value) for value in metrics.values())
    result = {
        "status": "passed" if finite and episodes else "failed",
        "run_dir": str(run_dir),
        "episodes": len(episodes),
        "updates": len(losses),
        "metrics": metrics,
        "convergence_metrics": convergence,
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
            "convergence_metrics": convergence,
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

    env_config = copy.deepcopy(config)
    configured_max_steps = config.get("maddpg", {}).get("max_steps", 10)
    if algorithm == "legacy_maps":
        configured_max_steps = config.get("legacy_maps", {}).get(
            "max_steps", configured_max_steps
        )
    elif algorithm in ("maps_no_annealing", "maps"):
        configured_max_steps = config.get(algorithm, {}).get(
            "max_steps",
            config.get("llm_maddpg", {}).get(
                "max_steps", configured_max_steps
            ),
        )
    env_config.setdefault("maddpg", {})["max_steps"] = configured_max_steps
    env = CloudEdgeDeviceEnv(env_config)
    num_edges = env.num_edges
    codec = HybridActionCodec(num_edges)
    state_dim = env.get_agent_state_dim()
    # action_dim is now 3 + num_edges (hybrid action)
    action_dim = codec.action_dim
    num_agents = env.num_devices
    base_cfg = copy.deepcopy(config.get("maddpg", {}))
    algorithm_cfg = copy.deepcopy(base_cfg)
    expert_provider = None
    expert_metadata: dict[str, Any] = {}

    # --- Expert provider and distillation schedule ---
    schedule: AnnealingSchedule | FixedSchedule | None = None
    if algorithm == "maddpg":
        # The named MADDPG baseline is the no-distillation ablation.
        algorithm_cfg["distill_weight"] = 0.0
    elif algorithm == "legacy_maps":
        legacy_cfg = config.get("legacy_maps", {})
        algorithm_cfg.update(legacy_cfg)
        algorithm_cfg["distill_weight"] = float(legacy_cfg.get("distill_weight", 0.2))
        schedule = FixedSchedule(lambda_value=algorithm_cfg["distill_weight"])
        cache_path = Path(legacy_cfg["expert_cache"])
        if not cache_path.is_absolute():
            cache_path = PROJECT_ROOT / cache_path
        expert_provider = FixedCacheExpertProvider(
            cache_path=cache_path,
            num_agents=num_agents,
            num_edges=num_edges,
        )
        expert_metadata = expert_provider.metadata
    elif algorithm in ("maps_no_annealing", "maps"):
        llm_cfg = copy.deepcopy(config.get("llm_maddpg", {}))
        mode_cfg = copy.deepcopy(config.get(algorithm, {}))
        algorithm_cfg.update(llm_cfg)
        algorithm_cfg.update(mode_cfg)
        expert_provider = _make_maps_expert_provider(
            config=config,
            num_agents=num_agents,
            num_edges=num_edges,
        )
        if isinstance(expert_provider, FixedCacheExpertProvider):
            expert_metadata = expert_provider.metadata

    if algorithm == "maps_no_annealing":
        algorithm_cfg["distill_weight"] = float(
            algorithm_cfg.get("constant_llm_distill_weight", 0.15)
        )
        schedule = FixedSchedule(lambda_value=algorithm_cfg["distill_weight"])
    elif algorithm == "maps":
        schedule = AnnealingSchedule(
            lambda_high=float(
                algorithm_cfg.get("initial_llm_distill_weight", 0.8)
            ),
            lambda_low=float(
                algorithm_cfg.get("constant_llm_distill_weight", 0.15)
            ),
            stage1_end=float(algorithm_cfg.get("stage1_end_progress", 0.3)),
            stage2_end=float(algorithm_cfg.get("stage2_end_progress", 0.7)),
        )
        algorithm_cfg["distill_weight"] = schedule.lambda_high

    schedule_metadata = (
        schedule.to_dict()
        if schedule is not None
        else {
            "type": "disabled",
            "lambda_value": 0.0,
            "progress_unit": "environment_steps",
        }
    )

    agents = [
        MADDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=num_agents,
            agent_idx=index,
            num_edges=num_edges,
            config=algorithm_cfg,
        )
        for index in range(num_agents)
    ]
    replay = JointReplayBuffer(
        int(algorithm_cfg.get("buffer_size", 100000)),
        num_edges=num_edges,
    )
    episodes_count = int(
        algorithm_cfg.get(
            "max_episodes", config.get("training", {}).get("episodes", 20)
        )
    )
    max_steps = int(algorithm_cfg.get("max_steps", 10))
    env.max_steps = max_steps
    train_frequency = int(algorithm_cfg.get("train_frequency", 1))
    batch_size = int(algorithm_cfg.get("batch_size", 64))
    total_train_steps = max(1, episodes_count * max_steps)
    exploration_episodes = int(
        algorithm_cfg.get(
            "exploration_episodes",
            max(1, int(episodes_count * 0.7)),
        )
    )
    if exploration_episodes < 0:
        raise ValueError("exploration_episodes must be non-negative")

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
            # Policy actions are probabilities [3+E] per agent
            policy_actions = np.asarray(
                [
                    agent.select_action(
                        states[index],
                        add_noise=episode < exploration_episodes,
                    )
                    for index, agent in enumerate(agents)
                ],
                dtype=np.float32,
            )
            # Convert to env actions [a1, a2, a3, edge_id]
            env_actions = codec.batch_policy_to_env_actions(policy_actions)

            expert_actions = None
            expert_mask = None
            if expert_provider is not None:
                if isinstance(expert_provider, CachedExpertProvider):
                    expert_state = build_ei_state_payload(env)
                    if any(
                        task is not None for task in expert_state["tasks"]
                    ):
                        expert_batch = expert_provider.get_actions(
                            expert_state
                        )
                    else:
                        expert_batch = expert_provider.get_noop_actions(
                            expert_state
                        )
                else:
                    expert_batch = expert_provider.get_actions(
                        episode, step
                    )
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
                # Compute annealing lambda for this training step.
                progress = global_step / total_train_steps
                current_lambda = (
                    schedule.get_lambda(progress) if schedule is not None else 0.0
                )

                batch = replay.sample(batch_size)
                step_losses = [
                    agent.update(batch, agents, lambda_distill=current_lambda)
                    for agent in agents
                ]
                losses.append(
                    {
                        "global_step": global_step,
                        "progress": round(progress, 6),
                        "critic_loss": float(
                            np.mean([item["critic_loss"] for item in step_losses])
                        ),
                        "actor_loss": float(
                            np.mean([item["actor_loss"] for item in step_losses])
                        ),
                        "policy_loss": float(
                            np.mean([item["policy_loss"] for item in step_losses])
                        ),
                        "distill_loss": float(
                            np.mean([item["distill_loss"] for item in step_losses])
                        ),
                        "L_distill": float(
                            np.mean([item["L_distill"] for item in step_losses])
                        ),
                        "L_part": float(
                            np.mean([item["L_part"] for item in step_losses])
                        ),
                        "L_edge": float(
                            np.mean([item["L_edge"] for item in step_losses])
                        ),
                        "lambda_distill": float(current_lambda),
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
                environment_steps=global_step,
                episode_index=episode,
            )
        )

    final_progress = min(global_step / total_train_steps, 1.0)
    final_lambda = (
        schedule.get_lambda(final_progress) if schedule is not None else 0.0
    )
    guided_losses = [
        item for item in losses if item["lambda_distill"] > 0.0
    ]
    if guided_losses:
        mean_abs_policy_loss = float(
            np.mean([abs(item["policy_loss"]) for item in guided_losses])
        )
        mean_distill_loss = float(
            np.mean([item["L_distill"] for item in guided_losses])
        )
        mean_weighted_distill = float(
            np.mean(
                [
                    item["lambda_distill"] * item["L_distill"]
                    for item in guided_losses
                ]
            )
        )
        loss_scale = {
            "mean_abs_policy_loss": mean_abs_policy_loss,
            "mean_L_distill": mean_distill_loss,
            "mean_weighted_L_distill": mean_weighted_distill,
            "guided_updates": len(guided_losses),
            "distill_to_policy_ratio": (
                mean_distill_loss / mean_abs_policy_loss
                if mean_abs_policy_loss > 0.0
                else None
            ),
            "weighted_distill_to_policy_ratio": (
                mean_weighted_distill / mean_abs_policy_loss
                if mean_abs_policy_loss > 0.0
                else None
            ),
        }
    else:
        loss_scale = {}
    distillation_metadata = {
        "mode": algorithm,
        "eta_edge": agents[0].eta_edge,
        "schedule": schedule_metadata,
        "expert": expert_metadata,
        "global_step": global_step,
        "planned_environment_steps": total_train_steps,
        "exploration_episodes": exploration_episodes,
        "final_progress": final_progress,
        "final_lambda": final_lambda,
        "loss_scale": loss_scale,
    }
    manifest["distillation"] = distillation_metadata
    for index, agent in enumerate(agents):
        agent.save_model(
            run_dir / "models" / f"agent_{index}_final.pt",
            metadata={
                "algorithm": algorithm,
                "distillation": distillation_metadata,
            },
        )

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
            "hybrid_action": True,
            "num_edges": num_edges,
            "action_dim": action_dim,
            "annealing": isinstance(schedule, AnnealingSchedule),
            "schedule_type": (
                type(schedule).__name__ if schedule is not None else None
            ),
            "distillation": distillation_metadata,
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
    num_edges = env.num_edges
    codec = HybridActionCodec(num_edges)
    state_dim = env.get_agent_state_dim()
    global_state_dim = env.observation_space.shape[0]
    action_dim = codec.action_dim  # 3 + num_edges
    num_agents = env.num_devices
    algorithm_cfg = copy.deepcopy(config.get(algorithm, {}))
    agent_class = MAPPOAgent if algorithm == "mappo" else HAPPOAgent
    agents = [
        agent_class(
            state_dim=state_dim,
            action_dim=action_dim,
            global_state_dim=global_state_dim,
            agent_idx=index,
            num_edges=num_edges,
            config=algorithm_cfg,
        )
        for index in range(num_agents)
    ]
    episodes_count = int(algorithm_cfg.get("max_episodes", 20))
    max_steps = int(algorithm_cfg.get("max_steps", 10))
    episodes: list[dict[str, float]] = []
    losses: list[dict[str, float]] = []
    global_step = 0

    for episode in range(episodes_count):
        global_state, _ = env.reset(seed=seed + episode)
        buffer = TrajectoryBuffer(num_agents)
        reward_steps: list[float] = []
        latencies: list[float] = []
        energies: list[float] = []
        info: dict[str, Any] = {}

        for _ in range(max_steps):
            global_step += 1
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
            # Convert hybrid policy actions to env actions
            env_actions = codec.batch_policy_to_env_actions(policy_actions_array)
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
                    action_info["partition_params"],
                    action_info["edge_logits"],
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
                environment_steps=global_step,
                episode_index=episode,
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
        extra={
            "hybrid_action": True,
            "num_edges": num_edges,
            "action_dim": action_dim,
        },
    )


def _run_greedy_min_cost(
    config: dict[str, Any],
    seed: int,
    output_root: Path,
    command: str,
) -> dict[str, Any]:
    set_global_seed(seed)
    run_dir = _make_run_dir(output_root, "greedy_min_cost", seed)
    manifest = build_manifest(PROJECT_ROOT, "greedy_min_cost", seed, config, command)
    write_manifest(run_dir / "run_manifest.json", manifest)

    env = CloudEdgeDeviceEnv(config)
    num_edges = env.num_edges
    codec = HybridActionCodec(num_edges)
    reward_cfg = config.get("reward", {})
    agent = GreedyMinCostAgent(
        num_devices=env.num_devices,
        num_edges=num_edges,
        latency_weight=float(reward_cfg.get("latency_weight", 1.0)),
        energy_weight=float(reward_cfg.get("energy_weight", 1.0)),
    )

    episodes_count = int(
        config.get("greedy_min_cost", {}).get(
            "max_episodes", config.get("training", {}).get("episodes", 20)
        )
    )
    max_steps = int(
        config.get("greedy_min_cost", {}).get("max_steps", 10)
    )

    episodes: list[dict[str, float]] = []
    global_step = 0
    for episode in range(episodes_count):
        _global_state, _ = env.reset(seed=seed + episode)
        reward_steps: list[float] = []
        latencies: list[float] = []
        energies: list[float] = []
        info: dict[str, Any] = {}

        for _step in range(max_steps):
            global_step += 1
            # GreedyMinCost now returns [N, 3+E] hybrid actions
            hybrid_actions = agent.compute_action(env)
            env_actions = codec.batch_policy_to_env_actions(hybrid_actions)
            _obs, rewards, terminated, truncated, info = env.step(env_actions)
            done = bool(terminated or truncated)

            reward_steps.append(float(np.mean(rewards)))
            latencies.append(float(np.mean(info.get("total_latencies", [0.0]))))
            energies.append(float(np.mean(info.get("total_energies", [0.0]))))

            if done:
                break

        episodes.append(
            _episode_metrics(
                reward_steps,
                latencies,
                energies,
                info.get("task_completion_stats", {}),
                environment_steps=global_step,
                episode_index=episode,
            )
        )

    return _finalize_run(
        run_dir,
        manifest,
        episodes,
        losses=[],  # no training
        extra={"heuristic": True, "hybrid_action": True, "num_edges": num_edges},
    )


def _run_llm_only(
    config: dict[str, Any],
    seed: int,
    output_root: Path,
    command: str,
) -> dict[str, Any]:
    """Run the LLM-only evaluator (no training, cache lookup only)."""
    set_global_seed(seed)
    run_dir = _make_run_dir(output_root, "llm_only", seed)
    manifest = build_manifest(PROJECT_ROOT, "llm_only", seed, config, command)
    write_manifest(run_dir / "run_manifest.json", manifest)

    llm_only_cfg = config.get("llm_only", {})
    scenario_bank_path = llm_only_cfg.get("scenario_bank")
    default_max_steps = int(llm_only_cfg.get("max_steps", 10))
    if scenario_bank_path:
        scenario_bank_path = _resolve_project_path(scenario_bank_path)
        scenario_bank_bytes = scenario_bank_path.read_bytes()
        scenario_bank = json.loads(scenario_bank_bytes.decode("utf-8"))
        scenario_bank_id = str(scenario_bank["scenario_bank_id"])
        if not scenario_bank_id:
            raise ValueError("frozen scenario bank has no scenario_bank_id")
        raw_scenarios = scenario_bank.get("scenarios")
        if raw_scenarios is None:
            raw_scenarios = [
                {"seed": value, "max_steps": default_max_steps}
                for value in scenario_bank.get("seeds", [])
            ]
        scenarios = [
            {
                "seed": int(item["seed"]),
                "max_steps": int(
                    item.get("max_steps", default_max_steps)
                ),
            }
            for item in raw_scenarios
        ]
        if not scenarios:
            raise ValueError("frozen scenario bank has no seeds")
        if any(item["max_steps"] <= 0 for item in scenarios):
            raise ValueError("scenario max_steps must be positive")
        scenario_bank_frozen = True
        scenario_bank_sha256 = hashlib.sha256(
            scenario_bank_bytes
        ).hexdigest()
        construction_seed = int(
            scenario_bank.get("construction_seed", seed)
        )
    else:
        episodes_count = int(
            llm_only_cfg.get(
                "max_episodes",
                config.get("training", {}).get("episodes", 20),
            )
        )
        scenario_bank_id = None
        scenarios = [
            {
                "seed": seed + episode,
                "max_steps": default_max_steps,
            }
            for episode in range(episodes_count)
        ]
        scenario_bank_frozen = False
        scenario_bank_sha256 = None
        construction_seed = seed

    set_global_seed(construction_seed)
    env = CloudEdgeDeviceEnv(config)
    num_edges = env.num_edges
    num_agents = env.num_devices
    codec = HybridActionCodec(num_edges)
    provider = _make_maps_expert_provider(
        config=config,
        num_agents=num_agents,
        num_edges=num_edges,
    )
    if isinstance(provider, CachedExpertProvider):
        agent = LLMOnlyAgent(
            cache=provider.cache,
            num_agents=num_agents,
            num_edges=num_edges,
            fallback_policy=provider.fallback_policy,
        )
    else:
        agent = None
    if scenario_bank_frozen and isinstance(provider, CachedExpertProvider):
        expected_cache_sha256 = scenario_bank.get("cache_sha256")
        if (
            expected_cache_sha256
            and provider.cache.file_sha256 != expected_cache_sha256
        ):
            raise ValueError(
                "frozen scenario bank does not match the configured cache"
            )
        expected_environment = scenario_bank.get(
            "environment_fingerprint"
        )
        actual_environment = ei_environment_fingerprint(config)
        if (
            expected_environment
            and actual_environment != expected_environment
        ):
            raise ValueError(
                "frozen scenario bank environment does not match config"
            )
    require_full_cache_coverage = bool(
        llm_only_cfg.get("require_full_cache_coverage", True)
    )

    episodes: list[dict[str, float]] = []
    last_expert_metadata: dict[str, Any] = {}
    global_step = 0
    for scenario in scenarios:
        scenario_seed = scenario["seed"]
        max_steps = scenario["max_steps"]
        env.max_steps = max_steps
        _global_state, _ = env.reset(seed=scenario_seed)
        reward_steps: list[float] = []
        latencies: list[float] = []
        energies: list[float] = []
        info: dict[str, Any] = {}

        for step in range(max_steps):
            global_step += 1
            if agent is not None:
                state = build_ei_state_payload(env)
                if any(task is not None for task in state["tasks"]):
                    state_hash = hash_state(state)
                    if (
                        scenario_bank_frozen
                        and require_full_cache_coverage
                        and not provider.cache.has(state_hash)
                    ):
                        raise RuntimeError(
                            "frozen LLM-only scenario is not fully covered by "
                            f"the expert cache: {state_hash}"
                        )
                    env_actions, expert_batch = agent.select_env_actions(
                        state
                    )
                else:
                    expert_batch = agent.provider.get_noop_actions(state)
                    env_actions = agent.policy_to_env_actions(
                        expert_batch.policy_actions
                    )
            else:
                expert_batch = provider.get_actions(
                    len(episodes), step
                )
                env_actions = codec.batch_policy_to_env_actions(
                    expert_batch.policy_actions
                )
            last_expert_metadata = expert_batch.metadata
            _global_state, rewards, terminated, truncated, info = env.step(
                env_actions
            )
            done = bool(terminated or truncated)

            reward_steps.append(float(np.mean(rewards)))
            latencies.append(float(np.mean(info.get("total_latencies", [0.0]))))
            energies.append(float(np.mean(info.get("total_energies", [0.0]))))

            if done:
                break

        episodes.append(
            _episode_metrics(
                reward_steps,
                latencies,
                energies,
                info.get("task_completion_stats", {}),
                environment_steps=global_step,
                episode_index=len(episodes),
            )
        )

    return _finalize_run(
        run_dir,
        manifest,
        episodes,
        losses=[],
        extra={
            "llm_only": True,
            "online_api_calls": 0,
            "hybrid_action": True,
            "num_edges": num_edges,
            "cache_stats": (
                provider.cache.stats()
                if isinstance(provider, CachedExpertProvider)
                else {}
            ),
            "cache_runtime_stats": (
                agent.runtime_stats if agent is not None else {}
            ),
            "expert": last_expert_metadata,
            "scenario_bank_id": scenario_bank_id,
            "scenario_bank_frozen": scenario_bank_frozen,
            "scenario_bank_path": (
                str(scenario_bank_path)
                if scenario_bank_path
                else None
            ),
            "scenario_bank_sha256": scenario_bank_sha256,
            "scenario_count": len(scenarios),
            "scenario_construction_seed": construction_seed,
            "environment_fingerprint": ei_environment_fingerprint(config),
            "require_full_cache_coverage": require_full_cache_coverage,
            "cache_coverage_complete": (
                agent.runtime_stats["misses"] == 0
                if agent is not None
                else None
            ),
        },
    )


def _resolve_model_dir(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None
    path = _resolve_project_path(path_value)
    return path / "models" if (path / "models").is_dir() else path


def _load_maddpg_eval_agents(
    *,
    algorithm: str,
    config: dict[str, Any],
    env: CloudEdgeDeviceEnv,
    model_dir: Path | None,
    allow_untrained: bool,
) -> list[MADDPGAgent]:
    codec = HybridActionCodec(env.num_edges)
    base_cfg = copy.deepcopy(config.get("maddpg", {}))
    algorithm_cfg = copy.deepcopy(base_cfg)
    if algorithm == "legacy_maps":
        algorithm_cfg.update(copy.deepcopy(config.get("legacy_maps", {})))
    elif algorithm in ("maps_no_annealing", "maps"):
        algorithm_cfg.update(copy.deepcopy(config.get("llm_maddpg", {})))
        algorithm_cfg.update(copy.deepcopy(config.get(algorithm, {})))
    agents = [
        MADDPGAgent(
            state_dim=env.get_agent_state_dim(),
            action_dim=codec.action_dim,
            num_agents=env.num_devices,
            agent_idx=index,
            num_edges=env.num_edges,
            config=algorithm_cfg,
        )
        for index in range(env.num_devices)
    ]
    if model_dir is None:
        if not allow_untrained:
            raise ValueError(
                f"evaluation.checkpoint_dir is required for {algorithm}"
            )
        return agents
    for index, agent in enumerate(agents):
        agent.load_model(model_dir / f"agent_{index}_final.pt")
    return agents


def _load_on_policy_eval_agents(
    *,
    algorithm: str,
    config: dict[str, Any],
    env: CloudEdgeDeviceEnv,
    model_dir: Path | None,
    allow_untrained: bool,
):
    codec = HybridActionCodec(env.num_edges)
    agent_class = MAPPOAgent if algorithm == "mappo" else HAPPOAgent
    agents = [
        agent_class(
            state_dim=env.get_agent_state_dim(),
            action_dim=codec.action_dim,
            global_state_dim=env.observation_space.shape[0],
            agent_idx=index,
            num_edges=env.num_edges,
            config=copy.deepcopy(config.get(algorithm, {})),
        )
        for index in range(env.num_devices)
    ]
    if model_dir is None:
        if not allow_untrained:
            raise ValueError(
                f"evaluation.checkpoint_dir is required for {algorithm}"
            )
        return agents
    for index, agent in enumerate(agents):
        checkpoint = torch.load(
            model_dir / f"agent_{index}_final.pt",
            map_location=agent.device,
        )
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
    return agents


def _load_evaluation_scenarios(
    config: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluation_cfg = config.get("evaluation", {})
    default_window_steps = int(
        evaluation_cfg.get(
            "window_steps",
            config.get("testing", {}).get("max_steps", 10),
        )
    )
    if default_window_steps <= 0:
        raise ValueError("evaluation.window_steps must be positive")
    scenario_bank_path = evaluation_cfg.get("scenario_bank")
    if scenario_bank_path:
        scenario_bank_path = _resolve_project_path(scenario_bank_path)
        scenario_bank_bytes = scenario_bank_path.read_bytes()
        scenario_bank = json.loads(scenario_bank_bytes.decode("utf-8"))
        expected_environment = scenario_bank.get("environment_fingerprint")
        actual_environment = ei_environment_fingerprint(config)
        if expected_environment and actual_environment != expected_environment:
            raise ValueError(
                "evaluation scenario bank environment does not match config"
            )
        raw_scenarios = scenario_bank.get("scenarios")
        if raw_scenarios is None:
            raw_scenarios = [
                {"seed": value, "max_steps": default_window_steps}
                for value in scenario_bank.get("seeds", [])
            ]
        scenarios = [
            {
                "scenario_id": (
                    f"{scenario_bank['scenario_bank_id']}-{index}"
                ),
                "seed": int(item["seed"]),
                "window_steps": int(
                    item.get("max_steps", default_window_steps)
                ),
            }
            for index, item in enumerate(raw_scenarios)
        ]
        scenario_meta = {
            "scenario_bank_id": scenario_bank["scenario_bank_id"],
            "scenario_bank_path": str(scenario_bank_path),
            "scenario_bank_sha256": hashlib.sha256(
                scenario_bank_bytes
            ).hexdigest(),
            "scenario_bank_frozen": True,
            "scenario_construction_seed": int(
                scenario_bank.get("construction_seed", seed)
            ),
        }
    else:
        scenario_count = int(evaluation_cfg.get("num_scenarios", 1))
        if scenario_count <= 0:
            raise ValueError("evaluation.num_scenarios must be positive")
        scenarios = [
            {
                "scenario_id": f"seed-{seed + index}",
                "seed": seed + index,
                "window_steps": default_window_steps,
            }
            for index in range(scenario_count)
        ]
        scenario_meta = {
            "scenario_bank_id": None,
            "scenario_bank_path": None,
            "scenario_bank_sha256": None,
            "scenario_bank_frozen": False,
            "scenario_construction_seed": seed,
        }
    max_scenarios = evaluation_cfg.get("max_scenarios")
    if max_scenarios is not None:
        scenarios = scenarios[: int(max_scenarios)]
    if not scenarios:
        raise ValueError("evaluation has no scenarios")
    if any(item["window_steps"] <= 0 for item in scenarios):
        raise ValueError("evaluation scenario window_steps must be positive")
    return scenarios, scenario_meta


def run_evaluation(
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
    set_global_seed(seed)
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    command = command or " ".join(sys.argv)
    run_dir = _make_run_dir(output_root, f"{algorithm}_eval", seed)
    manifest = build_manifest(PROJECT_ROOT, algorithm, seed, config, command)
    manifest["phase"] = "formal_evaluation"
    write_manifest(run_dir / "run_manifest.json", manifest)

    evaluation_cfg = config.get("evaluation", {})
    drain_horizon_steps = int(evaluation_cfg.get("drain_horizon_steps", 0))
    if drain_horizon_steps < 0:
        raise ValueError("evaluation.drain_horizon_steps must be non-negative")
    allow_untrained = bool(evaluation_cfg.get("allow_untrained", False))
    model_dir = _resolve_model_dir(evaluation_cfg.get("checkpoint_dir"))
    scenarios, scenario_meta = _load_evaluation_scenarios(config, seed)

    env = CloudEdgeDeviceEnv(config)
    codec = HybridActionCodec(env.num_edges)
    greedy_agent = None
    llm_agent = None
    llm_provider = None
    learning_agents = None
    last_expert_metadata: dict[str, Any] = {}

    if algorithm == "greedy_min_cost":
        reward_cfg = config.get("reward", {})
        greedy_agent = GreedyMinCostAgent(
            num_devices=env.num_devices,
            num_edges=env.num_edges,
            latency_weight=float(reward_cfg.get("latency_weight", 1.0)),
            energy_weight=float(reward_cfg.get("energy_weight", 1.0)),
        )
    elif algorithm == "llm_only":
        llm_provider = _make_maps_expert_provider(
            config=config,
            num_agents=env.num_devices,
            num_edges=env.num_edges,
        )
        if isinstance(llm_provider, CachedExpertProvider):
            llm_agent = LLMOnlyAgent(
                cache=llm_provider.cache,
                num_agents=env.num_devices,
                num_edges=env.num_edges,
                fallback_policy=llm_provider.fallback_policy,
            )
        else:
            llm_agent = None
    elif algorithm in ("maddpg", "legacy_maps", "maps", "maps_no_annealing"):
        learning_agents = _load_maddpg_eval_agents(
            algorithm=algorithm,
            config=config,
            env=env,
            model_dir=model_dir,
            allow_untrained=allow_untrained,
        )
    else:
        learning_agents = _load_on_policy_eval_agents(
            algorithm=algorithm,
            config=config,
            env=env,
            model_dir=model_dir,
            allow_untrained=allow_untrained,
        )

    require_full_cache_coverage = bool(
        evaluation_cfg.get(
            "require_full_cache_coverage",
            config.get("llm_only", {}).get("require_full_cache_coverage", True),
        )
    )
    scenario_metrics: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    all_reward_steps: list[float] = []
    episodes: list[dict[str, float]] = []
    global_step = 0

    def select_env_actions(global_state):
        nonlocal last_expert_metadata
        if greedy_agent is not None:
            hybrid_actions = greedy_agent.compute_action(env)
            return codec.batch_policy_to_env_actions(hybrid_actions)
        if algorithm == "llm_only":
            if isinstance(llm_provider, CachedExpertProvider) and llm_agent is not None:
                state = build_ei_state_payload(env)
                if any(task is not None for task in state["tasks"]):
                    state_hash = hash_state(state)
                    if require_full_cache_coverage and not llm_provider.cache.has(
                        state_hash
                    ):
                        raise RuntimeError(
                            "formal LLM-only evaluation is not fully covered by "
                            f"the expert cache: {state_hash}"
                        )
                    env_actions, expert_batch = llm_agent.select_env_actions(state)
                else:
                    expert_batch = llm_agent.provider.get_noop_actions(state)
                    env_actions = llm_agent.policy_to_env_actions(
                        expert_batch.policy_actions
                    )
                last_expert_metadata = expert_batch.metadata
                return env_actions
            expert_batch = llm_provider.get_actions(
                len(scenario_metrics),
                env.episode_step,
            )
            last_expert_metadata = expert_batch.metadata
            return codec.batch_policy_to_env_actions(expert_batch.policy_actions)
        if algorithm in ("maddpg", "legacy_maps", "maps", "maps_no_annealing"):
            states = _local_states(env, global_state)
            policy_actions = np.asarray(
                [
                    agent.select_action(states[index], add_noise=False)
                    for index, agent in enumerate(learning_agents)
                ],
                dtype=np.float32,
            )
            return codec.batch_policy_to_env_actions(policy_actions)
        policy_actions = []
        for index, agent in enumerate(learning_agents):
            state = env.extract_agent_state(global_state, index)
            action, _action_info = agent.select_action(
                state,
                global_state,
                deterministic=True,
            )
            policy_actions.append(action)
        return codec.batch_policy_to_env_actions(
            np.asarray(policy_actions, dtype=np.float32)
        )

    for scenario_index, scenario in enumerate(scenarios):
        scenario_seed = int(scenario["seed"])
        window_steps = int(scenario["window_steps"])
        env.max_steps = window_steps + drain_horizon_steps
        global_state, _ = env.reset(
            seed=scenario_seed,
            options={
                "evaluation": True,
                "scenario_id": scenario["scenario_id"],
            },
        )
        reward_steps: list[float] = []
        for step in range(window_steps):
            if step == window_steps - 1:
                env.set_task_arrivals_enabled(False)
            started = time.perf_counter()
            env_actions = select_env_actions(global_state)
            decision_latency_ms = (time.perf_counter() - started) * 1000.0
            global_state, rewards, terminated, truncated, info = env.step(
                env_actions,
                decision_latency_ms=decision_latency_ms,
            )
            global_step += 1
            has_task = info.get("has_task_list", [True] * env.num_devices)
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
            if terminated or truncated:
                break

        evaluation_end_time = float(env.global_time)
        drain_end_time = evaluation_end_time + (
            drain_horizon_steps * float(env.time_step_duration)
        )
        drain_steps_used = 0
        for _drain_step in range(drain_horizon_steps):
            if not env.has_pending_arrivals():
                break
            started = time.perf_counter()
            env_actions = select_env_actions(global_state)
            decision_latency_ms = (time.perf_counter() - started) * 1000.0
            global_state, _rewards, terminated, truncated, _info = env.step(
                env_actions,
                decision_latency_ms=decision_latency_ms,
            )
            global_step += 1
            drain_steps_used += 1
            if terminated or truncated:
                break

        records = _finalize_c4_task_records(
            env.get_task_records(),
            evaluation_end_time=evaluation_end_time,
            drain_end_time=drain_end_time,
        )
        metrics = _c4_metrics(records, reward_steps)
        metrics.update(
            {
                "scenario_id": scenario["scenario_id"],
                "scenario_index": scenario_index,
                "seed": scenario_seed,
                "window_steps": window_steps,
                "drain_horizon_steps": drain_horizon_steps,
                "drain_steps_used": drain_steps_used,
                "evaluation_end_time": evaluation_end_time,
                "drain_end_time": drain_end_time,
            }
        )
        scenario_metrics.append(metrics)
        all_records.extend(records)
        all_reward_steps.extend(reward_steps)
        episodes.append(
            {
                "episode": scenario_index,
                "environment_steps": global_step,
                "reward": float(metrics.get("evaluation_reward") or 0.0),
                "latency": float(metrics.get("mean_latency") or 0.0),
                "energy": float(
                    metrics.get("system_energy_per_completed_task") or 0.0
                ),
                "completion_rate": float(metrics["TCR"]),
            }
        )

    task_records_path = run_dir / "task_records.csv"
    evaluation_metrics_path = run_dir / "evaluation_metrics.json"
    _write_task_records_csv(task_records_path, all_records)
    overall_metrics = _c4_metrics(all_records, all_reward_steps)
    evaluation_payload = {
        "schema_version": "c4-evaluation-v1",
        "algorithm": algorithm,
        "seed": seed,
        "metrics": overall_metrics,
        "scenarios": scenario_metrics,
        "scenario_count": len(scenarios),
        "task_records_path": str(task_records_path),
        "drain_horizon_steps": drain_horizon_steps,
        "artifacts": {
            "task_records_csv": str(task_records_path),
            "evaluation_metrics_json": str(evaluation_metrics_path),
        },
        **scenario_meta,
    }
    evaluation_metrics_path.write_text(
        json.dumps(evaluation_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    extra = {
        "formal_evaluation": True,
        "evaluation_metrics": overall_metrics,
        "evaluation_metrics_path": str(evaluation_metrics_path),
        "task_records_path": str(task_records_path),
        "task_record_count": len(all_records),
        "drain_horizon_steps": drain_horizon_steps,
        "exploration_enabled": False,
        "llm_enabled": algorithm == "llm_only",
        "online_api_calls": 0,
        "maps_eval_uses_llm": False
        if algorithm in ("maps", "maps_no_annealing")
        else None,
        "checkpoint_dir": str(model_dir) if model_dir else None,
        "allow_untrained": allow_untrained,
        "expert": last_expert_metadata,
        **scenario_meta,
    }
    if isinstance(llm_provider, CachedExpertProvider):
        extra["cache_runtime_stats"] = (
            llm_agent.runtime_stats if llm_agent is not None else {}
        )
        extra["cache_stats"] = llm_provider.cache.stats()
        extra["cache_coverage_complete"] = (
            extra["cache_runtime_stats"].get("misses") == 0
            if llm_agent is not None
            else None
        )

    return _finalize_run(
        run_dir,
        manifest,
        episodes,
        losses=[],
        extra=extra,
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
    if algorithm == "greedy_min_cost":
        return _run_greedy_min_cost(config, seed, output_root, command)
    if algorithm == "llm_only":
        return _run_llm_only(config, seed, output_root, command)
    if algorithm in ("maddpg", "legacy_maps", "maps", "maps_no_annealing"):
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
        for algorithm in PHASE1_ALGORITHMS:
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
        for algorithm in PHASE1_ALGORITHMS
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
        "algorithms": list(PHASE1_ALGORITHMS),
        "checks": checks,
        "gate_1_passed": all(checks.values()),
        "runs": results,
        "reproducibility_probe": reproducibility,
        "limitations": [
            "The fixed expert cache is synthetic and is only for engineering validation.",
            "The Phase 1 environment still uses the legacy simplified physical model.",
            "Hybrid action codec is now active for all algorithms.",
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
