"""C3 tests for the formal state-keyed LLM evidence chain."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from LLM4RL.baselines.llm_only import LLMOnlyAgent
from LLM4RL.environment.cloud_edge_env import CloudEdgeDeviceEnv
from LLM4RL.experiments.runner import run_baseline
from LLM4RL.llm_assistant.cached_expert_provider import CachedExpertProvider
from LLM4RL.llm_assistant.ei_prompt_builder import EIPromptBuilder
from LLM4RL.llm_assistant.ei_state import (
    build_ei_state_payload,
    ei_environment_fingerprint,
)
from LLM4RL.llm_assistant.expert_cache import ExpertCache, hash_state
from LLM4RL.utils.config import load_config
from LLM4RL.utils.seed import set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def preserve_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)


def _metadata(request_id: str = "req-123") -> dict:
    return {
        "provider": "xiaomi_mimo",
        "requested_model": "mimo-v2.5",
        "returned_model": "mimo-v2.5",
        "temperature": 0.0,
        "request_id": request_id,
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    }


def _raw_actions(include_second: bool = True) -> str:
    actions = [
        {
            "ue_id": 0,
            "partition": {"local": 0.2, "edge": 0.6, "cloud": 0.2},
            "edge_id": 1,
            "reason_code": "latency_sensitive",
        }
    ]
    if include_second:
        actions.append(
            {
                "ue_id": 1,
                "partition": {
                    "local": 0.3,
                    "edge": 0.4,
                    "cloud": 0.3,
                },
                "edge_id": 0,
                "reason_code": "balanced",
            }
        )
    return json.dumps({"schema_version": "ei-v1", "actions": actions})


def _cache_with_state(state, include_second: bool = True) -> ExpertCache:
    cache = ExpertCache(
        "ei-v1", "xiaomi_mimo", "mimo-v2.5", temperature=0.0
    )
    cache.add_from_api_response(
        state_hash=hash_state(state),
        raw_response=_raw_actions(include_second),
        num_devices=2,
        num_edges=3,
        llm_metadata=_metadata(),
        query_latency_ms=125.0,
    )
    return cache


class TestHashState:
    def test_is_stable_full_sha256(self):
        state = {"b": [0.1, 0.2], "a": {"x": 1}}
        assert hash_state(state) == hash_state(
            {"a": {"x": 1}, "b": np.array([0.1, 0.2])}
        )
        assert len(hash_state(state)) == 64

    def test_preserves_joint_shape(self):
        assert hash_state([[1.0, 2.0]]) != hash_state([1.0, 2.0])

    def test_rounding_is_stable(self):
        assert hash_state([0.12345600000001]) == hash_state(
            [0.12345600000002]
        )
        assert hash_state([0.12345]) != hash_state([0.12346])
        assert hash_state([1e-9]) != hash_state([2e-9])

    def test_rejects_non_finite_state(self):
        with pytest.raises(ValueError):
            hash_state([float("nan")])


class TestPromptAndState:
    def test_prompt_contains_all_required_groups(self):
        prompt = EIPromptBuilder().build(
            ue_info=[
                {
                    "device_id": 0,
                    "cpu_frequency": 0.5,
                    "pending_queue_size": 1,
                    "ue_to_edge_rates": [1e6, 2e6],
                }
            ],
            es_info=[
                {
                    "server_id": 0,
                    "cpu_frequency": 5.0,
                    "queue_load": 10.0,
                }
            ],
            cs_info=[
                {
                    "server_id": 0,
                    "cpu_frequency": 20.0,
                    "queue_load": 0.0,
                    "parallel_factor": 8.0,
                }
            ],
            tasks_info=[
                {
                    "task_id": "task-1",
                    "device_id": 0,
                    "data_size": 100.0,
                    "cpu_cycles": 2e10,
                    "deadline_slack": 5.0,
                    "semantic_type": "control",
                    "priority": 3,
                    "output_ratio": 0.1,
                }
            ],
            backhaul_info={
                "rate_bps": 1e9,
                "energy_per_bit": 1e-9,
                "propagation_latency_s": 0.002,
            },
            num_edges=2,
        )
        for required in (
            "pending_queue_size",
            "queue_load",
            "ue_to_edge_rates",
            "queue_load",
            "backhaul",
            "deadline_slack",
            "semantic_type",
            "priority",
            "output_ratio",
            "partition_ratios",
            "edge_id_range",
            "Return JSON only",
        ):
            assert required in prompt

    def test_environment_state_hash_reproduces_for_same_seed(self):
        config = load_config(str(PROJECT_ROOT / "configs" / "smoke_phase2.yaml"))
        set_global_seed(42)
        first = CloudEdgeDeviceEnv(copy.deepcopy(config))
        first.reset(seed=42)
        first_state = build_ei_state_payload(first)
        set_global_seed(42)
        second = CloudEdgeDeviceEnv(copy.deepcopy(config))
        second.reset(seed=42)
        second_state = build_ei_state_payload(second)
        assert hash_state(first_state) == hash_state(second_state)
        assert first_state["state_key_version"] == "ei-state-v1"
        assert "backhaul" in first_state


class TestExpertCache:
    def test_valid_entry_contains_required_audit_fields(self):
        cache = _cache_with_state({"joint": [1, 2]})
        entry = cache.get(hash_state({"joint": [1, 2]}))
        assert entry is not None
        payload = entry.__dict__
        required = {
            "state_hash",
            "prompt_version",
            "provider",
            "requested_model",
            "returned_model",
            "temperature",
            "raw_response_hash",
            "num_edges",
            "parsed_action",
            "decision_mask",
            "parser_status",
            "fallback_status",
            "request_id",
            "token_usage",
            "query_latency_ms",
            "generation_time",
            "paper_evidence",
        }
        assert required.issubset(payload)
        assert len(entry.raw_response_hash) == 64
        assert entry.valid_mask == [True, True]
        assert entry.decision_mask == [True, True]

    def test_partial_response_preserves_per_ue_mask(self):
        cache = _cache_with_state({"joint": [1, 2]}, include_second=False)
        entry = cache.get(hash_state({"joint": [1, 2]}))
        assert entry.parser_status == "partial"
        assert entry.fallback_status == "partial_fallback"
        assert entry.valid_mask == [True, False]

    def test_failed_response_is_audited_as_full_fallback(self):
        cache = ExpertCache("ei-v1", "xiaomi_mimo", "mimo-v2.5")
        entry = cache.add_from_api_response(
            state_hash=hash_state({"state": "bad"}),
            raw_response="not json",
            num_devices=2,
            num_edges=3,
            llm_metadata=_metadata(),
            query_latency_ms=20.0,
        )
        assert entry.parser_status == "failed"
        assert entry.valid_mask == [False, False]

    def test_formal_parser_rejects_constraint_repairs(self):
        cache = ExpertCache("ei-v1", "xiaomi_mimo", "mimo-v2.5")
        raw = json.dumps(
            {
                "actions": [
                    {
                        "ue_id": 0,
                        "partition": {
                            "local": 2,
                            "edge": 1,
                            "cloud": 1,
                        },
                        "edge_id": 9,
                    }
                ]
            }
        )
        entry = cache.add_from_api_response(
            state_hash=hash_state({"strict": True}),
            raw_response=raw,
            num_devices=1,
            num_edges=3,
            llm_metadata=_metadata(),
            query_latency_ms=1.0,
        )
        assert entry.parser_status == "failed"
        assert entry.valid_mask == [False]

    def test_non_decision_ues_are_excluded_from_validity_rates(self):
        cache = ExpertCache("ei-v1", "xiaomi_mimo", "mimo-v2.5")
        entry = cache.add_from_api_response(
            state_hash=hash_state({"one-task": True}),
            raw_response=_raw_actions(include_second=False),
            num_devices=2,
            num_edges=3,
            llm_metadata=_metadata(),
            query_latency_ms=1.0,
            decision_mask=[True, False],
        )
        assert entry.parser_status == "valid"
        assert entry.decision_mask == [True, False]
        assert entry.valid_mask == [True, False]
        assert cache.stats()["valid_action_rate"] == 1.0

    def test_rejects_missing_request_id_and_model_mismatch(self):
        cache = ExpertCache("ei-v1", "xiaomi_mimo", "mimo-v2.5")
        with pytest.raises(ValueError, match="request_id"):
            cache.add_from_api_response(
                state_hash="a",
                raw_response=_raw_actions(),
                num_devices=2,
                num_edges=3,
                llm_metadata=_metadata(request_id=""),
                query_latency_ms=1.0,
            )
        bad_metadata = _metadata()
        bad_metadata["returned_model"] = "other-model"
        with pytest.raises(ValueError, match="returned_model"):
            cache.add_from_api_response(
                state_hash="b",
                raw_response=_raw_actions(),
                num_devices=2,
                num_edges=3,
                llm_metadata=bad_metadata,
                query_latency_ms=1.0,
            )

    def test_save_load_is_deterministic_and_validated(self, tmp_path):
        cache = _cache_with_state({"joint": [1, 2]})
        path = tmp_path / "cache.json"
        cache.save(path)
        first_bytes = path.read_bytes()
        loaded = ExpertCache.load(path)
        loaded.save(path)
        assert path.read_bytes() == first_bytes
        assert loaded.file_sha256 is not None
        assert loaded.source_path == str(path.resolve())

        tampered = json.loads(path.read_text(encoding="utf-8"))
        only_entry = next(iter(tampered["entries"].values()))
        only_entry["raw_response"] = "tampered"
        path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(ValueError, match="raw_response_hash"):
            ExpertCache.load(path)

        cache.save(path)
        tampered = json.loads(path.read_text(encoding="utf-8"))
        only_entry = next(iter(tampered["entries"].values()))
        only_entry["parsed_action"][0]["edge_id"] = 0
        path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(ValueError, match="parsed_action"):
            ExpertCache.load(path)

    def test_stats_include_parser_action_and_fallback_rates(self):
        cache = ExpertCache("ei-v1", "xiaomi_mimo", "mimo-v2.5")
        cache.add_from_api_response(
            state_hash=hash_state({"status": "valid"}),
            raw_response=_raw_actions(),
            num_devices=2,
            num_edges=3,
            llm_metadata=_metadata("req-valid"),
            query_latency_ms=10.0,
        )
        cache.add_from_api_response(
            state_hash=hash_state({"status": "partial"}),
            raw_response=_raw_actions(include_second=False),
            num_devices=2,
            num_edges=3,
            llm_metadata=_metadata("req-partial"),
            query_latency_ms=20.0,
        )
        stats = cache.stats()
        assert stats["parser_success_rate"] == 1.0
        assert stats["valid_action_rate"] == pytest.approx(0.75)
        assert stats["fallback_rate"] == pytest.approx(0.25)


class TestCachedProvider:
    def test_joint_lookup_returns_each_ue_own_action(self):
        state = {
            "ue_states": [
                {"device_id": 0, "local": [0.1, 0.2]},
                {"device_id": 1, "local": [0.9, 0.8]},
            ]
        }
        provider = CachedExpertProvider(
            _cache_with_state(state),
            num_agents=2,
            num_edges=3,
        )
        batch = provider.get_actions(state)
        assert batch.valid_mask.tolist() == [1.0, 1.0]
        np.testing.assert_allclose(
            batch.policy_actions[0, :3], [0.2, 0.6, 0.2]
        )
        np.testing.assert_allclose(
            batch.policy_actions[1, :3], [0.3, 0.4, 0.3]
        )
        assert batch.metadata["cache_hit"] is True

    def test_partial_entry_masks_only_fallback_ue(self):
        state = {"joint": [1, 2]}
        provider = CachedExpertProvider(
            _cache_with_state(state, include_second=False),
            num_agents=2,
            num_edges=3,
        )
        batch = provider.get_actions(state)
        assert batch.valid_mask.tolist() == [1.0, 0.0]
        np.testing.assert_allclose(
            batch.policy_actions[1, :3], [1.0, 0.0, 0.0]
        )

    def test_missing_joint_state_records_fallback(self):
        provider = CachedExpertProvider(
            ExpertCache("ei-v1", "xiaomi_mimo", "mimo-v2.5"),
            num_agents=2,
            num_edges=3,
        )
        missing_state = {"missing": True}
        batch = provider.get_actions(missing_state)
        assert batch.valid_mask.tolist() == [0.0, 0.0]
        assert batch.metadata["runtime_stats"]["fallback_rate"] == 1.0
        assert batch.metadata["runtime_stats"]["unique_misses"] == 1
        assert batch.metadata["paper_evidence"] is False
        assert provider.missing_state_records() == [
            {
                "state_hash": hash_state(missing_state),
                "state": missing_state,
            }
        ]

    def test_noop_state_does_not_count_as_cache_miss(self):
        provider = CachedExpertProvider(
            ExpertCache("ei-v1", "xiaomi_mimo", "mimo-v2.5"),
            num_agents=2,
            num_edges=3,
        )
        batch = provider.get_noop_actions({"tasks": [None, None]})
        assert batch.metadata["no_decision"] is True
        assert provider.runtime_stats()["lookups"] == 0

    def test_non_decision_ue_does_not_count_as_runtime_fallback(self):
        cache = ExpertCache("ei-v1", "xiaomi_mimo", "mimo-v2.5")
        state = {"one-task": True}
        cache.add_from_api_response(
            state_hash=hash_state(state),
            raw_response=_raw_actions(include_second=False),
            num_devices=2,
            num_edges=3,
            llm_metadata=_metadata(),
            query_latency_ms=1.0,
            decision_mask=[True, False],
        )
        provider = CachedExpertProvider(cache, num_agents=2, num_edges=3)
        batch = provider.get_actions(state)
        assert batch.valid_mask.tolist() == [1.0, 0.0]
        assert provider.runtime_stats()["decision_actions"] == 1
        assert provider.runtime_stats()["fallback_actions"] == 0

    def test_live_fill_on_miss_writes_cache_entry(self, tmp_path):
        class FakeClient:
            def __init__(self):
                self.calls = 0
                self.last_metadata = {}

            def query(self, _prompt):
                self.calls += 1
                self.last_metadata = _metadata("req-live-fill")
                return _raw_actions(include_second=False)

        state = {
            "state_key_version": "ei-state-v1",
            "num_edges": 2,
            "ue_states": [
                {
                    "device_id": 0,
                    "cpu_frequency": 0.5,
                    "pending_queue_size": 0,
                    "queue_load": 0.0,
                    "ue_to_edge_rates": [1.0, 1.0],
                }
            ],
            "es_states": [
                {"server_id": 0, "cpu_frequency": 5.0, "queue_load": 0.0},
                {"server_id": 1, "cpu_frequency": 6.0, "queue_load": 0.0},
            ],
            "cs_states": [
                {
                    "server_id": 0,
                    "cpu_frequency": 20.0,
                    "queue_load": 0.0,
                    "parallel_factor": 8.0,
                }
            ],
            "backhaul": {
                "rate_bps": 1e9,
                "energy_per_bit": 1e-9,
                "propagation_latency_s": 0.002,
            },
            "tasks": [
                {
                    "task_id": "task-0",
                    "device_id": 0,
                    "data_size": 100.0,
                    "cpu_cycles": 1000.0,
                    "deadline_slack": 1.0,
                    "semantic_type": "control",
                    "priority": 1,
                    "output_ratio": 0.1,
                }
            ],
        }
        cache = ExpertCache("ei-v1", "xiaomi_mimo", "mimo-v2.5")
        output_path = tmp_path / "filled-cache.json"
        client = FakeClient()
        provider = CachedExpertProvider(
            cache,
            num_agents=1,
            num_edges=2,
            live_client=client,
            cache_output_path=output_path,
        )

        first = provider.get_actions(state)
        second = provider.get_actions(state)

        assert first.valid_mask.tolist() == [1.0]
        assert second.valid_mask.tolist() == [1.0]
        assert client.calls == 1
        assert output_path.exists()
        assert cache.has(hash_state(state))
        assert provider.runtime_stats()["misses"] == 1
        assert provider.runtime_stats()["hits"] == 1
        assert provider.runtime_stats()["live_fills"] == 1
        assert provider.runtime_stats()["fallback_actions"] == 0


class TestLLMOnly:
    def test_agent_uses_joint_cache_without_online_query(self):
        state = {"joint": [1, 2]}
        agent = LLMOnlyAgent(
            _cache_with_state(state),
            num_agents=2,
            num_edges=3,
        )
        batch = agent.select_joint_actions(state)
        env_actions = agent.policy_to_env_actions(batch.policy_actions)
        assert env_actions.shape == (2, 4)
        assert env_actions[0, 3] == 1
        assert env_actions[1, 3] == 0

    def test_agent_can_execute_raw_cache_env_actions(self):
        state = {"joint": [1, 2]}
        agent = LLMOnlyAgent(
            _cache_with_state(state),
            num_agents=2,
            num_edges=3,
        )
        env_actions, batch = agent.select_env_actions(state)
        assert batch.metadata["cache_hit"] is True
        assert env_actions.dtype == np.float64
        np.testing.assert_allclose(env_actions[0], [0.2, 0.6, 0.2, 1.0])


def _formal_runner_config(tmp_path: Path) -> dict:
    config = load_config(str(PROJECT_ROOT / "configs" / "smoke_phase2.yaml"))
    config["maddpg"].update(
        {
            "max_episodes": 1,
            "max_steps": 1,
            "batch_size": 1,
            "train_frequency": 1,
        }
    )
    config["llm_maddpg"].update(
        {
            "max_episodes": 1,
            "max_steps": 1,
            "batch_size": 1,
            "train_frequency": 1,
        }
    )
    set_global_seed(42)
    env = CloudEdgeDeviceEnv(copy.deepcopy(config))
    env.reset(seed=42)
    state = build_ei_state_payload(env)
    cache = ExpertCache("ei-v1", "xiaomi_mimo", "mimo-v2.5")
    actions = []
    for agent_idx in range(env.num_devices):
        actions.append(
            {
                "ue_id": agent_idx,
                "partition": {
                    "local": 0.3,
                    "edge": 0.4,
                    "cloud": 0.3,
                },
                "edge_id": agent_idx % env.num_edges,
                "reason_code": "test",
            }
        )
    cache.add_from_api_response(
        state_hash=hash_state(state),
        raw_response=json.dumps({"actions": actions}),
        num_devices=env.num_devices,
        num_edges=env.num_edges,
        llm_metadata=_metadata(),
        query_latency_ms=10.0,
    )
    cache_path = tmp_path / "formal-cache.json"
    cache.save(cache_path)
    scenario_path = tmp_path / "scenario-bank.json"
    scenario_path.write_text(
        json.dumps(
            {
                "scenario_bank_id": "test-bank-v1",
                "seeds": [42],
                "construction_seed": 42,
                "cache_sha256": cache.file_sha256,
                "environment_fingerprint": ei_environment_fingerprint(
                    config
                ),
            }
        ),
        encoding="utf-8",
    )
    config["expert_cache"] = {
        "format": "state_keyed",
        "path": str(cache_path),
        "fallback_policy": "all_local",
    }
    config["llm_only"] = {
        "max_steps": 1,
        "scenario_bank": str(scenario_path),
    }
    return config


def test_three_methods_share_one_state_keyed_cache(tmp_path):
    config = _formal_runner_config(tmp_path)
    results = {
        algorithm: run_baseline(
            algorithm,
            copy.deepcopy(config),
            42,
            tmp_path / algorithm,
        )
        for algorithm in ("maps_no_annealing", "maps", "llm_only")
    }

    cache_paths = set()
    for algorithm, result in results.items():
        manifest = json.loads(
            Path(result["run_dir"], "run_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        expert = manifest["summary"]["extra"]["expert"]
        assert expert["cache_hit"] is True
        assert expert["paper_evidence"] is True
        cache_paths.add(expert["cache_path"])
        if algorithm == "llm_only":
            assert manifest["summary"]["extra"]["online_api_calls"] == 0
            assert manifest["summary"]["extra"]["scenario_bank_frozen"] is True
            assert manifest["summary"]["extra"]["scenario_bank_sha256"]
            assert manifest["summary"]["extra"]["cache_coverage_complete"] is True
            assert (
                manifest["summary"]["extra"]["scenario_construction_seed"]
                == 42
            )
            assert (
                manifest["summary"]["extra"]["scenario_bank_id"]
                == "test-bank-v1"
            )
    assert len(cache_paths) == 1
