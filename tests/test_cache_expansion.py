import json
from pathlib import Path

from LLM4RL.llm_assistant.expert_cache import ExpertCache, hash_state
from LLM4RL.scripts.expand_expert_cache_from_states import (
    expand_cache_from_records,
    load_missing_state_records,
)


def _metadata(request_id="req-cache-expansion"):
    return {
        "provider": "xiaomi_mimo",
        "requested_model": "mimo-v2.5",
        "returned_model": "mimo-v2.5",
        "temperature": 0.0,
        "request_id": request_id,
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }


def _raw_action():
    return json.dumps(
        {
            "schema_version": "ei-v1",
            "actions": [
                {
                    "ue_id": 0,
                    "partition": {
                        "local": 1.0,
                        "edge": 0.0,
                        "cloud": 0.0,
                    },
                    "edge_id": 0,
                    "reason_code": "unit",
                }
            ],
        }
    )


def _write_missing_states(path: Path, records):
    path.write_text(
        json.dumps(
            {
                "schema_version": "ei-missed-expert-states-v1",
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_load_missing_state_records_deduplicates_by_hash(tmp_path):
    state = {"state_key_version": "ei-state-v1", "value": [1, 2, 3]}
    record = {"state_hash": hash_state(state), "state": state}
    first = tmp_path / "missed_expert_states.json"
    second_dir = tmp_path / "nested"
    second_dir.mkdir()
    second = second_dir / "missed_expert_states.json"
    _write_missing_states(first, [record])
    _write_missing_states(second, [record])

    records = load_missing_state_records([tmp_path])

    assert records == [record]


def test_expand_cache_from_records_dry_run_counts_existing_and_missing(tmp_path):
    cached_state = {"state_key_version": "ei-state-v1", "value": "cached"}
    missing_state = {"state_key_version": "ei-state-v1", "value": "missing"}
    cache = ExpertCache("ei-v1", "xiaomi_mimo", "mimo-v2.5")
    cache.add_from_api_response(
        state_hash=hash_state(cached_state),
        raw_response=_raw_action(),
        num_devices=1,
        num_edges=1,
        llm_metadata=_metadata(),
        query_latency_ms=1.0,
    )
    cache_path = tmp_path / "expert_cache.json"
    cache.save(cache_path)

    summary = expand_cache_from_records(
        records=[
            {"state_hash": hash_state(cached_state), "state": cached_state},
            {"state_hash": hash_state(missing_state), "state": missing_state},
        ],
        cache_path=cache_path,
        llm_config_path="configs/ei/llm_mimo.yaml",
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert summary["input_state_count"] == 2
    assert summary["skipped_existing"] == 1
    assert summary["states_missing_from_cache"] == 1
    assert summary["states_selected_for_query"] == 1
    assert summary["added"] == 0
