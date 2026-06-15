import json

import numpy as np

from LLM4RL.llm_assistant.expert_provider import FixedCacheExpertProvider
from LLM4RL.llm_assistant.response_parser import ResponseParser


def test_parser_accepts_markdown_json_and_normalizes_ratios():
    response = """```json
    {"strategies": [
      {"device_id": 0, "local_ratio": 2, "edge_ratio": 1,
       "cloud_ratio": 1, "target_edge_server": 9}
    ]}
    ```"""
    strategies, metadata = ResponseParser.parse_with_metadata(response, 1, 3)

    assert metadata.valid
    assert np.isclose(
        strategies[0]["local_ratio"]
        + strategies[0]["edge_ratio"]
        + strategies[0]["cloud_ratio"],
        1.0,
    )
    assert strategies[0]["target_edge"] == 2


def test_fixed_cache_is_explicitly_not_paper_evidence(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "cache_version": "test-cache-v1",
                "prompt_version": "test-prompt-v1",
                "source": "test",
                "paper_evidence": False,
                "default": {
                    "strategies": [
                        {
                            "device_id": 0,
                            "local_ratio": 1,
                            "edge_ratio": 0,
                            "cloud_ratio": 0,
                            "target_edge": 0,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    provider = FixedCacheExpertProvider(path, num_agents=1, num_edges=2)

    batch = provider.get_actions(0, 0)

    assert batch.metadata["paper_evidence"] is False
    assert batch.valid_mask.tolist() == [1.0]
    assert batch.metadata["cache_version"] == "test-cache-v1"
    assert batch.metadata["prompt_version"] == "test-prompt-v1"
    assert len(batch.metadata["cache_sha256"]) == 64


def test_partial_expert_response_masks_only_missing_devices(tmp_path):
    path = tmp_path / "partial-cache.json"
    path.write_text(
        json.dumps(
            {
                "default": {
                    "strategies": [
                        {
                            "device_id": 0,
                            "local_ratio": 0.2,
                            "edge_ratio": 0.5,
                            "cloud_ratio": 0.3,
                            "target_edge": 1,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    batch = FixedCacheExpertProvider(
        path, num_agents=2, num_edges=2
    ).get_actions(0, 0)

    assert batch.valid_mask.tolist() == [1.0, 0.0]
    assert batch.metadata["valid_mask"] == [True, False]
    assert batch.metadata["fallback_count"] == 1


def test_missing_expert_field_marks_only_that_device_invalid(tmp_path):
    path = tmp_path / "missing-field-cache.json"
    path.write_text(
        json.dumps(
            {
                "default": {
                    "strategies": [
                        {
                            "device_id": 0,
                            "local_ratio": 0.2,
                            "edge_ratio": 0.5,
                            "cloud_ratio": 0.3,
                        },
                        {
                            "device_id": 1,
                            "local_ratio": 0.3,
                            "edge_ratio": 0.4,
                            "cloud_ratio": 0.3,
                            "target_edge": 0,
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    batch = FixedCacheExpertProvider(
        path, num_agents=2, num_edges=2
    ).get_actions(0, 0)

    assert batch.valid_mask.tolist() == [0.0, 1.0]
