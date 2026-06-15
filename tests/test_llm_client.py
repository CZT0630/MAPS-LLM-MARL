import json

import pytest

from LLM4RL.llm_assistant.llm_client import LLMClient
from LLM4RL.utils.run_manifest import build_manifest, sanitize_config


class FakeResponse:
    headers = {"x-request-id": "request-123"}

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "id": "completion-123",
            "model": "mimo-v2.5",
            "choices": [{"message": {"content": '{"actions": []}'}}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
            },
        }


def mimo_config():
    return {
        "llm": {
            "provider": "xiaomi_mimo",
            "display_name": "MiMo-V2.5",
            "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
            "endpoint": "/chat/completions",
            "model_name": "mimo-v2.5",
            "api_key_env": "MIMO_API_KEY",
            "temperature": 0.0,
            "max_completion_tokens": 1024,
            "enable_thinking": False,
            "max_retries": 0,
        }
    }


def test_mimo_client_uses_environment_credential_and_chat_payload(
    monkeypatch,
):
    monkeypatch.setenv("MIMO_API_KEY", "test-only-secret")
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return FakeResponse()

    monkeypatch.setattr(
        "LLM4RL.llm_assistant.llm_client.requests.post",
        fake_post,
    )
    client = LLMClient(mimo_config())

    content = client.query("Return JSON only.")

    assert content == '{"actions": []}'
    assert (
        captured["url"]
        == "https://token-plan-cn.xiaomimimo.com/v1/chat/completions"
    )
    assert captured["headers"]["Authorization"] == "Bearer test-only-secret"
    assert captured["json"]["model"] == "mimo-v2.5"
    assert captured["json"]["max_completion_tokens"] == 1024
    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert client.last_metadata["returned_model"] == "mimo-v2.5"
    assert client.last_metadata["temperature"] == 0.0
    assert client.last_metadata["usage"]["total_tokens"] == 16


def test_mimo_client_requires_environment_credential(monkeypatch):
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    client = LLMClient(mimo_config())

    with pytest.raises(RuntimeError, match="MIMO_API_KEY"):
        client.query("Return JSON only.")


def test_inline_api_key_is_rejected():
    config = mimo_config()
    config["llm"]["api_key"] = "must-not-be-persisted"

    with pytest.raises(ValueError, match="Inline llm.api_key"):
        LLMClient(config)


def test_manifest_redacts_inline_credentials(tmp_path):
    config = mimo_config()
    config["llm"]["api_key"] = "must-not-appear"
    config["llm"]["api_key_env"] = "MIMO_API_KEY"

    sanitized = sanitize_config(config)
    manifest = build_manifest(
        project_root=tmp_path,
        algorithm="test",
        seed=42,
        config=config,
        command="test",
    )
    serialized = json.dumps(manifest)

    assert sanitized["llm"]["api_key"] == "<redacted>"
    assert sanitized["llm"]["api_key_env"] == "MIMO_API_KEY"
    assert "must-not-appear" not in serialized
