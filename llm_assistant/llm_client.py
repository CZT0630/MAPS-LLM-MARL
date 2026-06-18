"""OpenAI-compatible HTTP client for auditable LLM expert generation."""

from __future__ import annotations

import os
import time
from typing import Any

import requests


class LLMClient:
    def __init__(self, config: dict[str, Any]):
        llm_config = config.get("llm", {})
        self.provider = str(llm_config.get("provider", "openai_compatible"))
        self.display_name = str(
            llm_config.get("display_name", llm_config.get("model_name", ""))
        )
        self.base_url = str(llm_config.get("base_url", "")).rstrip("/")
        self.model_name = str(llm_config.get("model_name", ""))
        self.api_key_env = str(llm_config.get("api_key_env", "LLM_API_KEY"))
        inline_api_key = str(llm_config.get("api_key", "")).strip()
        if inline_api_key:
            raise ValueError(
                "Inline llm.api_key is not allowed because resolved configs and "
                f"run manifests are persisted. Set {self.api_key_env} instead."
            )

        endpoint = str(llm_config.get("endpoint", "")).strip()
        if self.base_url.endswith(("/chat/completions", "/completions")):
            self.request_url = self.base_url
        else:
            endpoint = endpoint or "/chat/completions"
            self.request_url = f"{self.base_url}/{endpoint.lstrip('/')}"

        self.temperature = float(llm_config.get("temperature", 0.0))
        self.max_completion_tokens = int(
            llm_config.get(
                "max_completion_tokens",
                llm_config.get("max_tokens", 4096),
            )
        )
        self.enable_thinking = bool(llm_config.get("enable_thinking", False))
        self.max_retries = int(llm_config.get("max_retries", 2))
        self.retry_backoff_seconds = float(
            llm_config.get("retry_backoff_seconds", 1.0)
        )
        self.retry_backoff_max_seconds = float(
            llm_config.get("retry_backoff_max_seconds", 30.0)
        )
        self.timeout = (
            float(llm_config.get("timeout", 30)),
            float(llm_config.get("read_timeout", 120)),
        )
        self.last_metadata: dict[str, Any] = {}

    def query(self, prompt: str) -> str:
        if not self.base_url:
            raise RuntimeError("LLM base_url is not configured")
        if not self.model_name:
            raise RuntimeError("LLM model_name is not configured")

        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                "LLM credential is not configured. Set environment variable "
                f"{self.api_key_env}."
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        is_chat = "/chat/completions" in self.request_url
        payload: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "stream": False,
        }
        if is_chat:
            payload["messages"] = [{"role": "user", "content": prompt}]
            payload["max_completion_tokens"] = self.max_completion_tokens
            payload["thinking"] = {
                "type": "enabled" if self.enable_thinking else "disabled"
            }
        else:
            payload["prompt"] = prompt
            payload["max_tokens"] = self.max_completion_tokens

        started = time.perf_counter()
        response = None
        attempt_count = 0
        for attempt in range(self.max_retries + 1):
            attempt_count = attempt + 1
            try:
                response = requests.post(
                    self.request_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt >= self.max_retries:
                    raise
                delay = min(
                    self.retry_backoff_seconds * (2**attempt),
                    self.retry_backoff_max_seconds,
                )
                if delay > 0.0:
                    time.sleep(delay)

        if response is None:
            raise RuntimeError("LLM request did not produce a response")

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response has no choices")
        first = choices[0]
        self.last_metadata = {
            "provider": self.provider,
            "display_name": self.display_name,
            "requested_model": self.model_name,
            "returned_model": body.get("model"),
            "temperature": self.temperature,
            "request_id": (
                body.get("id")
                or response.headers.get("x-request-id")
                or response.headers.get("request-id")
            ),
            "query_latency_ms": elapsed_ms,
            "attempt_count": attempt_count,
            "usage": body.get("usage", {}),
        }
        if is_chat:
            return str(first.get("message", {}).get("content", ""))
        return str(first.get("text", ""))
