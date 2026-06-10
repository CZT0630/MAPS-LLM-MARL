"""Minimal OpenAI-compatible HTTP client for expert generation."""

from __future__ import annotations

from typing import Any

import requests


class LLMClient:
    def __init__(self, config: dict[str, Any]):
        llm_config = config.get("llm", {})
        self.base_url = llm_config.get("base_url", "")
        self.model_name = llm_config.get("model_name", "")
        self.api_key = llm_config.get("api_key", "")
        self.temperature = float(llm_config.get("temperature", 0.0))
        self.max_tokens = int(llm_config.get("max_tokens", 4096))
        self.timeout = (
            float(llm_config.get("timeout", 30)),
            float(llm_config.get("read_timeout", 120)),
        )

    def query(self, prompt: str) -> str:
        if not self.base_url:
            raise RuntimeError("LLM base_url is not configured")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        is_chat = "/chat/completions" in self.base_url
        payload: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if is_chat:
            payload["messages"] = [{"role": "user", "content": prompt}]
        else:
            payload["prompt"] = prompt

        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response has no choices")
        first = choices[0]
        if is_chat:
            return str(first.get("message", {}).get("content", ""))
        return str(first.get("text", ""))
