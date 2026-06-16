"""Auditable joint-state expert cache for formal EI experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .response_parser import ResponseParser


def _canonicalize(value: Any, precision: int) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(item, precision)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item, precision) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("state contains non-finite values")
        rounded = float(format(value, f".{precision}g"))
        return 0.0 if rounded == 0.0 else rounded
    raise TypeError(f"unsupported state value type: {type(value).__name__}")


def hash_state(state: Any, precision: int = 12) -> str:
    """Return a stable full SHA-256 hash for a canonical joint state."""
    canonical = _canonicalize(state, precision)
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    state_hash: str
    state_key_version: str
    prompt_version: str
    provider: str
    requested_model: str
    returned_model: str
    temperature: float
    raw_response_hash: str
    raw_response: str
    num_edges: int
    parsed_action: list[dict[str, Any]]
    decision_mask: list[bool]
    valid_mask: list[bool]
    parser_status: str
    fallback_status: str
    request_id: str
    token_usage: dict[str, int]
    query_latency_ms: float
    generation_time: str
    paper_evidence: bool = True

    @property
    def parsed_actions(self) -> list[dict[str, Any]]:
        """Compatibility alias for the pre-audit field name."""
        return self.parsed_action


class ExpertCache:
    """Frozen cache keyed by the full joint EI decision state."""

    SCHEMA_VERSION = "ei-cache-v1"
    STATE_KEY_VERSION = "ei-state-v1"

    def __init__(
        self,
        prompt_version: str,
        provider: str,
        requested_model: str,
        temperature: float = 0.0,
        state_key_version: str = STATE_KEY_VERSION,
    ):
        self.prompt_version = str(prompt_version)
        self.provider = str(provider)
        self.requested_model = str(requested_model)
        self.temperature = float(temperature)
        self.state_key_version = str(state_key_version)
        if not self.prompt_version:
            raise ValueError("prompt_version must not be empty")
        if not self.provider:
            raise ValueError("provider must not be empty")
        if not self.requested_model:
            raise ValueError("requested_model must not be empty")
        if not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite")
        self._entries: dict[str, CacheEntry] = {}
        self.source_path: str | None = None
        self.file_sha256: str | None = None

    def get(self, state_hash: str) -> CacheEntry | None:
        return self._entries.get(state_hash)

    def has(self, state_hash: str) -> bool:
        return state_hash in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, state_hash: str) -> bool:
        return state_hash in self._entries

    @property
    def state_hashes(self) -> list[str]:
        return list(self._entries)

    def add(self, entry: CacheEntry) -> None:
        if entry.state_hash in self._entries:
            raise ValueError(f"duplicate state_hash {entry.state_hash!r}")
        if (
            len(entry.state_hash) != 64
            or any(char not in "0123456789abcdef" for char in entry.state_hash)
        ):
            raise ValueError("state_hash must be a lowercase SHA-256 digest")
        if entry.state_key_version != self.state_key_version:
            raise ValueError("entry state_key_version does not match cache")
        if entry.prompt_version != self.prompt_version:
            raise ValueError("entry prompt_version does not match cache")
        if entry.provider != self.provider:
            raise ValueError("entry provider does not match cache")
        if entry.requested_model != self.requested_model:
            raise ValueError("entry requested_model does not match cache")
        if entry.returned_model != self.requested_model:
            raise ValueError("returned_model does not match requested_model")
        if not math.isclose(
            entry.temperature,
            self.temperature,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("entry temperature does not match cache")
        if not entry.request_id:
            raise ValueError("request_id must not be empty")
        if not math.isfinite(entry.query_latency_ms) or entry.query_latency_ms < 0:
            raise ValueError("query_latency_ms must be finite and non-negative")
        expected_raw_hash = hashlib.sha256(
            entry.raw_response.encode("utf-8")
        ).hexdigest()
        if entry.raw_response_hash != expected_raw_hash:
            raise ValueError("raw_response_hash does not match raw_response")
        if entry.num_edges <= 0:
            raise ValueError("num_edges must be positive")
        action_count = len(entry.parsed_action)
        if (
            action_count != len(entry.decision_mask)
            or action_count != len(entry.valid_mask)
        ):
            raise ValueError(
                "parsed_action, decision_mask, and valid_mask lengths differ"
            )
        if any(valid and not required for valid, required in zip(
            entry.valid_mask, entry.decision_mask
        )):
            raise ValueError("valid_mask cannot enable a non-decision UE")
        reparsed, parse_meta = ResponseParser.parse_with_metadata(
            entry.raw_response,
            num_devices=action_count,
            num_edges=entry.num_edges,
            strict_constraints=True,
        )
        expected_actions = [
            {
                "ue_id": strategy["device_id"],
                "partition": {
                    "local": strategy["local_ratio"],
                    "edge": strategy["edge_ratio"],
                    "cloud": strategy["cloud_ratio"],
                },
                "edge_id": strategy["target_edge"],
            }
            for strategy in reparsed
        ]
        if entry.parsed_action != expected_actions:
            raise ValueError("parsed_action does not match raw_response")
        expected_valid_mask = [
            bool(required and valid)
            for required, valid in zip(
                entry.decision_mask, parse_meta.valid_mask
            )
        ]
        if entry.valid_mask != expected_valid_mask:
            raise ValueError("valid_mask does not match raw_response")
        required_count = sum(entry.decision_mask)
        valid_count = sum(entry.valid_mask)
        expected_status = (
            "valid"
            if required_count > 0 and valid_count == required_count
            else "failed"
            if valid_count == 0
            else "partial"
        )
        expected_fallback = {
            "valid": "none",
            "partial": "partial_fallback",
            "failed": "full_fallback",
        }[expected_status]
        if entry.parser_status != expected_status:
            raise ValueError("parser_status is inconsistent with action masks")
        if entry.fallback_status != expected_fallback:
            raise ValueError("fallback_status is inconsistent with parser_status")
        if not entry.generation_time:
            raise ValueError("generation_time must not be empty")
        if not entry.token_usage:
            raise ValueError("token_usage must not be empty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in entry.token_usage.values()
        ):
            raise ValueError("token_usage values must be non-negative integers")
        if not entry.paper_evidence:
            raise ValueError("formal cache entries must be paper evidence")
        self._entries[entry.state_hash] = entry

    def add_from_api_response(
        self,
        *,
        state_hash: str,
        raw_response: str,
        num_devices: int,
        num_edges: int,
        llm_metadata: dict[str, Any],
        query_latency_ms: float,
        decision_mask: list[bool] | None = None,
    ) -> CacheEntry:
        if num_devices <= 0 or num_edges <= 0:
            raise ValueError("num_devices and num_edges must be positive")
        if decision_mask is None:
            decision_mask = [True] * num_devices
        if len(decision_mask) != num_devices:
            raise ValueError("decision_mask length does not match num_devices")
        decision_mask = [bool(value) for value in decision_mask]
        returned_model = str(llm_metadata.get("returned_model") or "")
        request_id = str(llm_metadata.get("request_id") or "")
        metadata_provider = str(llm_metadata.get("provider") or self.provider)
        metadata_requested = str(
            llm_metadata.get("requested_model") or self.requested_model
        )
        if metadata_provider != self.provider:
            raise ValueError("API provider does not match cache provider")
        if metadata_requested != self.requested_model:
            raise ValueError("API requested_model does not match cache")
        if returned_model != self.requested_model:
            raise ValueError(
                "API returned_model must match requested_model for paper evidence"
            )
        if not request_id:
            raise ValueError("API request_id is required for paper evidence")
        metadata_temperature = float(
            llm_metadata.get("temperature", self.temperature)
        )
        if not math.isclose(
            metadata_temperature,
            self.temperature,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("API temperature does not match cache")
        if not math.isfinite(float(query_latency_ms)) or query_latency_ms < 0:
            raise ValueError("query_latency_ms must be finite and non-negative")

        strategies, parse_meta = ResponseParser.parse_with_metadata(
            raw_response,
            num_devices=num_devices,
            num_edges=num_edges,
            strict_constraints=True,
        )
        valid_mask = [
            bool(required and valid)
            for required, valid in zip(decision_mask, parse_meta.valid_mask)
        ]
        parsed_action = [
            {
                "ue_id": strategy["device_id"],
                "partition": {
                    "local": strategy["local_ratio"],
                    "edge": strategy["edge_ratio"],
                    "cloud": strategy["cloud_ratio"],
                },
                "edge_id": strategy["target_edge"],
            }
            for strategy in strategies
        ]
        required_count = sum(decision_mask)
        valid_count = sum(valid_mask)
        if required_count > 0 and valid_count == required_count:
            parser_status = "valid"
            fallback_status = "none"
        elif valid_count == 0:
            parser_status = "failed"
            fallback_status = "full_fallback"
        else:
            parser_status = "partial"
            fallback_status = "partial_fallback"

        entry = CacheEntry(
            state_hash=state_hash,
            state_key_version=self.state_key_version,
            prompt_version=self.prompt_version,
            provider=self.provider,
            requested_model=self.requested_model,
            returned_model=returned_model,
            temperature=metadata_temperature,
            raw_response_hash=hashlib.sha256(
                raw_response.encode("utf-8")
            ).hexdigest(),
            raw_response=raw_response,
            num_edges=num_edges,
            parsed_action=parsed_action,
            decision_mask=decision_mask,
            valid_mask=valid_mask,
            parser_status=parser_status,
            fallback_status=fallback_status,
            request_id=request_id,
            token_usage={
                str(key): int(value)
                for key, value in dict(llm_metadata.get("usage", {})).items()
                if isinstance(value, (int, float))
            },
            query_latency_ms=float(query_latency_ms),
            generation_time=time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            paper_evidence=True,
        )
        self.add(entry)
        return entry

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "state_key_version": self.state_key_version,
            "prompt_version": self.prompt_version,
            "provider": self.provider,
            "requested_model": self.requested_model,
            "temperature": self.temperature,
            "paper_evidence": True,
            "entries": {
                state_hash: asdict(entry)
                for state_hash, entry in sorted(self._entries.items())
            },
        }
        cache_bytes = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        temporary_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary_path.write_bytes(cache_bytes)
        for attempt in range(10):
            try:
                temporary_path.replace(path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.1 * (attempt + 1))
        self.source_path = str(path.resolve())
        self.file_sha256 = hashlib.sha256(cache_bytes).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> "ExpertCache":
        path = Path(path)
        cache_bytes = path.read_bytes()
        data = json.loads(cache_bytes.decode("utf-8"))
        if data.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("unsupported expert cache schema_version")
        if data.get("paper_evidence") is not True:
            raise ValueError("formal expert cache must set paper_evidence=true")
        cache = cls(
            prompt_version=data["prompt_version"],
            provider=data["provider"],
            requested_model=data["requested_model"],
            temperature=data.get("temperature", 0.0),
            state_key_version=data.get(
                "state_key_version", cls.STATE_KEY_VERSION
            ),
        )
        for state_hash, entry_data in data.get("entries", {}).items():
            normalized = dict(entry_data)
            if "parsed_action" not in normalized:
                normalized["parsed_action"] = normalized.pop("parsed_actions")
            normalized.setdefault(
                "num_edges",
                max(
                    (
                        int(action.get("edge_id", 0))
                        for action in normalized["parsed_action"]
                    ),
                    default=0,
                )
                + 1,
            )
            normalized.setdefault(
                "decision_mask",
                [True] * len(normalized["parsed_action"]),
            )
            normalized.setdefault(
                "valid_mask",
                [normalized.get("parser_status") == "valid"]
                * len(normalized["parsed_action"]),
            )
            normalized.setdefault(
                "state_key_version", cache.state_key_version
            )
            entry = CacheEntry(**normalized)
            if state_hash != entry.state_hash:
                raise ValueError("cache entry key does not match state_hash")
            cache.add(entry)
        cache.source_path = str(path.resolve())
        cache.file_sha256 = hashlib.sha256(cache_bytes).hexdigest()
        return cache

    def stats(self) -> dict[str, Any]:
        total = len(self._entries)
        if total == 0:
            return {
                "total": 0,
                "parser_success_rate": 0.0,
                "valid_action_rate": 0.0,
                "fallback_rate": 0.0,
            }
        valid = sum(
            entry.parser_status == "valid"
            for entry in self._entries.values()
        )
        partial = sum(
            entry.parser_status == "partial"
            for entry in self._entries.values()
        )
        failed = total - valid - partial
        action_count = sum(
            sum(entry.decision_mask) for entry in self._entries.values()
        )
        valid_action_count = sum(
            sum(entry.valid_mask) for entry in self._entries.values()
        )
        latencies = [
            entry.query_latency_ms for entry in self._entries.values()
        ]
        return {
            "total": total,
            "valid": valid,
            "partial": partial,
            "failed": failed,
            "valid_rate": valid / total,
            "parser_success_rate": (valid + partial) / total,
            "valid_action_count": valid_action_count,
            "action_count": action_count,
            "valid_action_rate": (
                valid_action_count / action_count if action_count else 0.0
            ),
            "fallback_rate": (
                1.0 - valid_action_count / action_count
                if action_count
                else 0.0
            ),
            "mean_latency_ms": sum(latencies) / total,
            "max_latency_ms": max(latencies),
        }
