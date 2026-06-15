"""Strict parser for legacy LLM expert actions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParseMetadata:
    valid: bool
    source_count: int
    fallback_count: int
    valid_mask: tuple[bool, ...]
    error: str | None = None


class ResponseParser:
    @staticmethod
    def _decode(response: Any) -> list[dict[str, Any]]:
        if isinstance(response, list):
            return response
        if isinstance(response, dict):
            return response.get("strategies") or response.get("actions") or []
        if not isinstance(response, str) or not response.strip():
            return []

        text = response.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.I)
        candidates = [fenced.group(1)] if fenced else []
        candidates.append(text)
        object_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if object_match:
            candidates.append(object_match.group(1))

        for candidate in candidates:
            try:
                decoded = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, list):
                return decoded
            if isinstance(decoded, dict):
                return decoded.get("strategies") or decoded.get("actions") or []
        return []

    @staticmethod
    def parse_with_metadata(
        response: Any,
        num_devices: int,
        num_edges: int,
        num_clouds: int = 1,
    ) -> tuple[list[dict[str, Any]], ParseMetadata]:
        del num_clouds
        try:
            decoded = ResponseParser._decode(response)
            by_device = {}
            for item in decoded:
                if not isinstance(item, dict):
                    continue
                device_id = item.get("device_id", item.get("ue_id"))
                if device_id is None:
                    continue
                try:
                    by_device[int(device_id)] = item
                except (TypeError, ValueError, OverflowError):
                    continue

            strategies = []
            fallback_count = 0
            valid_mask = []
            missing = object()
            for device_id in range(num_devices):
                item = by_device.get(device_id)
                if item is None:
                    fallback_count += 1
                    valid_mask.append(False)
                    strategies.append(ResponseParser._fallback(device_id))
                    continue

                partition = item.get("partition", {})
                if not isinstance(partition, dict):
                    partition = {}
                local = item.get(
                    "local_ratio", partition.get("local", missing)
                )
                edge = item.get(
                    "edge_ratio", partition.get("edge", missing)
                )
                cloud = item.get(
                    "cloud_ratio", partition.get("cloud", missing)
                )
                target = item.get(
                    "target_edge",
                    item.get(
                        "target_edge_server",
                        item.get("edge_id", missing),
                    ),
                )
                if missing in (local, edge, cloud, target):
                    fallback_count += 1
                    valid_mask.append(False)
                    strategies.append(ResponseParser._fallback(device_id))
                    continue
                try:
                    ratios = [
                        max(0.0, float(local)),
                        max(0.0, float(edge)),
                        max(0.0, float(cloud)),
                    ]
                    total = sum(ratios)
                    target_edge = int(target)
                except (TypeError, ValueError, OverflowError):
                    fallback_count += 1
                    valid_mask.append(False)
                    strategies.append(ResponseParser._fallback(device_id))
                    continue
                if total <= 1e-12:
                    fallback_count += 1
                    valid_mask.append(False)
                    strategies.append(ResponseParser._fallback(device_id))
                    continue
                ratios = [value / total for value in ratios]
                valid_mask.append(True)
                strategies.append(
                    {
                        "device_id": device_id,
                        "local_ratio": ratios[0],
                        "edge_ratio": ratios[1],
                        "cloud_ratio": ratios[2],
                        "target_edge": max(
                            0, min(num_edges - 1, target_edge)
                        ),
                    }
                )

            metadata = ParseMetadata(
                valid=bool(decoded) and fallback_count == 0,
                source_count=len(decoded),
                fallback_count=fallback_count,
                valid_mask=tuple(valid_mask),
            )
            return strategies, metadata
        except (TypeError, ValueError, OverflowError) as exc:
            strategies = [
                ResponseParser._fallback(device_id) for device_id in range(num_devices)
            ]
            return strategies, ParseMetadata(
                valid=False,
                source_count=0,
                fallback_count=num_devices,
                valid_mask=tuple(False for _ in range(num_devices)),
                error=str(exc),
            )

    @staticmethod
    def parse_unload_strategy(
        response: Any,
        num_devices: int,
        num_edges: int,
        num_clouds: int = 1,
    ) -> list[dict[str, Any]]:
        strategies, _ = ResponseParser.parse_with_metadata(
            response, num_devices, num_edges, num_clouds
        )
        return strategies

    @staticmethod
    def _fallback(device_id: int) -> dict[str, Any]:
        return {
            "device_id": device_id,
            "local_ratio": 1.0,
            "edge_ratio": 0.0,
            "cloud_ratio": 0.0,
            "target_edge": 0,
        }
