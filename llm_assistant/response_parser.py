"""Strict parser for legacy LLM expert actions."""

from __future__ import annotations

import json
import math
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
        strict_constraints: bool = False,
    ) -> tuple[list[dict[str, Any]], ParseMetadata]:
        del num_clouds
        try:
            decoded = ResponseParser._decode(response)
            by_device = {}
            duplicate_devices = set()
            for item in decoded:
                if not isinstance(item, dict):
                    continue
                device_id = item.get("device_id", item.get("ue_id"))
                if device_id is None:
                    continue
                try:
                    parsed_device_id = int(device_id)
                except (TypeError, ValueError, OverflowError):
                    continue
                if parsed_device_id in by_device:
                    duplicate_devices.add(parsed_device_id)
                by_device[parsed_device_id] = item

            strategies = []
            fallback_count = 0
            valid_mask = []
            missing = object()
            for device_id in range(num_devices):
                item = by_device.get(device_id)
                if item is None or device_id in duplicate_devices:
                    fallback_count += 1
                    valid_mask.append(False)
                    strategies.append(ResponseParser._fallback(device_id))
                    continue
                if strict_constraints:
                    raw_device_id = item.get(
                        "device_id", item.get("ue_id")
                    )
                    if (
                        not isinstance(raw_device_id, int)
                        or isinstance(raw_device_id, bool)
                    ):
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
                    ratios = [float(local), float(edge), float(cloud)]
                    total = sum(ratios)
                    target_edge = int(target)
                except (TypeError, ValueError, OverflowError):
                    fallback_count += 1
                    valid_mask.append(False)
                    strategies.append(ResponseParser._fallback(device_id))
                    continue
                if not all(math.isfinite(value) for value in ratios):
                    fallback_count += 1
                    valid_mask.append(False)
                    strategies.append(ResponseParser._fallback(device_id))
                    continue
                if strict_constraints:
                    raw_ratios = (local, edge, cloud)
                    ratio_types_valid = all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        for value in raw_ratios
                    )
                    target_is_integer = (
                        isinstance(target, int)
                        and not isinstance(target, bool)
                    )
                    ratios_valid = (
                        ratio_types_valid
                        and all(0.0 <= value <= 1.0 for value in ratios)
                        and math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-5)
                    )
                    target_valid = (
                        target_is_integer
                        and 0 <= target_edge < num_edges
                    )
                    if not ratios_valid or not target_valid:
                        fallback_count += 1
                        valid_mask.append(False)
                        strategies.append(ResponseParser._fallback(device_id))
                        continue
                else:
                    ratios = [max(0.0, value) for value in ratios]
                    total = sum(ratios)
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
        strict_constraints: bool = False,
    ) -> list[dict[str, Any]]:
        strategies, _ = ResponseParser.parse_with_metadata(
            response,
            num_devices,
            num_edges,
            num_clouds,
            strict_constraints=strict_constraints,
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
