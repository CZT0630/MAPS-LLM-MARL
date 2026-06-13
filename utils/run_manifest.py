"""Run manifest utilities for reproducible experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


SENSITIVE_CONFIG_KEYS = {
    "api_key",
    "access_token",
    "authorization",
    "password",
    "secret",
}


def sanitize_config(config: Any) -> Any:
    """Remove inline credentials before hashing or persisting a run config."""
    if isinstance(config, dict):
        sanitized = {}
        for key, value in config.items():
            if str(key).lower() in SENSITIVE_CONFIG_KEYS:
                sanitized[key] = "<redacted>" if value else ""
            else:
                sanitized[key] = sanitize_config(value)
        return sanitized
    if isinstance(config, list):
        return [sanitize_config(value) for value in config]
    if isinstance(config, tuple):
        return tuple(sanitize_config(value) for value in config)
    return config


def stable_config_hash(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(
        sanitize_config(config), allow_unicode=True, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(
    project_root: Path,
    algorithm: str,
    seed: int,
    config: dict[str, Any],
    command: str,
    status: str = "running",
) -> dict[str, Any]:
    safe_config = sanitize_config(config)
    return {
        "schema_version": "1.0",
        "model_version": config.get("model_version", 2),
        "algorithm": algorithm,
        "seed": int(seed),
        "status": status,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config_hash": stable_config_hash(safe_config),
        "git_commit": get_git_commit(project_root),
        "system": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "config": safe_config,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
