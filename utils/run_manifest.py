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


def stable_config_hash(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(config, allow_unicode=True, sort_keys=True)
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
    return {
        "schema_version": "1.0",
        "model_version": config.get("model_version", 2),
        "algorithm": algorithm,
        "seed": int(seed),
        "status": status,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config_hash": stable_config_hash(config),
        "git_commit": get_git_commit(project_root),
        "system": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "config": config,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
