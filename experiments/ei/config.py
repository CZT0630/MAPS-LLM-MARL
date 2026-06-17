"""Configuration helpers for EI formal experiments."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

PACKAGE_PARENT = Path(__file__).resolve().parents[3]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from LLM4RL.llm_assistant.ei_state import ei_environment_fingerprint  # noqa: E402
from LLM4RL.utils.config import deep_update, load_config  # noqa: E402
from LLM4RL.utils.run_manifest import sanitize_config, stable_config_hash  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_fragment(path_value: str | Path) -> dict[str, Any]:
    path = resolve_project_path(path_value)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def compose_ei_config(
    base_config: str | Path = "configs/ei/base.yaml",
    fragments: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Load the EI base config and deep-merge method/scale/deadline fragments."""
    config = load_config(str(resolve_project_path(base_config)))
    sources = [str(resolve_project_path(base_config))]
    for fragment in fragments:
        if fragment is None:
            continue
        fragment_path = resolve_project_path(fragment)
        config = deep_update(config, _load_fragment(fragment_path))
        sources.append(str(fragment_path))
    config.setdefault("ei", {})["config_sources"] = sources
    config["ei"]["environment_fingerprint"] = ei_environment_fingerprint(config)
    config["ei"]["resolved_config_hash"] = stable_config_hash(config)
    return config


def with_bank_paths(
    config: dict[str, Any],
    *,
    train_bank: str | Path | None = None,
    test_bank: str | Path | None = None,
) -> dict[str, Any]:
    """Return a config copy with explicit train/test bank paths wired in."""
    output = copy.deepcopy(config)
    if train_bank is not None:
        output.setdefault("training", {})["scenario_bank"] = str(
            resolve_project_path(train_bank)
        )
    if test_bank is not None:
        output.setdefault("evaluation", {})["scenario_bank"] = str(
            resolve_project_path(test_bank)
        )
        output.setdefault("llm_only", {})["scenario_bank"] = str(
            resolve_project_path(test_bank)
        )
    return output


def write_resolved_config(path: str | Path, config: dict[str, Any]) -> Path:
    output = resolve_project_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(
            sanitize_config(config),
            allow_unicode=True,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output

