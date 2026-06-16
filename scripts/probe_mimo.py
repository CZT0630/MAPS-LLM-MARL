"""Issue one minimal MiMo request to validate local API configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_package_root = Path(__file__).resolve().parents[1]
_workspace_root = _package_root.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from LLM4RL.llm_assistant.llm_client import LLMClient
from LLM4RL.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/ei/llm_mimo.yaml",
        help="YAML configuration containing the llm section.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.is_file():
        config_path = _package_root / config_path
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    client = LLMClient(load_config(config_path))
    response = client.query(
        'Return JSON only with this exact object: {"status":"ok"}'
    )
    print(response)
    print(json.dumps(client.last_metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
