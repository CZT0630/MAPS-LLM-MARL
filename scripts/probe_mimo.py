"""Issue one minimal MiMo request to validate local API configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_assistant.llm_client import LLMClient
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/ei/llm_mimo.yaml",
        help="YAML configuration containing the llm section.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
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
