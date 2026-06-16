from pathlib import Path

from LLM4RL.llm_assistant.ei_state import ei_environment_fingerprint
from LLM4RL.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_formal_s1_config_is_phase2_state_keyed():
    config = load_config(str(PROJECT_ROOT / "configs" / "ei" / "formal_s1.yaml"))

    assert config["environment"]["physics_version"] == 2
    assert config["environment"]["num_devices"] == 10
    assert config["environment"]["num_edges"] == 5
    assert config["tasks"]["simple_mode"] is False
    assert config["expert_cache"] == {
        "format": "state_keyed",
        "path": "artifacts/ei/expert_cache.json",
        "fallback_policy": "all_local",
    }
    assert (
        config["llm_only"]["scenario_bank"]
        == "artifacts/ei/expert_scenario_bank.json"
    )
    assert len(ei_environment_fingerprint(config)) == 64
