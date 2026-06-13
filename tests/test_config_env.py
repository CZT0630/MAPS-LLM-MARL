import os

from LLM4RL.utils import config as config_module
from LLM4RL.utils.config import load_project_env


def test_load_project_env_reads_value(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("MIMO_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("MIMO_API_KEY", raising=False)

    loaded = load_project_env(env_path)

    assert loaded is True
    assert os.environ["MIMO_API_KEY"] == "from-dotenv"


def test_process_environment_overrides_dotenv(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("MIMO_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("MIMO_API_KEY", "from-process")

    load_project_env(env_path)

    assert os.environ["MIMO_API_KEY"] == "from-process"


def test_load_config_automatically_reads_project_env(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "MIMO_API_KEY=loaded-by-config\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("MIMO_API_KEY", raising=False)

    config_module.load_config(config_path)

    assert os.environ["MIMO_API_KEY"] == "loaded-by-config"
