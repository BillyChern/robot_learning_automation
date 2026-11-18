"""config.load follows RLPJ_TASK even after an earlier call cached another task."""
from rlpj import config


def test_load_follows_env(monkeypatch):
    monkeypatch.delenv("RLPJ_TASK", raising=False)
    first = config.load()["task"]["name"]
    monkeypatch.setenv("RLPJ_TASK", "drawer_rl")
    assert config.load()["task"]["name"] == "drawer_rl"
    monkeypatch.delenv("RLPJ_TASK")
    assert config.load()["task"]["name"] == first
