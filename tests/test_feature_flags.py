import asyncio

from agents.orchestrator import OrchestratorAgent
from engine import feature_flags


class _Agent:
    def __init__(self):
        self.called = False

    async def run(self, task):
        self.called = True
        return {"success": True}


def test_default_all_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_flags, "FLAGS_FILE", tmp_path / "missing.json")

    flags = feature_flags.load_flags()
    assert flags
    assert all(flags.values())


def test_set_flag_persists(tmp_path, monkeypatch):
    path = tmp_path / "feature_flags.json"
    monkeypatch.setattr(feature_flags, "FLAGS_FILE", path)

    feature_flags.set_flag("ouroboros", False)

    assert path.exists()
    assert feature_flags.load_flags()["ouroboros"] is False


def test_is_enabled_respects_override(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_flags, "FLAGS_FILE", tmp_path / "feature_flags.json")

    feature_flags.set_flag("memory", False)

    assert feature_flags.is_enabled("memory") is False
    assert feature_flags.is_enabled("unknown_component") is True


def test_disabled_components_list(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_flags, "FLAGS_FILE", tmp_path / "feature_flags.json")

    feature_flags.set_flag("ghost_memory", False)

    assert "ghost_memory" in feature_flags.disabled_components()
    assert "brain" in feature_flags.enabled_components()


def test_orchestrator_skips_disabled_component(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_flags, "FLAGS_FILE", tmp_path / "feature_flags.json")
    feature_flags.set_flag("memory", False)
    agent = _Agent()
    orchestrator = OrchestratorAgent(agent, agent, agent, agent)

    result = asyncio.run(orchestrator.dispatch({"kind": "memory"}))

    assert result["skipped"] is True
    assert "memory" in result["reason"]
    assert agent.called is False
