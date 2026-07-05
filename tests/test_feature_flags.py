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
    # An unregistered name must never read as an enabled capability: the old
    # always-True default let typos and unwired features pass every gate.
    assert feature_flags.is_enabled("unknown_component") is False


def test_every_registry_component_has_an_explicit_flag_default():
    from runtime.system_registry import component_registry

    registered = {component.name for component in component_registry()}
    missing = registered - set(feature_flags._DEFAULTS)
    assert not missing, f"registry components lack flag defaults: {sorted(missing)}"


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
