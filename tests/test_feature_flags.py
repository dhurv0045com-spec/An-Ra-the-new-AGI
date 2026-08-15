import asyncio

from agents.orchestrator import OrchestratorAgent
from engine import feature_flags


class _Agent:
    def __init__(self):
        self.called = False

    async def run(self, task):
        self.called = True
        return {"success": True}


def test_defaults_enable_canonical_and_disable_unproven_systems(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_flags, "FLAGS_FILE", tmp_path / "missing.json")

    flags = feature_flags.load_flags()
    assert flags["brain"] is True
    assert flags["memory"] is True
    assert flags["v4_training"] is True
    for name in ("ouroboros", "robotics", "multimodal"):
        assert flags[name] is False
    assert "v3_training" not in flags


def test_unknown_flag_cannot_be_persisted(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(feature_flags, "FLAGS_FILE", tmp_path / "flags.json")
    with pytest.raises(KeyError, match="unknown feature flag"):
        feature_flags.set_flag("imaginary_capability", True)


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

    feature_flags.set_flag("ouroboros", False)

    assert "ouroboros" in feature_flags.disabled_components()
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
