
import json

from anra.anra_paths import ROOT
from inference.full_system_connector import build_capability_graph, walk_repository
from runtime import system_registry
from runtime.system_registry import (
    build_system_manifest,
    component_registry,
    get_enabled_components,
    set_component_enabled,
    source_metrics,
)
from runtime.training_readiness import assess_training_readiness


def test_component_registry_has_core_layers():
    names = {component.name for component in component_registry()}
    assert {"brain", "tokenizer", "training_loop", "evaluation", "memory", "goals", "runtime"} <= names


def test_system_manifest_uses_current_repo_root():
    manifest = build_system_manifest(ROOT)
    assert manifest["repo_root"] == str(ROOT)
    assert manifest["metrics"]["python_files"] > 0
    assert all(component["source_ok"] for component in manifest["components"] if component["required"])


def test_source_metrics_exclude_runtime_state():
    metrics = source_metrics(ROOT)
    assert metrics["python_files"] > 0
    assert metrics["source_files"] >= metrics["python_files"]
    assert metrics["python_lines"] > 0


def test_capability_graph_not_stale_workspace_path():
    graph = build_capability_graph(ROOT)
    assert graph["repo_root"] == str(ROOT)
    stale_root = "/" + "workspace" + "/"
    assert stale_root not in str(graph)
    assert graph["capabilities"]["brain"]


def test_manifest_exposes_training_readiness():
    manifest = build_system_manifest(ROOT)
    readiness = manifest["training_readiness"]
    assert readiness["out_of"] == 10
    assert isinstance(readiness["ready_for_session"], bool)
    assert readiness["checks"]


def test_readiness_distinguishes_session_from_milestone():
    readiness = assess_training_readiness()
    assert readiness.out_of == 10
    assert isinstance(readiness.ready_for_session, bool)
    assert isinstance(readiness.ready_for_milestone, bool)


def test_all_components_have_metric_hooks(tmp_path, monkeypatch):
    from engine import feature_flags

    system_registry._COMPONENT_ENABLED_OVERRIDES.clear()
    monkeypatch.setattr(feature_flags, "FLAGS_FILE", tmp_path / "missing_flags.json")
    try:
        for component in component_registry():
            assert component.enabled is True
            assert {"latency_ms", "success", "error_type"} <= set(component.metric_hooks)
    finally:
        system_registry._COMPONENT_ENABLED_OVERRIDES.clear()


def test_get_enabled_components_filters_disabled(monkeypatch):
    system_registry._COMPONENT_ENABLED_OVERRIDES.clear()
    monkeypatch.setattr(system_registry, "component_status", lambda component: {"source_ok": True})
    set_component_enabled("memory", False)
    try:
        names = {component.name for component in get_enabled_components()}
        assert "memory" not in names
        assert "brain" in names
    finally:
        system_registry._COMPONENT_ENABLED_OVERRIDES.clear()


def test_set_component_enabled_persists(tmp_path, monkeypatch):
    from engine import feature_flags

    system_registry._COMPONENT_ENABLED_OVERRIDES.clear()
    flags_file = tmp_path / "feature_flags.json"
    monkeypatch.setattr(feature_flags, "FLAGS_FILE", flags_file)
    set_component_enabled("ghost_memory", False, persist=True)
    try:
        data = json.loads(flags_file.read_text(encoding="utf-8"))
        assert data["ghost_memory"] is False
    finally:
        system_registry._COMPONENT_ENABLED_OVERRIDES.clear()


def test_probe_module_capability_requires_real_health():
    from inference.full_system_connector import probe_module_capability

    # A module that does not exist can never be a capability.
    assert probe_module_capability("phase3.definitely_not_a_module") is False
    # A real module whose health_check reports ok is a capability; turboquant
    # defines health_check too, so that evidence outranks symbol presence.
    assert probe_module_capability("phase3.ouroboros_45o.ouroboros_numpy") is True
    assert (
        probe_module_capability(
            "archive.core_45eh_numpy_archived.turboquant", "CompressedKVCache"
        )
        is True
    )
    # A module without health_check must expose the required symbol.
    assert probe_module_capability("runtime.safe_load", "safe_torch_load") is True
    assert probe_module_capability("runtime.safe_load", "no_such_symbol") is False


def test_capability_graph_flags_come_from_probes(monkeypatch):
    from inference import full_system_connector

    # Force one probe to fail; the graph must report that capability False
    # instead of falling back to "a file with this substring exists".
    real_probe = full_system_connector.probe_module_capability

    def fake_probe(module_name: str, required_symbol=None):
        if "symbolic_bridge" in module_name:
            return False
        return real_probe(module_name, required_symbol)

    monkeypatch.setattr(full_system_connector, "probe_module_capability", fake_probe)
    graph = full_system_connector.build_capability_graph(ROOT)
    capabilities = graph["capabilities"]
    assert capabilities["symbolic_bridge"] is False
    assert capabilities["ouroboros"] is True


def test_repository_walk_prunes_runtime_data_and_virtualenvs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("def live():\n    return True\n")
    (tmp_path / "training_data").mkdir()
    (tmp_path / "training_data" / "huge.jsonl").write_text("ignored")
    (tmp_path / ".venv" / "Lib").mkdir(parents=True)
    (tmp_path / ".venv" / "Lib" / "dependency.py").write_text("def ignored(): ...\n")
    (tmp_path / ".venv-cuda" / "Lib").mkdir(parents=True)
    (tmp_path / ".venv-cuda" / "Lib" / "cuda_dependency.py").write_text(
        "def ignored_cuda(): ...\n"
    )

    nodes = walk_repository(tmp_path)

    assert [node.path for node in nodes] == ["src/module.py"]
    assert nodes[0].functions == ["live"]
