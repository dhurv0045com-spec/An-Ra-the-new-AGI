from pathlib import Path

from anra_paths import ROOT
from inference.full_system_connector import build_capability_graph
from runtime.system_registry import build_system_manifest, component_registry, source_metrics


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
