from __future__ import annotations

import importlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from anra_paths import (
    DATASET_CANONICAL,
    OUTPUT_V2_DIR,
    ROOT,
    SCRIPTS_DIR,
    TOKENIZER_DIR,
    V3_TOKENIZER_FILE,
    WORKSPACE_DIR,
    get_v2_checkpoint,
    inject_all_paths,
)


SOURCE_SUFFIXES = {".py", ".md", ".ipynb", ".yaml", ".yml", ".json", ".toml"}
IGNORED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "output",
    "state",
    "workspace",
}
IGNORED_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".faiss", ".index", ".npy", ".npz", ".pt", ".pth"}
IGNORED_FILENAMES = {
    "package-lock.json",
    "tokenizer_v2.json",
    "tokenizer_v3.json",
    "anra_training.txt",
}


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


@dataclass(frozen=True)
class SystemComponent:
    name: str
    layer: str
    role: str
    paths: tuple[str, ...]
    import_name: str | None = None
    required: bool = True


def component_registry() -> list[SystemComponent]:
    """Canonical architecture map for the current An-Ra mainline."""
    return [
        SystemComponent(
            name="brain",
            layer="model",
            role="V2 causal transformer core: GQA, RoPE/YaRN, MoD, Flash SDP path, tied embeddings.",
            paths=("anra_brain.py", "training/v2_config.py", "training/v2_runtime.py"),
            import_name="anra_brain",
        ),
        SystemComponent(
            name="tokenizer",
            layer="data",
            role="Canonical 8192-token BPE tokenizer and dependency-light adapter surface.",
            paths=(
                _rel(V3_TOKENIZER_FILE),
                _rel(TOKENIZER_DIR / "tokenizer_adapter.py"),
                _rel(SCRIPTS_DIR / "train_tokenizer_v3.py"),
            ),
            import_name="tokenizer.tokenizer_adapter",
        ),
        SystemComponent(
            name="data_mix",
            layer="data",
            role="Owner-data-first corpus contract, teacher/symbolic/replay buckets, and dataset setup.",
            paths=(_rel(DATASET_CANONICAL), "training/v2_data_mix.py", "scripts/setup_dataset.py"),
        ),
        SystemComponent(
            name="training_loop",
            layer="learning",
            role="Daily/milestone trainer, dataset resolution, Drive restore, checkpoint and report runtime.",
            paths=("training/train_unified.py", "scripts/build_brain.py", "training/finetune_anra.py"),
            import_name="training.train_unified",
        ),
        SystemComponent(
            name="evaluation",
            layer="measurement",
            role="Compact eval, model-running benchmark suite, verifier, and hard-example feedback.",
            paths=("training/eval_v2.py", "training/benchmark.py", "training/verifier.py"),
            import_name="training.benchmark",
        ),
        SystemComponent(
            name="runtime",
            layer="serving",
            role="Generation, streaming, trace capture, connector refresh, and local inference helpers.",
            paths=("generate.py", "inference/full_system_connector.py", "inference/anra_infer.py"),
            import_name="generate",
        ),
        SystemComponent(
            name="api_web",
            layer="interface",
            role="FastAPI backend plus Phase 4 Vite/React operator interface.",
            paths=("app.py", "phase4/web/src/App.jsx", "phase4/web/src/index.css", "phase4/web/README.md"),
        ),
        SystemComponent(
            name="identity",
            layer="alignment",
            role="CIV residual guard, ESV modulation, watcher checks, and Phase 3 identity injection.",
            paths=("identity/civ.py", "identity/esv.py", "identity/civ_watcher.py", "phase3/identity (45N)/identity_injector.py"),
            import_name="identity.civ",
        ),
        SystemComponent(
            name="memory",
            layer="continuity",
            role="Unified memory router over episodic, short-term, graph, ghost, and ESV-gated writes.",
            paths=("memory/memory_router.py", "memory/faiss_store.py"),
            import_name="memory.memory_router",
        ),
        SystemComponent(
            name="phase2_memory",
            layer="continuity",
            role="45J typed memory, retrieval, vector index, context builder, and personal graph.",
            paths=(
                "phase2/memory (45J)/memory_manager.py",
                "phase2/memory (45J)/store.py",
                "phase2/memory (45J)/vectors.py",
                "phase2/memory (45J)/context_builder.py",
            ),
        ),
        SystemComponent(
            name="goals",
            layer="agency",
            role="Persistent priority queue for goals, retries, successors, and orchestrator dispatch.",
            paths=("goals/goal_queue.py", "agents/orchestrator.py", "agents/specialists.py"),
            import_name="goals.goal_queue",
        ),
        SystemComponent(
            name="agent_loop",
            layer="agency",
            role="45K goal interpretation, planning, dispatch, execution, monitoring, and evaluation.",
            paths=(
                "phase2/agent_loop (45k)/agent_main.py",
                "phase2/agent_loop (45k)/planner.py",
                "phase2/agent_loop (45k)/executor.py",
                "phase2/agent_loop (45k)/evaluator.py",
            ),
        ),
        SystemComponent(
            name="master_system",
            layer="autonomy",
            role="45M owner-control system, persistent service, long-horizon goals, safety, and personalization.",
            paths=(
                "phase2/master_system (45M)/system.py",
                "phase2/master_system (45M)/llm_bridge.py",
                "phase2/master_system (45M)/autonomy/engine.py",
                "phase2/master_system (45M)/control/control.py",
            ),
        ),
        SystemComponent(
            name="self_improvement",
            layer="learning",
            role="45L improvement engine, dashboard, prompt/skill refinement, and session learning hooks.",
            paths=(
                "phase2/self_improvement (45l)/improve.py",
                "phase2/self_improvement (45l)/self_improvement/engine.py",
                "phase2/self_improvement (45l)/dashboard/dashboard.py",
            ),
        ),
        SystemComponent(
            name="self_modification",
            layer="governance",
            role="Type-A/Type-B patch gates, sandbox execution, audit logging, and atomic filesystem actions.",
            paths=("self_modification/type_a.py", "self_modification/type_b.py", "execution/sandbox.py", "execution/fs_agent.py"),
            import_name="self_modification.type_a",
        ),
        SystemComponent(
            name="ouroboros",
            layer="reflection",
            role="45O recursive reasoning, adaptive pass selection, pass gates, and milestone refinement.",
            paths=(
                "phase3/ouroboros (45O)/ouroboros_numpy.py",
                "phase3/ouroboros (45O)/adaptive.py",
                "phase3/ouroboros (45O)/pass_gates.py",
            ),
        ),
        SystemComponent(
            name="ghost_memory",
            layer="continuity",
            role="45P compressed conversation memory, retrieval, decay, and Ghost Context injection.",
            paths=(
                "phase3/ghost_memory (45P)/ghost_memory/memory_store.py",
                "phase3/ghost_memory (45P)/ghost_memory/retriever.py",
                "phase3/ghost_memory (45P)/ghost_memory/injector.py",
            ),
        ),
        SystemComponent(
            name="symbolic_bridge",
            layer="verification",
            role="45Q deterministic math, logic, code analysis, cross-checking, and verified response objects.",
            paths=(
                "phase3/symbolic_bridge (45Q)/symbolic_bridge.py",
                "phase3/symbolic_bridge (45Q)/math_solver.py",
                "phase3/symbolic_bridge (45Q)/logic_checker.py",
                "phase3/symbolic_bridge (45Q)/code_verifier.py",
            ),
        ),
        SystemComponent(
            name="sovereignty",
            layer="governance",
            role="45R audit, dead-code sweep, benchmark deltas, reports, and checkpoint promotion gates.",
            paths=(
                "phase3/sovereignty (45R)/sovereignty_bridge.py",
                "phase3/sovereignty (45R)/auditor.py",
                "phase3/sovereignty (45R)/benchmarks.py",
                "phase3/sovereignty (45R)/reporter.py",
            ),
        ),
    ]


def _is_source_file(path: Path) -> bool:
    rel_parts = set(path.relative_to(ROOT).parts)
    if rel_parts & IGNORED_PARTS:
        return False
    if path.suffix in IGNORED_SUFFIXES:
        return False
    if path.name in IGNORED_FILENAMES:
        return False
    return path.suffix in SOURCE_SUFFIXES


def source_files(root: Path = ROOT) -> list[Path]:
    return [path for path in sorted(root.rglob("*")) if path.is_file() and _is_source_file(path)]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def source_metrics(root: Path = ROOT) -> dict[str, int]:
    files = source_files(root)
    python_files = [p for p in files if p.suffix == ".py"]
    markdown_files = [p for p in files if p.suffix == ".md"]
    return {
        "source_files": len(files),
        "python_files": len(python_files),
        "markdown_files": len(markdown_files),
        "python_lines": sum(_read_text(p).count("\n") + 1 for p in python_files),
        "source_lines": sum(_read_text(p).count("\n") + 1 for p in files),
    }


def _path_entry(path_str: str) -> dict[str, object]:
    path = ROOT / path_str
    return {
        "path": path_str,
        "exists": path.exists(),
        "kind": "dir" if path.is_dir() else "file",
    }


def component_status(component: SystemComponent) -> dict[str, object]:
    paths = [_path_entry(path) for path in component.paths]
    missing = [p["path"] for p in paths if not p["exists"]]
    import_status = "not_checked"
    if component.import_name:
        try:
            inject_all_paths()
            mod = importlib.import_module(component.import_name)
            health = getattr(mod, "health_check", None)
            if callable(health):
                result = health()
                import_status = str(result.get("status", "ok")) if isinstance(result, dict) else "ok"
            else:
                import_status = "ok"
        except Exception as exc:
            import_status = f"degraded:{type(exc).__name__}"
    source_ok = not missing or not component.required
    runtime_ok = not import_status.startswith("degraded")
    return {
        **asdict(component),
        "paths": paths,
        "missing": missing,
        "import_status": import_status,
        "source_ok": source_ok,
        "runtime_ok": runtime_ok,
        "ok": source_ok,
    }


def artifact_status() -> dict[str, object]:
    reports = {
        "metrics": OUTPUT_V2_DIR / "reports" / "metrics.json",
        "eval_summary": OUTPUT_V2_DIR / "v2_eval_summary.json",
        "curriculum": OUTPUT_V2_DIR / "v2_next_session_curriculum.json",
    }
    checkpoints = {
        "brain": get_v2_checkpoint("brain"),
        "identity": get_v2_checkpoint("identity"),
        "ouroboros": get_v2_checkpoint("ouroboros"),
    }

    def file_info(path: Path) -> dict[str, object]:
        return {
            "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }

    return {
        "dataset": {
            "canonical": file_info(DATASET_CANONICAL),
        },
        "tokenizer": file_info(V3_TOKENIZER_FILE),
        "checkpoints": {name: file_info(path) for name, path in checkpoints.items()},
        "reports": {name: file_info(path) for name, path in reports.items()},
    }


def build_system_manifest(root: Path = ROOT) -> dict[str, object]:
    components = [component_status(component) for component in component_registry()]
    metrics = source_metrics(root)
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "metrics": metrics,
        "components": components,
        "capabilities": {c["name"]: bool(c["source_ok"]) for c in components},
        "artifacts": artifact_status(),
    }


def write_system_manifest(output: Path | None = None, root: Path = ROOT) -> dict[str, object]:
    manifest = build_system_manifest(root)
    target = output or (root / "system_graph.json")
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def missing_required_components(components: Iterable[dict[str, object]] | None = None) -> list[str]:
    rows = list(components) if components is not None else [component_status(c) for c in component_registry()]
    missing: list[str] = []
    for row in rows:
        if row.get("required") and row.get("missing"):
            missing.extend(str(item) for item in row["missing"])  # type: ignore[index]
    return missing
