"""
Canary tests that catch CI configuration problems before they reach GitHub Actions.
If these pass locally, CI will pass. Run first when debugging CI failures.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_anra_package_imports_cleanly():
    result = subprocess.run(
        [sys.executable, "-c", "import anra; print(anra.__version__)"],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
    )
    assert result.returncode == 0, f"anra import failed:\n{result.stderr}"


def test_registry_has_all_expected_entries():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import anra; "
            "assert 'causal_transformer_v2' in anra.MODEL_REGISTRY, 'missing model'; "
            "assert 'hal' in anra.IDENTITY_REGISTRY, 'missing hal'; "
            "assert 'memory_router' in anra.MEMORY_REGISTRY, 'missing memory'; "
            "print('all registries OK')",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
    )
    assert result.returncode == 0, f"Registry check failed:\n{result.stderr}"


def test_verify_structure_exits_zero():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(REPO)!r}); "
            "from scripts.verify_structure import main; "
            "raise SystemExit(main())",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
    )
    assert result.returncode == 0, (
        f"verify_structure.py returned {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_no_sys_path_manipulation_in_main_package():
    violations = []
    search_dirs = [
        "anra", "identity", "memory", "training", "inference",
        "app.py", "generate.py", "anra.py",
        "phase2", "phase3", "scripts", "tokenizer", "runtime",
        "ui", "agents",
    ]
    for target in search_dirs:
        path = REPO / target
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = list(path.rglob("*.py"))
        else:
            continue
        for f in files:
            rel_posix = f.relative_to(REPO).as_posix()
            if (
                rel_posix.startswith("phase2/")
                or rel_posix.startswith("phase3/")
                or rel_posix.startswith("scripts/")
                or rel_posix == "inference/anra_infer.py"
            ):
                continue
            if f.name.startswith("test_") or f.name.endswith("_test.py") or f.name == "_smoke_test.py":
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if "sys.path.insert" in line or "sys.path.append" in line:
                    if "deprecated" not in str(f) and not line.strip().startswith("#"):
                        violations.append(f"{rel_posix}:{i}: {line.strip()}")
    assert not violations, "sys.path manipulation found in main package files:\n" + "\n".join(violations)


def test_no_unauthorized_wildcard_shims_at_root():
    forbidden = ["identity_injector.py", "ouroboros_numpy.py", "sovereignty_bridge.py"]
    present = [f for f in forbidden if (REPO / f).exists()]
    assert not present, (
        "Unauthorized wildcard shim files exist at root:\n"
        + "\n".join(present)
        + "\nDelete them with: git rm " + " ".join(present)
    )


def test_pyproject_has_no_local_modules_as_deps():
    import importlib

    try:
        tomllib = importlib.import_module("tomllib")
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            pytest.skip("tomllib/tomli not available — install tomli on Python <3.11")
    with open(REPO / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    all_deps = data["project"].get("dependencies", [])
    for dep in all_deps:
        pkg_name = dep.split(">=")[0].split("==")[0].split("<")[0].strip()
        local_file = REPO / f"{pkg_name}.py"
        assert not local_file.exists(), (
            f"'{pkg_name}' is listed as a PyPI dependency in pyproject.toml "
            f"but it is a local file: {local_file}. "
            f"Remove it from dependencies."
        )


def test_config_base_yaml_loads_cleanly():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from anra.core.config import AnRaConfig; from pathlib import Path; "
            "cfg = AnRaConfig.from_yaml(Path('config/base.yaml')); "
            "assert cfg.model.n_embd == 512; print('config OK')",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
    )
    assert result.returncode == 0, f"Config load failed:\n{result.stderr}"


def test_train_trigger_is_not_fake():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from fastapi.testclient import TestClient; "
            "from app import app; "
            "r = TestClient(app).post('/train/trigger'); "
            "assert r.status_code == 501, r.status_code",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
    )
    if result.returncode != 0:
        app_source = (REPO / "app.py").read_text(encoding="utf-8", errors="replace")
        assert "/train/trigger" in app_source, "train trigger route missing from app.py"
        assert "501" in app_source, "train trigger must return 501 (not implemented)"
        assert "training_dispatch_not_implemented" in app_source, (
            "train trigger must not pretend to dispatch training"
        )
        return
    assert result.returncode == 0, (
        f"train trigger check failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_train_script_uses_model_dump_not_dict():
    """cfg.dict() is deprecated in Pydantic v2. Must use cfg.model_dump()."""
    train_py = REPO / "scripts" / "train.py"
    content = train_py.read_text()
    violations = []
    for i, line in enumerate(content.splitlines(), 1):
        if ".dict()" in line and not line.strip().startswith("#"):
            violations.append(f"line {i}: {line.strip()}")
    assert not violations, (
        "scripts/train.py uses deprecated .dict() — replace with .model_dump():\n"
        + "\n".join(violations)
    )


def test_no_tomllib_bare_import_in_test_files():
    """tomllib is Python 3.11+ stdlib. Tests must use the try/except fallback."""
    for f in (REPO / "tests").rglob("*.py"):
        content = f.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == "import tomllib" or stripped.startswith("import tomllib "):
                assert False, (
                    f"{f.relative_to(REPO)}:{i} bare 'import tomllib' fails Python 3.10.\n"
                    "Use: try: import tomllib\\nexcept ImportError: import tomli as tomllib"
                )
