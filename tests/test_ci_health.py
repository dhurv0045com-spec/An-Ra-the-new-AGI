"""
Canary tests that catch CI configuration problems before they reach GitHub Actions.
If these pass locally, CI will pass. Run first when debugging CI failures.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
        [sys.executable, "scripts/verify_structure.py"],
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
    search_dirs = ["anra", "identity", "memory", "training", "inference", "app.py", "generate.py", "anra.py"]
    for target in search_dirs:
        path = REPO / target
        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = list(path.rglob("*.py"))
        else:
            continue
        for f in files:
            for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if "sys.path.insert" in line or "sys.path.append" in line:
                    if "deprecated" not in str(f) and "#" not in line.lstrip()[:3]:
                        violations.append(f"{f.relative_to(REPO)}:{i}: {line.strip()}")
    assert not violations, "sys.path manipulation found in main package files:\n" + "\n".join(violations)


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
    from fastapi.testclient import TestClient

    from app import app

    with TestClient(app) as client:
        r = client.post("/train/trigger")
        assert r.status_code == 501, (
            f"Expected 501 (not implemented), got {r.status_code}. "
            "The train trigger must not pretend to do something it doesn't."
        )
