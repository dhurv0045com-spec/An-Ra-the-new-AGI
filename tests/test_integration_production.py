"""Production integration tests - import chain, config, and structural correctness."""

from __future__ import annotations

import ast
import importlib
import pathlib


def test_anra_package_imports_cleanly() -> None:
    import anra

    assert anra.__version__ is not None


def test_anra_config_instantiable() -> None:
    from anra.core.config import AnRaConfig

    cfg = AnRaConfig()
    assert cfg is not None


def test_model_py_imports_from_anra_brain_not_scripts() -> None:
    """anra/core/model.py must NOT import from scripts.*"""
    src = pathlib.Path("anra/core/model.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("scripts"), (
                f"anra/core/model.py must not import from scripts.*, found: {node.module}"
            )


def test_serving_imports_from_app_not_scripts() -> None:
    """anra/serving/__init__.py must NOT import from scripts.*"""
    src = pathlib.Path("anra/serving/__init__.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("scripts"), (
                f"anra/serving/__init__.py must not import from scripts.*, found: {node.module}"
            )


def test_inference_imports_from_generate_not_scripts() -> None:
    """anra/inference/__init__.py must NOT import from scripts.*"""
    src = pathlib.Path("anra/inference/__init__.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("scripts"), (
                f"anra/inference/__init__.py must not import from scripts.*, found: {node.module}"
            )


def test_anra_training_exports_trainers() -> None:
    from anra.training import RLVRTrainer, STaRLoop

    assert RLVRTrainer is not None
    assert STaRLoop is not None


def test_scripts_not_in_pyproject_include() -> None:
    """pyproject.toml must not include scripts* as an installable package."""
    try:
        tomllib = importlib.import_module("tomllib")
    except ModuleNotFoundError:
        tomllib = importlib.import_module("tomli")

    data = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
    include = data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {}).get("include", [])
    for entry in include:
        assert not entry.startswith("scripts"), (
            f"pyproject.toml must not include scripts* as a package, found: {entry}"
        )
