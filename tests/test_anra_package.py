"""
Tests that importing 'anra' triggers all registry registrations correctly.
This is the canary for circular import bugs and missing __init__.py chains.
"""
from __future__ import annotations

from pathlib import Path


def test_import_anra_succeeds():
    import anra

    assert anra.__version__ == "0.3.0"


def test_model_registry_has_causal_transformer():
    import anra

    assert "causal_transformer_v2" in anra.MODEL_REGISTRY
    registered = anra.MODEL_REGISTRY.list()
    assert len(registered) >= 1


def test_identity_registry_has_hal():
    import anra

    assert "hal" in anra.IDENTITY_REGISTRY


def test_identity_registry_has_hal_esv_civ() -> None:
    import anra

    identities = anra.IDENTITY_REGISTRY.list()
    assert "hal" in identities, f"hal missing from IDENTITY_REGISTRY: {identities}"
    assert "esv" in identities, f"esv missing from IDENTITY_REGISTRY: {identities}"
    assert "civ" in identities, f"civ missing from IDENTITY_REGISTRY: {identities}"


def test_memory_registry_has_memory_router():
    import anra

    assert "memory_router" in anra.MEMORY_REGISTRY


def test_memory_registry_has_bm25() -> None:
    import anra

    assert "bm25" in anra.MEMORY_REGISTRY


def test_make_train_tiny_config_is_valid() -> None:
    from anra.core.config import AnRaConfig

    cfg = AnRaConfig.from_yaml(Path("config/tiny.yaml"))
    assert cfg.model.n_embd == 128
    assert cfg.model.n_layer == 4
    assert cfg.training.learning_rate == 1e-3


def test_config_protocol_from_package():
    from anra import AnRaConfig, ModelConfig

    cfg = AnRaConfig(experiment_name="test")
    assert cfg.experiment_name == "test"
    assert isinstance(cfg.model, ModelConfig)


def test_serving_importable_from_package():
    from anra.serving import create_app

    assert callable(create_app)


def test_inference_importable_from_package():
    from anra.inference import generate_stream, generate_traced

    assert callable(generate_traced)
    assert callable(generate_stream)


def test_schemas_importable():
    from anra.serving.schemas import GenerateRequest

    r = GenerateRequest(prompt="test")
    assert r.prompt == "test"


def test_no_wildcard_star_imports_in_anra_package():
    repo = Path(__file__).resolve().parents[1]
    violations = []
    for f in (repo / "anra").rglob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("from ") and "import *" in stripped:
                violations.append(f"{f.relative_to(repo)}:{i}: {stripped}")
    assert not violations, "Wildcard imports in anra package:\n" + "\n".join(violations)
