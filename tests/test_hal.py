"""Tests for anra.identity.hal - HALModule registration and interface."""

from __future__ import annotations

from anra.core.registry import IDENTITY_REGISTRY


def test_hal_registered_by_name() -> None:
    assert "hal" in IDENTITY_REGISTRY


def test_hal_default_alias() -> None:
    assert "default" in IDENTITY_REGISTRY


def test_hal_registered_class_is_hal_module() -> None:
    from anra.identity.hal import HALModule

    assert IDENTITY_REGISTRY["hal"] is HALModule


def test_hal_and_default_resolve_same_class() -> None:
    assert IDENTITY_REGISTRY["hal"] is IDENTITY_REGISTRY["default"]


def test_hal_module_instantiable() -> None:
    from anra.identity.hal import HALModule

    instance = HALModule()
    assert instance is not None


def test_hal_module_has_all_export() -> None:
    import anra.identity.hal as m

    assert "HALModule" in m.__all__


def test_hal_canonical_import_path() -> None:
    """Must import from anra.identity.hal, never from scripts.hal or identity.hal directly."""
    from anra.identity.hal import HALModule  # noqa: F401


def test_hal_is_type() -> None:
    from anra.identity.hal import HALModule

    assert isinstance(HALModule, type)
