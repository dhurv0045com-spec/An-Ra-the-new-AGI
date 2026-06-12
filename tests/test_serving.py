"""Tests for anra.serving - FastAPI app factory."""

from __future__ import annotations

from fastapi import FastAPI


def test_create_app_returns_fastapi() -> None:
    from anra.serving import create_app

    app = create_app()
    assert isinstance(app, FastAPI)


def test_create_app_has_routes() -> None:
    from anra.serving import create_app

    app = create_app()
    routes = [r.path for r in app.routes]
    assert len(routes) > 0
    for path in (
        "/generate",
        "/goals",
        "/plans",
        "/memory",
        "/status",
        "/eval",
        "/training/candidates",
        "/robotics/workflows",
    ):
        assert path in routes


def test_app_attribute_accessible() -> None:
    from anra.serving import app

    assert app is not None


def test_sqlite_session_store_accessible() -> None:
    from anra.serving import SQLiteSessionStore

    assert SQLiteSessionStore is not None


def test_serving_import_canonical() -> None:
    """anra.serving must import cleanly."""
    import anra.serving  # noqa: F401


def test_serving_all_export() -> None:
    import anra.serving as m

    for name in ("app", "create_app", "SQLiteSessionStore"):
        assert name in m.__all__
