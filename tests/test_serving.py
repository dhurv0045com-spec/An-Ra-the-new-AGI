"""Tests for anra.serving - FastAPI app factory."""

from __future__ import annotations

from fastapi import FastAPI
from starlette.requests import Request


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


def test_production_service_fails_closed_without_owner_token(monkeypatch, tmp_path) -> None:
    import app as service

    monkeypatch.setattr(service, "STATE_DIR", tmp_path)
    monkeypatch.setenv("ANRA_SERVICE_MODE", "production")
    monkeypatch.delenv("ANRA_OWNER_TOKEN", raising=False)
    request = Request({"type": "http", "headers": []})

    assert service._owner_auth_required() is True
    assert service._authorized_owner(request) is False
