"""
anra/serving — HTTP API serving layer.

The production FastAPI application lives in app.py.
This package provides the clean import path and re-exports the app factory.
"""

from __future__ import annotations


def create_app() -> object:
    """Return the configured FastAPI application instance."""
    from app import app

    return app


def __getattr__(name: str) -> object:
    if name == "app":
        from app import app

        return app
    if name == "SQLiteSessionStore":
        from app import SQLiteSessionStore

        return SQLiteSessionStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app", "create_app", "SQLiteSessionStore"]
