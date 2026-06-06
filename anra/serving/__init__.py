"""
anra/serving — HTTP API serving layer.

The production FastAPI application lives in app.py at the project root
for backward compatibility. This package provides the clean import path
and re-exports the app factory for use in tests and deployment scripts.

Usage:
    from anra.serving import create_app, SQLiteSessionStore
    app = create_app()
"""

from __future__ import annotations


def create_app():
    """Return the configured FastAPI application instance."""
    from app import app

    return app


def __getattr__(name: str):
    if name == "app":
        from app import app

        return app
    if name == "SQLiteSessionStore":
        from app import SQLiteSessionStore

        return SQLiteSessionStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app", "create_app", "SQLiteSessionStore"]
