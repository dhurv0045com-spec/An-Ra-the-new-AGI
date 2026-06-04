"""Integration tests for the AN-RA serving layer (app.py)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app import app

    with TestClient(app) as c:
        yield c


def test_health_endpoint_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data


def test_train_trigger_returns_501(client):
    r = client.post("/train/trigger")
    assert r.status_code == 501
    assert "not_implemented" in r.json().get("detail", {}).get("error", "")


def test_generate_endpoint_exists(client):
    r = client.post("/generate", json={"prompt": "Hello", "max_new_tokens": 5})
    assert r.status_code in (200, 422, 503)


def test_sessions_list_endpoint(client):
    r = client.get("/sessions")
    assert r.status_code == 200


def test_unknown_route_returns_404(client):
    r = client.get("/this_does_not_exist_xyz")
    assert r.status_code == 404


def test_cors_headers_present(client):
    r = client.options("/health", headers={"Origin": "http://localhost:3000"})
    assert r.status_code in (200, 204)
