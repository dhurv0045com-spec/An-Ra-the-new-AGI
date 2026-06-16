from app import app
from fastapi.testclient import TestClient


def test_train_trigger_creates_persistent_job(monkeypatch):
    monkeypatch.setattr("app.asyncio.create_task", lambda coroutine: coroutine.close())
    response = TestClient(app).post(
        "/train/trigger",
        json={"model_size": "frontier", "minutes": 1},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
