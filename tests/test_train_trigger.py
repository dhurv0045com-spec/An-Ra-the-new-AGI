from app import app
from fastapi.testclient import TestClient
from training.v2_config import CANONICAL_MODEL_PROFILE


def test_train_trigger_creates_persistent_job(monkeypatch):
    monkeypatch.setattr("app.asyncio.create_task", lambda coroutine: coroutine.close())
    response = TestClient(app).post(
        "/train/trigger",
        json={"model_size": CANONICAL_MODEL_PROFILE, "minutes": 1},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
