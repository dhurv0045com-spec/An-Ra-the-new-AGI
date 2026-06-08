from app import app
from fastapi.testclient import TestClient


def test_train_trigger_returns_501():
    response = TestClient(app).post("/train/trigger")
    assert response.status_code == 501
    assert response.json()["detail"]["error"] == "training_dispatch_not_implemented"
