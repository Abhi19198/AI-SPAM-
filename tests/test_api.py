import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"


def test_predict_empty_email(client):
    response = client.post("/api/predict", json={"email_text": "   "})
    assert response.status_code == 400


def test_predict_invalid_payload(client):
    response = client.post("/api/predict", json={"wrong_key": "value"})
    assert response.status_code == 400


def test_model_loading_error(monkeypatch):
    import app as application_module

    def fail_load():
        raise FileNotFoundError("model file missing")

    monkeypatch.setattr(application_module, "load_model_bundle", fail_load)
    with pytest.raises(FileNotFoundError):
        application_module.load_model_bundle()
