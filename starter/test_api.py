import pytest
from fastapi.testclient import TestClient
from starter.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_get_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the app of census data classification"
    }


def test_predict_output_0(client, json_sample_1):
    response = client.post("/predict", json=json_sample_1)
    assert response.status_code == 200
    assert response.json()["prediction"] == 0


def test_predict_output_1(client, json_sample_2):
    response = client.post("/predict", json=json_sample_2)
    assert response.status_code == 200
    assert response.json()["prediction"] == 1


def test_predict_error_422(client, json_sample_with_error):
    response = client.post("/predict", json=json_sample_with_error)
    assert response.status_code == 422
