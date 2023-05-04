from api import app
from titanic_model.train import run
from fastapi.testclient import TestClient


# Make sure that the model exists
run()

client = TestClient(app)


def test_prediction_post():
    # Non surviving example
    response = client.post(
        "/predict",
        json={
            "sex": "male",
            "passenger_class": 3,
            "age": 90
        }
    )
    assert response.status_code == 200
    assert response.json() == {"survives": 0}

    # Surviving example
    response = client.post(
        "/predict",
        json={
            "sex": "female",
            "passenger_class": 1,
            "age": 15
        }
    )
    assert response.status_code == 200
    assert response.json() == {"survives": 1}

    # Test validation
    # Invalid sex input
    response = client.post(
        "/predict",
        json={
            "sex": "???",
            "passenger_class": 1,
            "age": 30
        }
    )
    assert response.status_code == 422

    # Invalid passenger_class input
    response = client.post(
        "/predict",
        json={
            "sex": "female",
            "passenger_class": 9,
            "age": 30
        }
    )
    assert response.status_code == 422

    # Age too high
    response = client.post(
        "/predict",
        json={
            "sex": "female",
            "passenger_class": 1,
            "age": 120
        }
    )
    assert response.status_code == 422

    # Age too low
    response = client.post(
        "/predict",
        json={
            "sex": "female",
            "passenger_class": 1,
            "age": -5
        }
    )
    assert response.status_code == 422
