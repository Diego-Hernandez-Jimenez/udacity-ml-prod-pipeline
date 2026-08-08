"""
Unit tests for inference API.

Author: Diego Hernández Jiménez
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_welcome():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the salary predictions API!"}


def test_predict_negative_class():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    response = client.post("predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}


def test_predict_positive_class():
    data = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States",
    }
    response = client.post("predict", json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": ">=50K"}
