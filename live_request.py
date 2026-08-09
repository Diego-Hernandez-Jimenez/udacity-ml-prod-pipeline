"""
Sends a POST request to the live salary prediction API and prints
the status code and model inference result.

Usage:
    python live_request.py
"""

import requests

API_URL = "https://udacity-ml-prod-pipeline.onrender.com"

payload = {
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

response = requests.post(f"{API_URL}/predict", json=payload)

print(f"Status code : {response.status_code}")
print(f"Prediction  : {response.json()}")
