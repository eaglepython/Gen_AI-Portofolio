import requests
import pytest

BASE_URL = "http://localhost:8000"
SAMPLE_INPUT = {"features": [0.5, 100000, 5, 1, 0]}

def test_score_pd():
    response = requests.post(f"{BASE_URL}/score/pd", json=SAMPLE_INPUT)
    assert response.status_code == 200
    result = response.json()
    assert "pd" in result
    assert 0 <= result["pd"] <= 1

def test_score_lgd():
    response = requests.post(f"{BASE_URL}/score/lgd", json=SAMPLE_INPUT)
    assert response.status_code == 200
    result = response.json()
    assert "lgd" in result
    assert 0 <= result["lgd"] <= 1

def test_score_ead():
    response = requests.post(f"{BASE_URL}/score/ead", json=SAMPLE_INPUT)
    assert response.status_code == 200
    result = response.json()
    assert "ead" in result
    assert result["ead"] >= 0
