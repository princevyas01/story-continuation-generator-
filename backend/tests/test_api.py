"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data

def test_model_status_endpoint(client):
    response = client.get("/api/v1/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "story_continuation_lstm"
    assert "vocab_size" in data

def test_model_metrics_endpoint(client):
    response = client.get("/api/v1/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data

def test_generate_validation_empty_prompt(client):
    response = client.post("/api/v1/generate", json={
        "prompt": "   ",
        "max_new_tokens": 20
    })
    assert response.status_code == 422

def test_generate_validation_temperature_zero(client):
    response = client.post("/api/v1/generate", json={
        "prompt": "Once upon a time",
        "temperature": 0.0
    })
    assert response.status_code == 422

def test_generate_validation_tokens_too_large(client):
    response = client.post("/api/v1/generate", json={
        "prompt": "Once upon a time",
        "max_new_tokens": 5000
    })
    assert response.status_code == 422

def test_generate_valid_prompt(client):
    response = client.post("/api/v1/generate", json={
        "prompt": "Once upon a time",
        "max_new_tokens": 5,
        "temperature": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.2,
        "seed": 42
    })
    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    assert "runtime_ms" in data
    assert data["settings"]["max_new_tokens"] == 5
    assert data["settings"]["repetition_penalty"] == 1.2

def test_compare_endpoint(client):
    response = client.post("/api/v1/generate/compare", json={
        "prompt": "Once upon a time",
        "max_new_tokens": 5,
        "settings_list": [
            {"max_new_tokens": 5, "temperature": 0.3, "top_k": 20, "top_p": 0.85, "repetition_penalty": 1.15, "seed": 42},
            {"max_new_tokens": 5, "temperature": 0.9, "top_k": 40, "top_p": 0.95, "repetition_penalty": 1.15, "seed": 42}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
