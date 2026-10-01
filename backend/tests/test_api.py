"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data

def test_model_status_endpoint():
    response = client.get("/api/v1/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "story_continuation_lstm"
    assert "vocab_size" in data

def test_model_metrics_endpoint():
    response = client.get("/api/v1/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data

def test_generate_validation_empty_prompt():
    response = client.post("/api/v1/generate", json={
        "prompt": "   ",
        "max_new_tokens": 20
    })
    assert response.status_code == 422

def test_generate_validation_temperature_zero():
    response = client.post("/api/v1/generate", json={
        "prompt": "Once upon a time",
        "temperature": 0.0
    })
    assert response.status_code == 422

def test_generate_validation_tokens_too_large():
    response = client.post("/api/v1/generate", json={
        "prompt": "Once upon a time",
        "max_new_tokens": 5000
    })
    assert response.status_code == 422

def test_generate_valid_prompt():
    response = client.post("/api/v1/generate", json={
        "prompt": "Once upon a time",
        "max_new_tokens": 5,
        "temperature": 0.8,
        "top_k": 20,
        "seed": 42
    })
    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    assert "runtime_ms" in data
    assert data["settings"]["max_new_tokens"] == 5
