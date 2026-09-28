"""Tests for configuration loading and validation."""

import os
import pytest
from src.config import load_config, resolve_path

def test_load_config_default():
    config = load_config()
    assert isinstance(config, dict)
    assert "project" in config
    assert "data" in config
    assert "tokenization" in config
    assert "model" in config
    assert "training" in config
    assert "generation" in config
    assert "artifacts" in config

def test_config_values():
    config = load_config()
    assert config["tokenization"]["sequence_length"] > 0
    assert config["model"]["embedding_dim"] > 0
    assert config["model"]["lstm_units"] > 0
    assert 0.0 <= config["model"]["dropout_rate"] < 1.0
    assert config["training"]["batch_size"] > 0

def test_resolve_path():
    base_dir = "E:\\mdm"
    rel_path = "configs\\config.yaml"
    resolved = resolve_path(base_dir, rel_path)
    assert os.path.isabs(resolved)
    assert resolved.endswith("config.yaml")
