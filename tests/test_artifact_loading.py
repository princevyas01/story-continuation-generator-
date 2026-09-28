"""Tests for artifact presence, serialization, and structure."""

import os
import json
import pytest
from src.config import load_config, resolve_path
from src.tokenizer import StoryTokenizer

def test_manifest_and_vocab_exist():
    cfg = load_config()
    bdir = cfg["base_dir"]
    manifest_p = resolve_path(bdir, cfg["data"]["manifest_path"])
    vocab_p = resolve_path(bdir, cfg["tokenization"]["vocab_path"])

    assert os.path.exists(manifest_p), f"Manifest missing at {manifest_p}"
    assert os.path.exists(vocab_p), f"Vocab artifact missing at {vocab_p}"

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "source_name" in manifest
    assert "document_count" in manifest
    assert "vocabulary_size" in manifest
    assert "story_split_mapping" in manifest

    tokenizer = StoryTokenizer.load(vocab_p)
    assert tokenizer.vocab_size == manifest["vocabulary_size"]
