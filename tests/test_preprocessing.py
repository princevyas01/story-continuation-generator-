"""Tests for text normalization, cleaning, and document-level splitting."""

import pytest
from src.preprocessing import normalize_text, strip_gutenberg_boilerplate, tokenize_words, split_documents

def test_normalize_text():
    sample = "“Once upon a time,” said the King—‘Look here!’"
    normalized = normalize_text(sample)
    # Quotes normalized, punctuation padded with spaces, lowercase
    assert "\"" in normalized
    assert "'" in normalized
    assert "upon a time" in normalized
    assert normalized.islower()

def test_strip_gutenberg_boilerplate():
    raw = """*** START OF THE PROJECT GUTENBERG EBOOK 2591 ***
    Once upon a time there was a princess.
    *** END OF THE PROJECT GUTENBERG EBOOK 2591 ***"""
    stripped = strip_gutenberg_boilerplate(raw)
    assert "START OF THE PROJECT" not in stripped
    assert "END OF THE PROJECT" not in stripped
    assert "Once upon a time there was a princess." in stripped

def test_tokenize_words():
    text = "the quick brown fox ."
    tokens = tokenize_words(text)
    assert tokens == ["the", "quick", "brown", "fox", "."]

def test_split_documents():
    docs = [{"id": f"d_{i}", "text": "story"} for i in range(20)]
    train, val, test = split_documents(docs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    assert len(train) == 16
    assert len(val) == 2
    assert len(test) == 2

    # Check leakage prevention: disjoint sets of document IDs
    train_ids = {d["id"] for d in train}
    val_ids = {d["id"] for d in val}
    test_ids = {d["id"] for d in test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
