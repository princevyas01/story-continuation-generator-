"""Unit tests for ML components: Tokenizer, Sequences, Model, and Sampling."""

import pytest
import numpy as np
import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.src.tokenizer import StoryTokenizer, clean_and_tokenize
from ml.src.build_sequences import create_sequences_from_token_ids
from ml.src.model import build_model
from ml.src.generate_cli import sample_next_token

def test_tokenizer_encode_decode():
    tok = StoryTokenizer(max_vocab_size=100)
    texts = [["once", "upon", "a", "time", "in", "a", "forest"]]
    tok.fit_on_texts(texts)

    # Known token mapping
    tokens = ["once", "upon", "a", "time"]
    encoded = tok.encode(tokens)
    assert len(encoded) == 4
    assert all(isinstance(x, int) for x in encoded)

    decoded = tok.decode(encoded)
    assert "once upon a time" in decoded

def test_tokenizer_unknown_token():
    tok = StoryTokenizer(max_vocab_size=10)
    tok.fit_on_texts([["known", "word"]])
    encoded = tok.encode(["unknownwordxyz"])
    assert encoded[0] == tok.unk_idx

def test_sequence_builder():
    token_ids = [10, 20, 30, 40, 50, 60]
    seq_len = 3
    x, y = create_sequences_from_token_ids(token_ids, seq_len=seq_len, step=1)
    assert len(x) == 3
    assert len(y) == 3
    np.testing.assert_array_equal(x[0], [10, 20, 30])
    assert y[0] == 40

def test_model_output_shape():
    vocab_size = 50
    seq_len = 10
    model = build_model(vocab_size=vocab_size, seq_len=seq_len, embedding_dim=16, lstm_units=32, num_layers=1)
    dummy_input = np.ones((2, seq_len), dtype=np.int32)
    output = model(dummy_input)
    assert output.shape == (2, vocab_size)

def test_sample_next_token_bounds():
    probs = np.array([0.1, 0.2, 0.7])
    sampled = sample_next_token(probs, temperature=1.0, top_k=2)
    assert sampled in [1, 2]
