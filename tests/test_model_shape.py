"""Tests for model building, layers, output shapes, and compilation."""

import pytest
import numpy as np
from src.model import build_model, compile_model, get_model_summary_string

def test_model_build_and_shapes():
    vocab_size = 500
    seq_len = 20
    embedding_dim = 64
    lstm_units = 128
    
    model = build_model(
        vocab_size=vocab_size,
        seq_len=seq_len,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        num_layers=2,
        dropout_rate=0.1
    )
    
    # Check input shape
    assert model.input_shape == (None, seq_len)
    # Check output shape
    assert model.output_shape == (None, vocab_size)
    
    # Forward pass test
    dummy_input = np.random.randint(0, vocab_size, size=(2, seq_len), dtype=np.int32)
    output = model(dummy_input, training=False)
    assert output.shape == (2, vocab_size)
    # Check probabilities sum to 1.0 approximately
    sums = np.sum(output.numpy(), axis=-1)
    np.testing.assert_allclose(sums, [1.0, 1.0], atol=1e-5)

def test_compile_model():
    model = build_model(vocab_size=100, seq_len=10, embedding_dim=32, lstm_units=64)
    compiled = compile_model(model, learning_rate=0.001)
    assert compiled.optimizer is not None
    assert compiled.loss is not None

def test_model_summary_string():
    model = build_model(vocab_size=100, seq_len=10, embedding_dim=32, lstm_units=64)
    summary_text = get_model_summary_string(model)
    assert "word_embedding" in summary_text
    assert "lstm_layer_1" in summary_text
    assert "next_token_softmax" in summary_text
