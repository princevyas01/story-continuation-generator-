"""Tests for autoregressive sampling and generation engine."""

import pytest
import numpy as np
from src.tokenizer import StoryTokenizer
from src.model import build_model, compile_model
from src.generate import apply_temperature_and_sampling, generate_continuation

def test_apply_temperature_and_sampling_greedy():
    # Deterministic test: highest probability should always be chosen when temp is near 0
    probs = np.array([0.1, 0.7, 0.15, 0.05])
    token = apply_temperature_and_sampling(probs, temperature=0.01)
    assert token == 1

def test_top_k_sampling_bounds():
    probs = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
    # Top-k = 2 should only choose index 0 or 1
    rng = np.random.default_rng(42)
    chosen_indices = set()
    for _ in range(50):
        token = apply_temperature_and_sampling(probs, temperature=1.0, top_k=2, rng=rng)
        chosen_indices.add(token)
    assert chosen_indices.issubset({0, 1})

def test_generate_continuation_empty_prompt():
    tok = StoryTokenizer(max_vocab_size=100)
    tok.fit_on_texts([["once", "upon", "a", "time", "there", "lived", "a", "king"]])
    model = build_model(vocab_size=tok.vocab_size, seq_len=10, embedding_dim=16, lstm_units=32, num_layers=1)
    compile_model(model)

    # Empty prompt should not crash
    res = generate_continuation(model, tok, prompt="", seq_len=10, max_new_tokens=5)
    assert isinstance(res["continuation"], str)
    assert res["tokens_generated"] <= 5

def test_generation_reproducibility_with_seed():
    tok = StoryTokenizer(max_vocab_size=100)
    tok.fit_on_texts([["the", "dragon", "flew", "over", "the", "mountains", "into", "the", "clouds"]])
    model = build_model(vocab_size=tok.vocab_size, seq_len=5, embedding_dim=16, lstm_units=32, num_layers=1)
    compile_model(model)

    res1 = generate_continuation(model, tok, prompt="the dragon", seq_len=5, max_new_tokens=6, seed=1234)
    res2 = generate_continuation(model, tok, prompt="the dragon", seq_len=5, max_new_tokens=6, seed=1234)
    assert res1["continuation"] == res2["continuation"]
