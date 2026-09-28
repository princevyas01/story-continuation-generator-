"""Tests for perplexity, lexical diversity, and repetition metrics."""

import math
import pytest
from src.metrics import calculate_perplexity, compute_distinct_n, compute_repetition_ratio, evaluate_text_generation_quality

def test_calculate_perplexity():
    # ppl = exp(loss)
    loss = 2.0
    ppl = calculate_perplexity(loss)
    assert math.isclose(ppl, math.exp(2.0), rel_tol=1e-4)

    # Overflow protection
    large_ppl = calculate_perplexity(500.0)
    assert not math.isnan(large_ppl)

def test_distinct_n():
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    # Distinct-1: 5 unique words out of 6 total words
    d1 = compute_distinct_n(tokens, n=1)
    assert math.isclose(d1, 5 / 6, rel_tol=1e-4)

    # Distinct-2: bigrams ("the", "cat"), ("cat", "sat"), ("sat", "on"), ("on", "the"), ("the", "mat") -> 5 unique / 5 total
    d2 = compute_distinct_n(tokens, n=2)
    assert math.isclose(d2, 1.0, rel_tol=1e-4)

def test_repetition_ratio():
    repetitive_tokens = ["the", "king", "the", "king", "the", "king"]
    rep = compute_repetition_ratio(repetitive_tokens, n=2)
    assert rep > 0.0

def test_evaluate_text_generation_quality():
    text = "once upon a time in a far castle lived a kind prince"
    res = evaluate_text_generation_quality(text)
    assert "distinct_1" in res
    assert "distinct_2" in res
    assert "repetition_ratio_3gram" in res
    assert res["total_tokens"] > 0
