"""Evaluation metrics: Perplexity, Distinct-1, Distinct-2, Repetition, and Latency."""

import math
from typing import List, Dict, Any, Union

def calculate_perplexity(loss: float, max_loss: float = 100.0) -> float:
    """Calculate perplexity from cross-entropy loss with overflow protection.
    
    Perplexity represents the effective branching factor / uncertainty of the model:
      PPL = exp(CrossEntropyLoss)
    """
    if math.isnan(loss) or math.isinf(loss):
        return float("inf")
    # Clamp loss to prevent float overflow in exp()
    clamped_loss = min(loss, max_loss)
    try:
        return float(math.exp(clamped_loss))
    except OverflowError:
        return float("inf")

def compute_distinct_n(tokens: List[str], n: int = 1) -> float:
    """Compute Distinct-n lexical diversity score (ratio of unique n-grams to total n-grams).
    
    Score is bounded in [0.0, 1.0]. Higher values indicate higher lexical diversity.
    """
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / float(len(ngrams))

def compute_repetition_ratio(tokens: List[str], n: int = 3) -> float:
    """Compute the repeated n-gram ratio.
    
    Returns proportion of n-grams that are repeated occurrences: 1.0 - (unique ngrams / total ngrams).
    """
    if len(tokens) < n:
        return 0.0
    distinct = compute_distinct_n(tokens, n=n)
    return round(1.0 - distinct, 4)

def evaluate_text_generation_quality(text: str) -> Dict[str, Any]:
    """Compute lexical and repetition metrics for a generated text sample."""
    tokens = [w.strip() for w in text.split() if w.strip()]
    total_tokens = len(tokens)
    
    if total_tokens == 0:
        return {
            "total_tokens": 0,
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "distinct_3": 0.0,
            "repetition_ratio_3gram": 0.0
        }

    d1 = round(compute_distinct_n(tokens, n=1), 4)
    d2 = round(compute_distinct_n(tokens, n=2), 4)
    d3 = round(compute_distinct_n(tokens, n=3), 4)
    rep3 = round(compute_repetition_ratio(tokens, n=3), 4)

    return {
        "total_tokens": total_tokens,
        "distinct_1": d1,
        "distinct_2": d2,
        "distinct_3": d3,
        "repetition_ratio_3gram": rep3
    }
