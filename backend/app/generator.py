"""Generation engine for autoregressive LSTM story continuation."""

import time
import numpy as np
from typing import List, Optional, Tuple
from ml.src.tokenizer import StoryTokenizer, clean_and_tokenize

def sample_next_token(
    probabilities: np.ndarray,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.90,
    generated_token_ids: Optional[List[int]] = None,
    repetition_penalty: float = 1.15,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Sample next token index using temperature scaling, top-k, and nucleus (top-p) filtering."""
    if rng is None:
        rng = np.random.default_rng()

    probs = np.array(probabilities, dtype=np.float64)

    # 1. Repetition penalty
    if repetition_penalty > 1.0 and generated_token_ids:
        recent = generated_token_ids[-30:]
        for t_id in set(recent):
            if 0 <= t_id < len(probs):
                probs[t_id] /= repetition_penalty

    # 2. Greedy if temperature near zero
    if temperature <= 0.05:
        return int(np.argmax(probs))

    # 3. Temperature scaling
    eps = 1e-12
    logits = np.log(np.maximum(probs, eps)) / float(temperature)
    logits = logits - np.max(logits)
    scaled_probs = np.exp(logits)
    s = np.sum(scaled_probs)
    scaled_probs = scaled_probs / s if s > 0 else np.ones_like(scaled_probs) / len(scaled_probs)

    # 4. Top-k filtering
    if 0 < top_k < len(scaled_probs):
        top_indices = np.argsort(scaled_probs)[-top_k:]
        mask = np.zeros_like(scaled_probs, dtype=bool)
        mask[top_indices] = True
        scaled_probs[~mask] = 0.0
        sk = np.sum(scaled_probs)
        if sk > 0:
            scaled_probs /= sk

    # 5. Top-p (Nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(scaled_probs)[::-1]
        sorted_probs = scaled_probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative, top_p)
        kept = sorted_indices[:cutoff + 1]
        mask = np.zeros_like(scaled_probs, dtype=bool)
        mask[kept] = True
        scaled_probs[~mask] = 0.0
        sp = np.sum(scaled_probs)
        if sp > 0:
            scaled_probs /= sp

    total_p = np.sum(scaled_probs)
    if total_p <= 0 or np.isnan(total_p):
        return int(np.argmax(probabilities))

    scaled_probs /= total_p
    return int(rng.choice(len(scaled_probs), p=scaled_probs))

def run_generation(
    model,
    tokenizer: StoryTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: Optional[float] = 0.90,
    seed: Optional[int] = None,
    seq_len: int = 50
) -> Tuple[str, int]:
    """Execute autoregressive generation loop.
    
    Returns (generated_text, runtime_ms).
    """
    start_time = time.time()
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    tokens = clean_and_tokenize(prompt)
    if not tokens:
        tokens = ["once", "upon", "a", "time"]

    token_ids = tokenizer.encode(tokens)
    if len(token_ids) < seq_len:
        current_input = [tokenizer.pad_idx] * (seq_len - len(token_ids)) + token_ids
    else:
        current_input = token_ids[-seq_len:]

    generated_ids: List[int] = []

    for _ in range(max_new_tokens):
        x = np.array([current_input[-seq_len:]], dtype=np.int32)
        pred = model(x, training=False).numpy()[0]

        # Prevent PAD and START tokens from being sampled
        pred[tokenizer.pad_idx] = 0.0
        pred[tokenizer.start_idx] = 0.0
        s = np.sum(pred)
        if s > 0:
            pred /= s

        next_id = sample_next_token(
            probabilities=pred,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generated_token_ids=generated_ids,
            rng=rng
        )

        if next_id == tokenizer.end_idx:
            break

        generated_ids.append(next_id)
        current_input.append(next_id)

    runtime_ms = int((time.time() - start_time) * 1000)
    generated_text = tokenizer.decode(generated_ids, skip_special=True)
    return generated_text, runtime_ms
