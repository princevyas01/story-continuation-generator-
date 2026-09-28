"""Autoregressive text generation and sampling engine."""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import tensorflow as tf
from src.tokenizer import StoryTokenizer
from src.preprocessing import normalize_text, tokenize_words

def apply_temperature_and_sampling(
    probabilities: np.ndarray,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.90,
    generated_token_ids: Optional[List[int]] = None,
    repetition_penalty: float = 1.15,
    rng: Optional[np.random.Generator] = None
) -> int:
    """Apply temperature scaling, repetition penalty, top-k, and top-p sampling.
    
    Returns selected token index.
    """
    if rng is None:
        rng = np.random.default_rng()

    probs = np.array(probabilities, dtype=np.float64)

    # 1. Repetition penalty: reduce probability of tokens that already appeared recently
    if repetition_penalty > 1.0 and generated_token_ids:
        # Count recent occurrences
        recent_tokens = generated_token_ids[-30:]
        for t_id in set(recent_tokens):
            if 0 <= t_id < len(probs):
                probs[t_id] /= repetition_penalty

    # 2. Temperature scaling
    # If temperature is near zero, use greedy argmax
    if temperature <= 0.05:
        return int(np.argmax(probs))

    # Convert probabilities to log-space safely for temperature scaling
    eps = 1e-12
    logits = np.log(np.maximum(probs, eps))
    logits = logits / float(temperature)

    # Numerically stable softmax
    logits = logits - np.max(logits)
    scaled_probs = np.exp(logits)
    sum_scaled = np.sum(scaled_probs)
    if sum_scaled > 0:
        scaled_probs = scaled_probs / sum_scaled
    else:
        scaled_probs = np.ones_like(scaled_probs) / len(scaled_probs)

    # 3. Top-k filtering
    if top_k > 0 and top_k < len(scaled_probs):
        top_k_indices = np.argsort(scaled_probs)[-top_k:]
        mask = np.zeros_like(scaled_probs, dtype=bool)
        mask[top_k_indices] = True
        scaled_probs[~mask] = 0.0
        sum_k = np.sum(scaled_probs)
        if sum_k > 0:
            scaled_probs /= sum_k

    # 4. Top-p (Nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(scaled_probs)[::-1]
        sorted_probs = scaled_probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        # Keep indices where cumulative probability <= top_p (plus at least one)
        cutoff_idx = np.searchsorted(cumulative_probs, top_p)
        kept_indices = sorted_indices[:cutoff_idx + 1]

        mask = np.zeros_like(scaled_probs, dtype=bool)
        mask[kept_indices] = True
        scaled_probs[~mask] = 0.0
        sum_p = np.sum(scaled_probs)
        if sum_p > 0:
            scaled_probs /= sum_p

    # Final safeguard against non-finite or all-zero probabilities
    if np.sum(scaled_probs) <= 0 or np.isnan(scaled_probs).any():
        return int(np.argmax(probabilities))

    scaled_probs /= np.sum(scaled_probs)
    return int(rng.choice(len(scaled_probs), p=scaled_probs))

def generate_continuation(
    model: Any,
    tokenizer: StoryTokenizer,
    prompt: str,
    seq_len: int = 50,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.90,
    repetition_penalty: float = 1.15,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Autoregressively generate story continuation from prompt."""
    start_time = time.time()
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    # Preprocess prompt
    clean_prompt = normalize_text(prompt)
    prompt_tokens = tokenize_words(clean_prompt)

    # Empty prompt fallback
    if not prompt_tokens:
        prompt_tokens = ["once", "upon", "a", "time"]

    # Encode prompt to tokens
    token_ids = tokenizer.encode(prompt_tokens)

    # Context window: keep most recent seq_len tokens
    # Left pad if shorter than seq_len
    if len(token_ids) < seq_len:
        current_input = [tokenizer.pad_idx] * (seq_len - len(token_ids)) + token_ids
    else:
        current_input = token_ids[-seq_len:]

    generated_ids: List[int] = []

    for _ in range(max_new_tokens):
        input_array = np.array([current_input[-seq_len:]], dtype=np.int32)
        
        # Predict next token distribution
        pred = model(input_array, training=False).numpy()[0]

        # Never sample PAD or START tokens during generation
        pred[tokenizer.pad_idx] = 0.0
        pred[tokenizer.start_idx] = 0.0
        sum_pred = np.sum(pred)
        if sum_pred > 0:
            pred /= sum_pred

        # Sample next token
        next_token_id = apply_temperature_and_sampling(
            probabilities=pred,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generated_token_ids=generated_ids,
            repetition_penalty=repetition_penalty,
            rng=rng
        )

        # Stop if END token reached
        if next_token_id == tokenizer.end_idx:
            break

        generated_ids.append(next_token_id)
        current_input.append(next_token_id)

    elapsed_sec = time.time() - start_time
    continuation_text = tokenizer.decode(generated_ids, skip_special=True)
    full_story = (clean_prompt + " " + continuation_text).strip()

    tokens_generated = len(generated_ids)
    latency_ms_per_token = (elapsed_sec * 1000.0 / tokens_generated) if tokens_generated > 0 else 0.0

    return {
        "prompt": prompt,
        "normalized_prompt": clean_prompt,
        "continuation": continuation_text,
        "full_story": full_story,
        "tokens_generated": tokens_generated,
        "elapsed_seconds": round(elapsed_sec, 3),
        "latency_ms_per_token": round(latency_ms_per_token, 2),
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "seed": seed
    }
