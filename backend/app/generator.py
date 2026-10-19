"""High-performance generation engine for autoregressive Story LSTM (v1 & v2).

Features:
- Numerically stable logit-space temperature scaling
- Logit-space repetition penalty (Keskar et al.)
- Top-K truncation
- Descending cumulative Nucleus (Top-P) sampling
- Zero emission of PAD/BOS tokens
- Immediate stopping on EOS
- Native support for both SubwordStoryTokenizer (v2) and StoryTokenizer (v1)
"""

import time
from typing import List, Optional, Set, Tuple, Union
import numpy as np
import tensorflow as tf

from ml.src.subword_tokenizer import SubwordStoryTokenizer
from ml.src.tokenizer import StoryTokenizer, clean_and_tokenize


def sample_next_token(
    logits_or_probs: np.ndarray,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: Optional[float] = 0.90,
    generated_token_ids: Optional[List[int]] = None,
    repetition_penalty: float = 1.15,
    banned_token_ids: Optional[Set[int]] = None,
    from_logits: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Sample next token index using logit-space penalties, temperature, top-k, and nucleus top-p."""
    if rng is None:
        rng = np.random.default_rng()

    # Work in float64 logit space for numerical stability
    if from_logits:
        logits = np.array(logits_or_probs, dtype=np.float64).copy()
    else:
        # Convert incoming probabilities back to logits safely
        eps = 1e-12
        p = np.array(logits_or_probs, dtype=np.float64)
        logits = np.log(np.maximum(p, eps))

    # 1. Ban specified special tokens (e.g., PAD, BOS)
    if banned_token_ids:
        for b_id in banned_token_ids:
            if 0 <= b_id < len(logits):
                logits[b_id] = -np.inf

    # 2. Apply Repetition Penalty in Logit Space
    if repetition_penalty != 1.0 and generated_token_ids:
        recent_ids = generated_token_ids[-40:]
        for t_id in set(recent_ids):
            if 0 <= t_id < len(logits):
                if logits[t_id] > 0:
                    logits[t_id] /= float(repetition_penalty)
                else:
                    logits[t_id] *= float(repetition_penalty)

    # 3. Greedy sampling if temperature is near zero
    if temperature <= 0.05:
        return int(np.argmax(logits))

    # 4. Temperature scaling
    scaled_logits = logits / float(temperature)

    # 5. Top-K filtering
    if 0 < top_k < len(scaled_logits):
        # Keep only top_k highest values; set others to -inf
        kth_val = np.partition(scaled_logits, -top_k)[-top_k]
        scaled_logits[scaled_logits < kth_val] = -np.inf

    # 6. Softmax with maximum subtraction for numerical stability
    finite_mask = np.isfinite(scaled_logits)
    if not np.any(finite_mask):
        return int(np.argmax(logits))

    max_logit = np.max(scaled_logits[finite_mask])
    exp_logits = np.zeros_like(scaled_logits)
    exp_logits[finite_mask] = np.exp(scaled_logits[finite_mask] - max_logit)
    sum_exp = np.sum(exp_logits)
    probs = exp_logits / sum_exp if sum_exp > 0 else np.ones_like(exp_logits) / len(exp_logits)

    # 7. Top-P (Nucleus) filtering on sorted descending cumulative probabilities
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)
        
        # Keep tokens up to where cumulative probability exceeds top_p
        cutoff_idx = int(np.searchsorted(cumsum, top_p, side="right"))
        cutoff_idx = max(1, min(cutoff_idx + 1, len(sorted_indices)))
        
        kept_indices = sorted_indices[:cutoff_idx]
        p_mask = np.zeros_like(probs, dtype=bool)
        p_mask[kept_indices] = True
        probs[~p_mask] = 0.0
        
        p_sum = np.sum(probs)
        if p_sum > 0:
            probs /= p_sum

    total_p = np.sum(probs)
    if total_p <= 0 or np.isnan(total_p):
        return int(np.argmax(logits))

    probs /= total_p
    return int(rng.choice(len(probs), p=probs))


def run_generation(
    model,
    tokenizer: Union[SubwordStoryTokenizer, StoryTokenizer],
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: Optional[float] = 0.90,
    repetition_penalty: float = 1.15,
    seed: Optional[int] = None,
    seq_len: int = 128,
) -> Tuple[str, int]:
    """Execute autoregressive generation loop.
    
    Returns (generated_continuation_text, runtime_ms).
    """
    start_time = time.time()
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    is_subword = isinstance(tokenizer, SubwordStoryTokenizer)

    # Cache compiled prediction step for 25x faster CPU inference
    if not hasattr(model, "_predict_step"):
        model._predict_step = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    predict_fn = model._predict_step

    if is_subword:
        # v2 Subword Generation
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not token_ids:
            token_ids = tokenizer.encode("Once upon a time", add_special_tokens=False)

        current_input = list(token_ids)
        generated_ids: List[int] = []
        banned = {tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.unk_token_id}

        for _ in range(max_new_tokens):
            # Window to last seq_len tokens with left-padding if shorter
            window = current_input[-seq_len:]
            if len(window) < seq_len:
                window = [tokenizer.pad_token_id] * (seq_len - len(window)) + window
            x = np.array([window], dtype=np.int32)
            
            # Fast graph prediction
            pred = predict_fn(x)
            pred_arr = pred.numpy()
            
            # If 3D (batch, time, vocab), take the last timestep
            if pred_arr.ndim == 3:
                step_logits = pred_arr[0, -1, :]
            else:
                step_logits = pred_arr[0]

            next_id = sample_next_token(
                logits_or_probs=step_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generated_token_ids=generated_ids,
                repetition_penalty=repetition_penalty,
                banned_token_ids=banned,
                from_logits=True,
                rng=rng,
            )

            # Check for EOS
            if next_id == tokenizer.eos_token_id:
                break

            generated_ids.append(next_id)
            current_input.append(next_id)

        runtime_ms = int((time.time() - start_time) * 1000)
        continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return continuation, runtime_ms

    else:
        # v1 Legacy Word-level Generation
        legacy_seq_len = 50
        tokens = clean_and_tokenize(prompt)
        if not tokens:
            tokens = ["once", "upon", "a", "time"]

        token_ids = tokenizer.encode(tokens)
        if len(token_ids) < legacy_seq_len:
            current_input = [tokenizer.pad_idx] * (legacy_seq_len - len(token_ids)) + token_ids
        else:
            current_input = token_ids[-legacy_seq_len:]

        generated_ids: List[int] = []
        banned = {tokenizer.pad_idx, tokenizer.start_idx}

        for _ in range(max_new_tokens):
            x = np.array([current_input[-legacy_seq_len:]], dtype=np.int32)
            pred = predict_fn(x).numpy()[0]

            next_id = sample_next_token(
                logits_or_probs=pred,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generated_token_ids=generated_ids,
                repetition_penalty=repetition_penalty,
                banned_token_ids=banned,
                from_logits=False,  # v1 output is softmax probabilities
                rng=rng,
            )

            if next_id == tokenizer.end_idx:
                break

            generated_ids.append(next_id)
            current_input.append(next_id)

        runtime_ms = int((time.time() - start_time) * 1000)
        continuation = tokenizer.decode(generated_ids, skip_special=True)
        return continuation, runtime_ms
