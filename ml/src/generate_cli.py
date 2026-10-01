#!/usr/bin/env python3
"""Interactive and CLI generation tool for trained Story LSTM."""

import os
import sys
import argparse
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.src.tokenizer import StoryTokenizer, clean_and_tokenize
from ml.src.model import load_trained_model
from ml.src.utils import load_config, resolve_path

def sample_next_token(
    probabilities: np.ndarray,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.90,
    generated_token_ids: list = None,
    repetition_penalty: float = 1.15,
    rng: np.random.Generator = None
) -> int:
    """Sample next token with temperature scaling, repetition penalty, top-k, and top-p."""
    if rng is None:
        rng = np.random.default_rng()

    probs = np.array(probabilities, dtype=np.float64)

    # 1. Repetition penalty
    if repetition_penalty > 1.0 and generated_token_ids:
        recent = generated_token_ids[-30:]
        for t_id in set(recent):
            if 0 <= t_id < len(probs):
                probs[t_id] /= repetition_penalty

    # 2. Temperature
    if temperature <= 0.05:
        return int(np.argmax(probs))

    eps = 1e-12
    logits = np.log(np.maximum(probs, eps)) / float(temperature)
    logits = logits - np.max(logits)
    scaled_probs = np.exp(logits)
    s = np.sum(scaled_probs)
    scaled_probs = scaled_probs / s if s > 0 else np.ones_like(scaled_probs) / len(scaled_probs)

    # 3. Top-k
    if top_k > 0 and top_k < len(scaled_probs):
        top_indices = np.argsort(scaled_probs)[-top_k:]
        mask = np.zeros_like(scaled_probs, dtype=bool)
        mask[top_indices] = True
        scaled_probs[~mask] = 0.0
        sk = np.sum(scaled_probs)
        if sk > 0:
            scaled_probs /= sk

    # 4. Top-p
    if 0.0 < top_p < 1.0:
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

def generate(
    model,
    tokenizer: StoryTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.90,
    repetition_penalty: float = 1.15,
    seed: int = None,
    seq_len: int = 50
) -> str:
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    tokens = clean_and_tokenize(prompt)
    if not tokens:
        tokens = ["once", "upon", "a", "time"]

    token_ids = tokenizer.encode(tokens)
    if len(token_ids) < seq_len:
        current_input = [tokenizer.pad_idx] * (seq_len - len(token_ids)) + token_ids
    else:
        current_input = token_ids[-seq_len:]

    generated_ids = []
    for _ in range(max_new_tokens):
        x = np.array([current_input[-seq_len:]], dtype=np.int32)
        pred = model(x, training=False).numpy()[0]

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
            repetition_penalty=repetition_penalty,
            rng=rng
        )

        if next_id == tokenizer.end_idx:
            break

        generated_ids.append(next_id)
        current_input.append(next_id)

    continuation = tokenizer.decode(generated_ids, skip_special=True)
    return continuation

def main():
    parser = argparse.ArgumentParser(description="CLI generator for Story LSTM")
    parser.add_argument("--prompt", type=str, default="Once upon a time in a deep dark forest")
    parser.add_argument("--tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    bdir = cfg["base_dir"]
    vocab_path = resolve_path(bdir, cfg["tokenization"]["vocab_path"])
    legacy_vocab = resolve_path(bdir, cfg["tokenization"].get("legacy_vocab_path", "artifacts/tokenizer_vocab.json"))
    target_vocab = vocab_path if os.path.exists(vocab_path) else legacy_vocab

    print("Loading model and tokenizer...")
    tokenizer = StoryTokenizer.load(target_vocab)
    model = load_trained_model()

    print(f"\nPrompt: {args.prompt}")
    continuation = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed
    )
    print(f"\nContinuation: {continuation}\n")

if __name__ == "__main__":
    main()
