#!/usr/bin/env python3
"""Interactive and CLI generation tool for trained Story LSTM (v1 & v2)."""

import argparse
import os
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.app.generator import run_generation, sample_next_token
from ml.src.model import load_trained_model
from ml.src.subword_tokenizer import SubwordStoryTokenizer
from ml.src.tokenizer import StoryTokenizer


def main():
    parser = argparse.ArgumentParser(description="CLI generator for Story LSTM")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time, Lily found a magical key hidden in the garden",
    )
    parser.add_argument("--tokens", type=int, default=60, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.75, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K cutoff")
    parser.add_argument("--top-p", type=float, default=0.90, help="Nucleus Top-P")
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="Repetition penalty")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-path", type=str, default=None, help="Explicit model path")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Explicit tokenizer path")
    args = parser.parse_args()

    # Determine tokenizer and model
    v2_tok_path = os.path.join("artifacts", "tokenizers", "tokenizer_v2.json")
    v1_tok_path = os.path.join("artifacts", "tokenizers", "tokenizer.json")

    tok_path = args.tokenizer_path
    if tok_path is None:
        tok_path = v2_tok_path if os.path.exists(v2_tok_path) else v1_tok_path

    print(f"Loading tokenizer from: {tok_path}")
    if "v2" in tok_path or "subword" in tok_path:
        tokenizer = SubwordStoryTokenizer.load(tok_path)
    else:
        tokenizer = StoryTokenizer.load(tok_path)

    print(f"Loading model (explicit={args.model_path})...")
    model = load_trained_model(args.model_path)
    print(f"Model loaded: {model.name} ({int(model.count_params()):,} parameters)")

    print("\n" + "=" * 60)
    print(f"PROMPT: {args.prompt}")
    print("=" * 60)

    continuation, ms = run_generation(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )

    print(f"CONTINUATION:\n{continuation}")
    print("=" * 60)
    print(f"Generation runtime: {ms} ms\n")


if __name__ == "__main__":
    main()
