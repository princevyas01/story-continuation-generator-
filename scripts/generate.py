#!/usr/bin/env python3
"""Interactive Terminal / Command-Line Story Continuation Generator.

Runs directly in your terminal without requiring Streamlit or a browser.
"""

import os
import sys
import argparse
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config, resolve_path
from src.tokenizer import StoryTokenizer
from src.model import load_trained_model
from src.generate import generate_continuation
from src.metrics import evaluate_text_generation_quality

def print_banner():
    print("=" * 65)
    print("  STORY CONTINUATION GENERATOR (LSTM CLI)")
    print("  MDM Foundation of Generative AI - Word-Level Neural LM")
    print("=" * 65)

def run_cli_generator():
    parser = argparse.ArgumentParser(description="Generate story continuations using local LSTM.")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Initial story prefix")
    parser.add_argument("--length", "-l", type=int, default=50, help="Number of new tokens to generate")
    parser.add_argument("--temperature", "-t", type=float, default=0.8, help="Sampling temperature (0.1 to 1.5)")
    parser.add_argument("--top-k", "-k", type=int, default=40, help="Top-k sampling threshold (0 to disable)")
    parser.add_argument("--top-p", type=float, default=0.90, help="Top-p nucleus threshold (0.1 to 1.0)")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed for reproducible outputs")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    print_banner()

    # Load configuration
    cfg = load_config(args.config)
    bdir = cfg["base_dir"]

    model_path = resolve_path(bdir, cfg["training"]["best_model_path"])
    vocab_path = resolve_path(bdir, cfg["tokenization"]["vocab_path"])

    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model checkpoint not found at: {model_path}")
        print("Please train the model first by running: python src/train.py")
        sys.exit(1)

    if not os.path.exists(vocab_path):
        print(f"[ERROR] Vocabulary file not found at: {vocab_path}")
        print("Please run: python scripts/prepare_data.py")
        sys.exit(1)

    print(f"Loading vocabulary from: {os.path.basename(vocab_path)}...")
    tokenizer = StoryTokenizer.load(vocab_path)
    print(f"Loading best LSTM weights from: {os.path.basename(model_path)}...")
    model = load_trained_model(model_path)
    print(f"Model ready! Vocabulary: {tokenizer.vocab_size} words | Parameters: {model.count_params():,}")
    print("-" * 65)

    seq_len = cfg["tokenization"]["sequence_length"]

    # Single-prompt CLI execution
    if args.prompt:
        print(f"\n[PROMPT]: {args.prompt}")
        print(f"[CONFIG]: Length={args.length}, Temp={args.temperature}, Top-k={args.top_k}, Top-p={args.top_p}, Seed={args.seed}")
        print("\nGenerating...")
        res = generate_continuation(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            seq_len=seq_len,
            max_new_tokens=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=cfg["generation"].get("repetition_penalty", 1.15),
            seed=args.seed
        )
        print("\n" + "=" * 65)
        print("CONTINUATION:")
        print("=" * 65)
        print(res["full_story"])
        print("-" * 65)
        q = evaluate_text_generation_quality(res["continuation"])
        print(f"Generated {res['tokens_generated']} tokens in {res['elapsed_seconds']}s ({res['latency_ms_per_token']} ms/token)")
        print(f"Distinct-1: {q['distinct_1']} | Distinct-2: {q['distinct_2']} | 3-gram Repetition: {q['repetition_ratio_3gram']}")
        print("=" * 65)
        return

    # Interactive REPL Loop
    print("Entering interactive mode. Type your story prompt and press Enter.")
    print("Commands: 'quit' or 'exit' to exit, 'sample' for a preset prompt.\n")

    presets = [
        "once upon a time in a dark and ancient forest",
        "the brave little prince drew his sword and entered the castle",
        "there was once a poor woodcutter who lived with his three daughters",
        "deep in the enchanted mountain the old wizard found a magical stone"
    ]
    preset_idx = 0

    while True:
        try:
            prompt_input = input("\nEnter Story Prompt > ").strip()
            if not prompt_input:
                continue
            if prompt_input.lower() in ["quit", "exit", "q"]:
                print("Exiting generator. Goodbye!")
                break
            if prompt_input.lower() == "sample":
                prompt_input = presets[preset_idx % len(presets)]
                preset_idx += 1
                print(f"Using preset prompt: \"{prompt_input}\"")

            print("\nGenerating continuation...")
            res = generate_continuation(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_input,
                seq_len=seq_len,
                max_new_tokens=args.length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=cfg["generation"].get("repetition_penalty", 1.15),
                seed=args.seed
            )

            print("-" * 65)
            print(f">> FULL STORY:\n{res['full_story']}")
            print("-" * 65)
            q = evaluate_text_generation_quality(res["continuation"])
            print(f"[{res['tokens_generated']} tokens | {res['latency_ms_per_token']} ms/token | Distinct-1: {q['distinct_1']} | Distinct-2: {q['distinct_2']}]")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

if __name__ == "__main__":
    run_cli_generator()
