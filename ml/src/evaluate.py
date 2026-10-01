#!/usr/bin/env python3
"""Evaluation suite: Perplexity, Lexical Diversity, Latency, and N-gram Baseline."""

import os
import sys
import time
import json
import argparse
import numpy as np
import tensorflow as tf
from collections import defaultdict, Counter

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.src.utils import (
    load_config,
    resolve_path,
    calculate_perplexity,
    evaluate_text_generation_quality,
)
from ml.src.tokenizer import StoryTokenizer
from ml.src.model import load_trained_model
from ml.src.generate_cli import generate

class BigramBaselineModel:
    """Standard Bigram (Markov-1) baseline language model with Laplace smoothing."""

    def __init__(self, vocab_size: int, pad_idx: int = 0, unk_idx: int = 1):
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.counts = defaultdict(Counter)
        self.context_totals = defaultdict(int)

    def fit(self, token_ids_list: list):
        for tokens in token_ids_list:
            if not tokens:
                continue
            for i in range(len(tokens) - 1):
                w_prev = tokens[i]
                w_curr = tokens[i + 1]
                self.counts[w_prev][w_curr] += 1
                self.context_totals[w_prev] += 1

    def evaluate_cross_entropy(self, X: np.ndarray, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        alpha = 0.1
        V = self.vocab_size
        total_loss = 0.0
        for i in range(len(y)):
            prev_token = int(X[i][-1])
            true_token = int(y[i])
            count = self.counts[prev_token][true_token] if prev_token in self.counts else 0
            prob = (count + alpha) / (self.context_totals[prev_token] + alpha * V)
            total_loss += -np.log(max(prob, 1e-12))
        return float(total_loss / len(y))

def run_evaluation(config_path: str = "ml/configs/base.yaml"):
    print("=" * 60)
    print("STARTING EVALUATION SUITE")
    print("=" * 60)

    cfg = load_config(config_path)
    bdir = cfg["base_dir"]

    vocab_path = resolve_path(bdir, cfg["tokenization"]["vocab_path"])
    legacy_vocab = resolve_path(bdir, cfg["tokenization"].get("legacy_vocab_path", "artifacts/tokenizer_vocab.json"))
    target_vocab = vocab_path if os.path.exists(vocab_path) else legacy_vocab

    tokenizer = StoryTokenizer.load(target_vocab)
    model = load_trained_model()

    test_x_path = resolve_path(bdir, "data/processed/test_x.npy")
    test_y_path = resolve_path(bdir, "data/processed/test_y.npy")

    lstm_loss = 0.0
    lstm_acc = 0.0
    lstm_ppl = 0.0
    if os.path.exists(test_x_path) and os.path.exists(test_y_path):
        x_test = np.load(test_x_path)
        y_test = np.load(test_y_path)
        print(f"Evaluating LSTM on {len(x_test)} test sequences...")
        res = model.evaluate(x_test, y_test, batch_size=64, verbose=0)
        lstm_loss = float(res[0])
        lstm_acc = float(res[1])
        lstm_ppl = calculate_perplexity(lstm_loss)
        print(f"LSTM Test Loss: {lstm_loss:.4f} | Perplexity: {lstm_ppl:.2f} | Accuracy: {lstm_acc*100:.2f}%")

    test_prompts = [
        "The ancient castle was silent until a shadow moved",
        "Once upon a time in a deep dark forest",
        "The young prince set out on a perilous journey",
        "A little bird landed on the windowsill and whispered",
    ]

    sample_results = []
    for p in test_prompts:
        t0 = time.time()
        cont = generate(model, tokenizer, prompt=p, max_new_tokens=40, temperature=0.8, top_k=40)
        lat = time.time() - t0
        metrics = evaluate_text_generation_quality(cont)
        sample_results.append({
            "prompt": p,
            "continuation": cont,
            "generation_time_sec": round(lat, 3),
            "metrics": metrics
        })

    eval_summary = {
        "lstm_loss": round(lstm_loss, 4),
        "lstm_accuracy": round(lstm_acc, 4),
        "lstm_perplexity": round(lstm_ppl, 2),
        "samples_evaluated": len(sample_results),
        "sample_generations": sample_results
    }

    out_json = resolve_path(bdir, "artifacts/metrics/eval_results.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"Saved evaluation results to {out_json}")
    return eval_summary

if __name__ == "__main__":
    run_evaluation()
