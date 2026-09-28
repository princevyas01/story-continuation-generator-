#!/usr/bin/env python3
"""Evaluation suite: Loss, Perplexity, Lexical Diversity, Latency, and N-gram Baseline."""

import os
import sys
import time
import json
import csv
import datetime
import argparse
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple
import numpy as np
import tensorflow as tf

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config, resolve_path
from src.tokenizer import StoryTokenizer
from src.sequences import build_dataset_arrays
from src.model import load_trained_model
from src.generate import generate_continuation
from src.metrics import calculate_perplexity, compute_distinct_n, compute_repetition_ratio, evaluate_text_generation_quality

class BigramBaselineModel:
    """Standard Bigram (Markov-1) baseline language model with Laplace smoothing."""

    def __init__(self, vocab_size: int, pad_idx: int = 0, unk_idx: int = 1):
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.counts = defaultdict(Counter)
        self.context_totals = defaultdict(int)
        self.unigram_counts = Counter()
        self.total_tokens = 0

    def fit(self, token_ids_list: List[List[int]]):
        """Estimate transition probabilities from training sequences."""
        for tokens in token_ids_list:
            if not tokens:
                continue
            for i in range(len(tokens) - 1):
                w_prev = tokens[i]
                w_curr = tokens[i + 1]
                self.counts[w_prev][w_curr] += 1
                self.context_totals[w_prev] += 1
                self.unigram_counts[w_curr] += 1
                self.total_tokens += 1

    def evaluate_cross_entropy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute average cross entropy on (X, y) test pairs."""
        if len(y) == 0:
            return 0.0

        total_loss = 0.0
        # Laplace smoothing alpha
        alpha = 0.1
        V = self.vocab_size

        for i in range(len(y)):
            prev_token = int(X[i][-1])
            true_token = int(y[i])

            prev_counter = self.counts.get(prev_token, None)
            count = prev_counter[true_token] if prev_counter else 0
            numerator = count + alpha
            denominator = self.context_totals[prev_token] + alpha * V
            prob = numerator / denominator
            total_loss += -np.log(max(prob, 1e-12))

        return float(total_loss / len(y))

def run_evaluation(config_path: str = "configs/config.yaml"):
    print("=" * 60)
    print("RUNNING COMPREHENSIVE EVALUATION SUITE")
    print("=" * 60)

    cfg = load_config(config_path)
    bdir = cfg["base_dir"]

    manifest_path = resolve_path(bdir, cfg["data"]["manifest_path"])
    vocab_path = resolve_path(bdir, cfg["tokenization"]["vocab_path"])
    best_model_path = resolve_path(bdir, cfg["training"]["best_model_path"])
    metrics_json_path = resolve_path(bdir, cfg["artifacts"]["metrics_json"])
    samples_md_path = resolve_path(bdir, cfg["artifacts"]["sample_generations_md"])
    samples_json_path = resolve_path(bdir, cfg["artifacts"]["sample_generations_json"])
    human_sheet_path = resolve_path(bdir, cfg["artifacts"]["human_eval_sheet_csv"])

    # Load manifest and tokenizer
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    tokenizer = StoryTokenizer.load(vocab_path)
    model = load_trained_model(best_model_path)
    print(f"Loaded best checkpoint: {best_model_path}")

    # Load test split tokens
    processed_dir = resolve_path(bdir, cfg["data"]["processed_dir"])
    test_ids = manifest["story_split_mapping"]["test"]
    train_ids = manifest["story_split_mapping"]["train"]

    def load_tokens(ids):
        res = []
        for d_id in ids:
            p = os.path.join(processed_dir, f"{d_id}.txt")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    res.append(f.read().split())
        return res

    test_tokens_docs = load_tokens(test_ids)
    train_tokens_docs = load_tokens(train_ids)

    seq_len = cfg["tokenization"]["sequence_length"]
    X_test, y_test = build_dataset_arrays(test_tokens_docs, tokenizer, seq_len=seq_len, step=2)

    # 1. Model Loss & Perplexity on Held-Out Test Set
    print(f"Evaluating LSTM on {len(X_test)} test sequences...")
    eval_res = model.evaluate(X_test, y_test, batch_size=cfg["training"]["batch_size"], verbose=0)
    test_loss = float(eval_res[0])
    test_acc = float(eval_res[1])
    test_ppl = calculate_perplexity(test_loss)
    print(f"LSTM Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} | Perplexity: {test_ppl:.2f}")

    # 2. Bigram Baseline Comparison
    print("Training Bigram Baseline...")
    bigram = BigramBaselineModel(vocab_size=tokenizer.vocab_size)
    train_token_ids = [tokenizer.encode(tokens) for tokens in train_tokens_docs]
    bigram.fit(train_token_ids)
    bigram_loss = bigram.evaluate_cross_entropy(X_test, y_test)
    bigram_ppl = calculate_perplexity(bigram_loss)
    print(f"Bigram Baseline Test Loss: {bigram_loss:.4f} | Perplexity: {bigram_ppl:.2f}")

    # 3. Parameter Grid Generation Evaluation
    test_prompts = [
        "once upon a time in a dark and ancient forest",
        "the brave little prince drew his golden sword",
        "deep inside the hidden cave, the old wizard whispered",
        "the poor woodcutter looked into the crystal river and saw"
    ]

    temperatures = [0.5, 0.8, 1.0, 1.2]
    top_k_values = [0, 20, 50]
    sweep_results: List[Dict[str, Any]] = []

    print("\nExecuting sampling grid across temperatures and top-k values...")
    for prompt in test_prompts:
        for temp in temperatures:
            for k in top_k_values:
                gen_res = generate_continuation(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    seq_len=seq_len,
                    max_new_tokens=40,
                    temperature=temp,
                    top_k=k,
                    top_p=0.90,
                    seed=42
                )
                lexical = evaluate_text_generation_quality(gen_res["continuation"])
                sweep_results.append({
                    "prompt": prompt,
                    "temperature": temp,
                    "top_k": k,
                    "top_p": 0.90,
                    "continuation": gen_res["continuation"],
                    "full_story": gen_res["full_story"],
                    "latency_ms_per_token": gen_res["latency_ms_per_token"],
                    "metrics": lexical
                })

    # Save sample generations JSON
    os.makedirs(os.path.dirname(samples_json_path), exist_ok=True)
    with open(samples_json_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    # Save sample generations Markdown
    os.makedirs(os.path.dirname(samples_md_path), exist_ok=True)
    with open(samples_md_path, "w", encoding="utf-8") as f:
        f.write("# Qualitative Generation Analysis & Sampling Behavior\n\n")
        f.write("This artifact records deterministic continuation outputs across temperature and top-k combinations.\n\n")
        for res in sweep_results[:12]:
            f.write(f"### Prompt: *\"{res['prompt']}\"*\n")
            f.write(f"- **Sampling Params:** Temperature={res['temperature']}, Top-k={res['top_k']}, Top-p={res['top_p']}\n")
            f.write(f"- **Continuation:** {res['continuation']}\n")
            f.write(f"- **Distinct-1:** {res['metrics']['distinct_1']} | **Distinct-2:** {res['metrics']['distinct_2']} | **3-gram Repetition:** {res['metrics']['repetition_ratio_3gram']}\n")
            f.write(f"- **Inference Latency:** {res['latency_ms_per_token']:.2f} ms/token\n\n")

    # 4. Generate Unfilled Human Evaluation Worksheet
    os.makedirs(os.path.dirname(human_sheet_path), exist_ok=True)
    with open(human_sheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "prompt",
            "temperature",
            "top_k",
            "generated_continuation",
            "coherence_1_5",
            "grammaticality_1_5",
            "relevance_1_5",
            "repetition_1_5",
            "reviewer_notes"
        ])
        for idx, res in enumerate(sweep_results[:20], 1):
            writer.writerow([
                f"S{idx:02d}",
                res["prompt"],
                res["temperature"],
                res["top_k"],
                res["continuation"],
                "", # Left empty for manual review
                "",
                "",
                "",
                ""
            ])

    print(f"Generated human evaluation template with 20 samples at: {human_sheet_path}")

    # 5. Average Generation Statistics
    avg_d1 = np.mean([r["metrics"]["distinct_1"] for r in sweep_results])
    avg_d2 = np.mean([r["metrics"]["distinct_2"] for r in sweep_results])
    avg_rep = np.mean([r["metrics"]["repetition_ratio_3gram"] for r in sweep_results])
    avg_lat = np.mean([r["latency_ms_per_token"] for r in sweep_results])

    eval_summary = {
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
        "lstm_test_loss": round(test_loss, 4),
        "lstm_test_accuracy": round(test_acc, 4),
        "lstm_test_perplexity": round(test_ppl, 2),
        "bigram_baseline_loss": round(bigram_loss, 4),
        "bigram_baseline_perplexity": round(bigram_ppl, 2),
        "mean_distinct_1": round(float(avg_d1), 4),
        "mean_distinct_2": round(float(avg_d2), 4),
        "mean_repetition_ratio_3gram": round(float(avg_rep), 4),
        "mean_latency_ms_per_token": round(float(avg_lat), 2),
        "num_sweep_samples": len(sweep_results)
    }

    # Update metrics.json
    if os.path.exists(metrics_json_path):
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(eval_summary)
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

    print("\nSummary Evaluation Metrics:")
    for k, v in eval_summary.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    return eval_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on trained LSTM model.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    run_evaluation(config_path=args.config)
