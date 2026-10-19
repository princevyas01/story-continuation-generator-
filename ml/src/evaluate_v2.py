"""Comprehensive comparative evaluation suite for Story Continuation Generator (v1 vs v2).

Computes:
- Perplexity (masked on non-padding tokens)
- Cross-entropy loss
- Next-token accuracy
- Distinct-1 and Distinct-2 diversity
- Repetition-4 degradation rate
- CPU inference latency (ms/token)
- Generates side-by-side human evaluation CSV and Markdown training report.
"""

import csv
import json
import os
import sys
import time
from typing import Dict, List, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import tensorflow as tf
import keras

from backend.app.generator import run_generation
from ml.src.dataset_v2 import build_sequences_from_jsonl, create_tf_dataset
from ml.src.masked_loss import (
    MaskedAccuracy,
    MaskedPerplexity,
    MaskedSparseCategoricalCrossentropy,
)
from ml.src.model import compile_model_v2
from ml.src.subword_tokenizer import SubwordStoryTokenizer
from ml.src.tokenizer import StoryTokenizer


def compute_ngram_diversity(texts: List[str]) -> Tuple[float, float, float]:
    """Compute Distinct-1, Distinct-2, and Repetition-4 across generated texts."""
    all_unigrams = []
    all_bigrams = []
    total_4grams = 0
    repeated_4grams = 0

    for text in texts:
        words = text.lower().split()
        all_unigrams.extend(words)

        for i in range(len(words) - 1):
            all_bigrams.append((words[i], words[i + 1]))

        # Within-story 4-gram repetition
        seen_4grams = set()
        for i in range(len(words) - 3):
            fg = tuple(words[i : i + 4])
            total_4grams += 1
            if fg in seen_4grams:
                repeated_4grams += 1
            else:
                seen_4grams.add(fg)

    distinct_1 = len(set(all_unigrams)) / len(all_unigrams) if all_unigrams else 0.0
    distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0
    repetition_4 = (repeated_4grams / total_4grams) if total_4grams > 0 else 0.0

    return distinct_1, distinct_2, repetition_4


def evaluate_test_split_v2(model, tokenizer: SubwordStoryTokenizer, test_jsonl: str) -> Dict[str, float]:
    """Evaluate candidate model on held-out test split."""
    x_test, y_test = build_sequences_from_jsonl(
        test_jsonl,
        tokenizer=tokenizer,
        seq_len=128,
        stride=128,
    )
    test_ds = create_tf_dataset(x_test, y_test, batch_size=64, shuffle=False)
    
    results = model.evaluate(test_ds, verbose=0)
    test_loss = float(results[0])
    test_accuracy = float(results[1]) if len(results) > 1 else 0.0
    test_perplexity = float(results[2]) if len(results) > 2 else float(np.exp(test_loss))
    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_perplexity": test_perplexity,
        "test_sequences": len(x_test),
    }


def main():
    print("=" * 60)
    print("Executing Comprehensive Comparative Evaluation Suite")
    print("=" * 60)

    # 1. Paths
    v1_model_path = os.path.join("models", "best", "story_lstm.keras")
    if not os.path.exists(v1_model_path):
        v1_model_path = os.path.join("artifacts", "models", "story_lstm.keras")

    v1_tok_path = os.path.join("artifacts", "tokenizers", "tokenizer.json")
    if not os.path.exists(v1_tok_path):
        v1_tok_path = os.path.join("artifacts", "tokenizer_vocab.json")

    v2_model_path = os.path.join("artifacts", "models", "story_lstm_v2.keras")
    if not os.path.exists(v2_model_path):
        v2_model_path = os.path.join("models", "best", "story_lstm_v2.keras")

    v2_tok_path = os.path.join("artifacts", "tokenizers", "tokenizer_v2.json")
    test_jsonl = os.path.join("data", "processed", "v2", "test.jsonl")
    suite_path = os.path.join("artifacts", "reports", "evaluation_prompts_suite.json")

    if not os.path.exists(suite_path):
        raise FileNotFoundError(f"Evaluation prompts suite not found: {suite_path}")

    with open(suite_path, "r", encoding="utf-8") as f:
        prompts_suite = json.load(f)

    # 2. Load models and tokenizers
    print(f"Loading v1 Baseline from: {v1_model_path}")
    v1_model = keras.models.load_model(v1_model_path)
    v1_tokenizer = StoryTokenizer.load(v1_tok_path)

    print(f"Loading v2 Candidate from: {v2_model_path}")
    v2_model = keras.models.load_model(v2_model_path)
    v2_tokenizer = SubwordStoryTokenizer.load(v2_tok_path)
    v2_model = compile_model_v2(v2_model, pad_id=v2_tokenizer.pad_token_id)

    # 3. Quantitative Test Evaluation for v2
    print(f"Evaluating v2 Candidate on test split ({test_jsonl})...")
    v2_test_metrics = evaluate_test_split_v2(v2_model, v2_tokenizer, test_jsonl)
    print(f"v2 Test Loss:       {v2_test_metrics['test_loss']:.4f}")
    print(f"v2 Test Accuracy:   {v2_test_metrics['test_accuracy']:.4f}")
    print(f"v2 Test Perplexity: {v2_test_metrics['test_perplexity']:.4f}")

    # Baseline recorded metrics (from existing metrics.json & project docs)
    v1_recorded_metrics = {
        "train_loss": 3.799,
        "val_loss": 5.053,
        "test_loss": 5.218,
        "test_perplexity": 184.51,
        "test_accuracy": 0.1928,
        "vocab_size": 5044,
        "architecture": "StackedLSTM-ManyToOne",
        "parameters": int(v1_model.count_params()),
    }

    # 4. Generate on Suite for both models
    print(f"\nRunning 30-Prompt Evaluation Suite across 8 categories...")
    v1_texts = []
    v2_texts = []
    v1_times = []
    v2_times = []

    eval_rows = []

    for item in prompts_suite:
        p_id = item["id"]
        cat = item["category"]
        prompt = item["prompt"]

        # Run v1 baseline
        v1_gen, v1_ms = run_generation(
            model=v1_model,
            tokenizer=v1_tokenizer,
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=40,
            top_p=0.90,
            repetition_penalty=1.15,
            seed=42 + p_id,
        )
        v1_texts.append(v1_gen)
        v1_times.append(v1_ms)

        # Run v2 candidate
        v2_gen, v2_ms = run_generation(
            model=v2_model,
            tokenizer=v2_tokenizer,
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=40,
            top_p=0.90,
            repetition_penalty=1.15,
            seed=42 + p_id,
        )
        v2_texts.append(v2_gen)
        v2_times.append(v2_ms)

        eval_rows.append({
            "prompt_id": p_id,
            "category": cat,
            "prompt": prompt,
            "v1_generation": v1_gen.replace("\n", " "),
            "v2_generation": v2_gen.replace("\n", " "),
            "v1_runtime_ms": v1_ms,
            "v2_runtime_ms": v2_ms,
        })

    # 5. Compute Diversity & Latency
    d1_v1, d2_v1, rep4_v1 = compute_ngram_diversity(v1_texts)
    d1_v2, d2_v2, rep4_v2 = compute_ngram_diversity(v2_texts)

    avg_latency_v1 = float(np.mean(v1_times))
    avg_latency_v2 = float(np.mean(v2_times))
    ms_per_tok_v1 = avg_latency_v1 / 50.0
    ms_per_tok_v2 = avg_latency_v2 / 50.0

    print("\n--- Diversity and Generation Metrics ---")
    print(f"Distinct-1:   v1={d1_v1:.4f}  vs  v2={d1_v2:.4f}")
    print(f"Distinct-2:   v1={d2_v1:.4f}  vs  v2={d2_v2:.4f}")
    print(f"Repetition-4: v1={rep4_v1:.4f}  vs  v2={rep4_v2:.4f}")
    print(f"Latency/Tok:  v1={ms_per_tok_v1:.1f}ms  vs  v2={ms_per_tok_v2:.1f}ms")

    # 6. Save Comparison JSON
    reports_dir = os.path.join("artifacts", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    comparison_json_path = os.path.join(reports_dir, "evaluation_comparison_v2.json")

    comparison_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_v1": {
            "name": "Story Continuation LSTM v1 (Baseline)",
            "tokenizer": "Word-level (5,044 words)",
            "objective": "Many-to-One (next token)",
            "parameters": int(v1_model.count_params()),
            "test_loss": v1_recorded_metrics["test_loss"],
            "test_perplexity": v1_recorded_metrics["test_perplexity"],
            "test_accuracy": v1_recorded_metrics["test_accuracy"],
            "distinct_1": float(d1_v1),
            "distinct_2": float(d2_v1),
            "repetition_4": float(rep4_v1),
            "mean_latency_ms": avg_latency_v1,
            "ms_per_token": ms_per_tok_v1,
        },
        "candidate_v2": {
            "name": "Story Continuation LSTM v2 (Candidate)",
            "tokenizer": "Byte-Level BPE (8,000 subwords)",
            "objective": "Many-to-Many autoregressive with Masked Crossentropy",
            "parameters": int(v2_model.count_params()),
            "test_loss": v2_test_metrics["test_loss"],
            "test_perplexity": v2_test_metrics["test_perplexity"],
            "test_accuracy": v2_test_metrics["test_accuracy"],
            "distinct_1": float(d1_v2),
            "distinct_2": float(d2_v2),
            "repetition_4": float(rep4_v2),
            "mean_latency_ms": avg_latency_v2,
            "ms_per_token": ms_per_tok_v2,
        },
        "improvements": {
            "perplexity_reduction_percent": float(
                (v1_recorded_metrics["test_perplexity"] - v2_test_metrics["test_perplexity"])
                / v1_recorded_metrics["test_perplexity"]
                * 100.0
            ),
            "accuracy_gain_percent": float(
                (v2_test_metrics["test_accuracy"] - v1_recorded_metrics["test_accuracy"])
                * 100.0
            ),
            "distinct_1_gain": float(d1_v2 - d1_v1),
            "distinct_2_gain": float(d2_v2 - d2_v1),
            "repetition_reduction": float(rep4_v1 - rep4_v2),
        },
    }

    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"Saved evaluation comparison JSON to {comparison_json_path}")

    # 7. Save Human Evaluation CSV
    csv_path = os.path.join(reports_dir, "human_evaluation_sheet_v2.csv")
    fieldnames = [
        "prompt_id",
        "category",
        "prompt",
        "v1_generation",
        "v2_generation",
        "v1_runtime_ms",
        "v2_runtime_ms",
        "v1_grammaticality_score_1to5",
        "v2_grammaticality_score_1to5",
        "v1_fluency_score_1to5",
        "v2_fluency_score_1to5",
        "v1_coherence_score_1to5",
        "v2_coherence_score_1to5",
        "evaluator_notes",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in eval_rows:
            row_dict = {
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "prompt": row["prompt"],
                "v1_generation": row["v1_generation"],
                "v2_generation": row["v2_generation"],
                "v1_runtime_ms": row["v1_runtime_ms"],
                "v2_runtime_ms": row["v2_runtime_ms"],
                "v1_grammaticality_score_1to5": "",
                "v2_grammaticality_score_1to5": "",
                "v1_fluency_score_1to5": "",
                "v2_fluency_score_1to5": "",
                "v1_coherence_score_1to5": "",
                "v2_coherence_score_1to5": "",
                "evaluator_notes": "",
            }
            writer.writerow(row_dict)
    print(f"Saved human evaluation sheet CSV to {csv_path}")

    # 8. Save Markdown Report
    md_path = os.path.join(reports_dir, "training_report_v2.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Story Continuation Generator v2 — Training & Evaluation Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}  \n")
        f.write(f"**Target System:** Intel Core Ultra 5 125H, 16 GB RAM, Windows 11 (Pure CPU Execution)  \n\n")

        f.write("## 1. Executive Summary\n\n")
        f.write("The Story Continuation Generator was systematically re-architected from the baseline ")
        f.write("word-level, many-to-one LSTM into a subword Byte-Level BPE Many-to-Many autoregressive model ")
        f.write("with masked cross-entropy loss and logit-space nucleus sampling.\n\n")

        f.write("### Quantitative Head-to-Head Comparison\n\n")
        f.write("| Metric | Baseline (v1) | Upgraded Candidate (v2) | Delta / Improvement |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| **Architecture** | Stacked LSTM Many-to-One | Stacked LSTM Many-to-Many | Full sequence supervision |\n")
        f.write(f"| **Tokenizer** | Word-level (5,044 vocab) | Byte-Level BPE (8,000 vocab) | Zero OOV / Subword coverage |\n")
        f.write(f"| **Parameters** | {v1_model.count_params():,} | {v2_model.count_params():,} | Scaled capacity |\n")
        f.write(f"| **Test Loss** | {v1_recorded_metrics['test_loss']:.3f} | {v2_test_metrics['test_loss']:.3f} | Lower is better |\n")
        f.write(f"| **Test Perplexity** | {v1_recorded_metrics['test_perplexity']:.2f} | {v2_test_metrics['test_perplexity']:.2f} | **{comparison_summary['improvements']['perplexity_reduction_percent']:.1f}% reduction** |\n")
        f.write(f"| **Next-Token Accuracy** | {v1_recorded_metrics['test_accuracy']*100:.2f}% | {v2_test_metrics['test_accuracy']*100:.2f}% | **+{comparison_summary['improvements']['accuracy_gain_percent']:.2f}% gain** |\n")
        f.write(f"| **Distinct-1 (Unigrams)** | {d1_v1:.4f} | {d1_v2:.4f} | +{d1_v2 - d1_v1:.4f} diversity |\n")
        f.write(f"| **Distinct-2 (Bigrams)** | {d2_v1:.4f} | {d2_v2:.4f} | +{d2_v2 - d2_v1:.4f} diversity |\n")
        f.write(f"| **Repetition-4 Rate** | {rep4_v1:.4f} | {rep4_v2:.4f} | -{rep4_v1 - rep4_v2:.4f} repetition |\n")
        f.write(f"| **CPU Latency per Token** | {ms_per_tok_v1:.1f} ms | {ms_per_tok_v2:.1f} ms | Interactive real-time |\n\n")

        f.write("## 2. Sample Generation Comparisons\n\n")
        for i in range(min(5, len(eval_rows))):
            item = eval_rows[i]
            f.write(f"### Prompt {item['prompt_id']}: {item['category']}\n")
            f.write(f"> **Seed Prompt:** \"*{item['prompt']}*\"\n\n")
            f.write(f"- **v1 Baseline:** {item['v1_generation']}\n")
            f.write(f"- **v2 Candidate:** {item['v2_generation']}\n\n")

    print(f"Saved Markdown report to {md_path}")
    print("\nEvaluation successfully completed!")


if __name__ == "__main__":
    main()
