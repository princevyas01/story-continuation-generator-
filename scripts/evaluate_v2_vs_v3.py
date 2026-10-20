#!/usr/bin/env python3
"""Run reproducible 30-prompt side-by-side evaluation comparing Story LSTM v2 and v3.

Produces:
- artifacts/reports/v2_vs_v3_generation_comparison.json
- artifacts/reports/v2_vs_v3_generation_comparison.md
- artifacts/reports/human_evaluation_sheet_v3.csv
"""

import csv
import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.config import settings
from backend.app.model_service import model_service


def compute_diversity_metrics(text: str) -> Dict[str, float]:
    """Calculate distinct-1, distinct-2, and 4-gram repetition rate on whitespace tokens."""
    tokens = [w.lower().strip(".,!?;:\"'()[]") for w in text.split() if w.strip()]
    n = len(tokens)
    if n == 0:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "repetition_4": 0.0, "word_count": 0}

    unigrams = set(tokens)
    distinct_1 = len(unigrams) / n

    if n > 1:
        bigrams = [tuple(tokens[i : i + 2]) for i in range(n - 1)]
        distinct_2 = len(set(bigrams)) / len(bigrams)
    else:
        distinct_2 = 1.0

    if n >= 4:
        fourgrams = [tuple(tokens[i : i + 4]) for i in range(n - 3)]
        total_4g = len(fourgrams)
        unique_4g = len(set(fourgrams))
        rep_4 = 1.0 - (unique_4g / total_4g)
    else:
        rep_4 = 0.0

    return {
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "repetition_4": round(rep_4, 4),
        "word_count": n,
    }


def automated_quality_checks(prompt: str, text: str) -> Dict[str, Any]:
    """Automated linguistic heuristic diagnostics."""
    stripped = text.strip()
    words = [w.lower().strip(".,!?;:\"'()[]") for w in text.split() if w.strip()]
    prompt_words = set(w.lower().strip(".,!?;:\"'()[]") for w in prompt.split() if len(w) > 3)

    # Relevance: check if continuation reuses prompt vocabulary or stays related
    overlap = len(set(words) & prompt_words) if words else 0
    relevance_score = min(1.0, overlap / max(1, len(prompt_words)))

    # Grammar / formatting checks
    ends_with_terminal = bool(stripped and stripped[-1] in ".!?\"'")
    has_repeated_consecutive = any(
        len(words) > i + 1 and words[i] == words[i + 1] and len(words[i]) > 3
        for i in range(len(words) - 1)
    )

    return {
        "ends_with_punctuation": ends_with_terminal,
        "prompt_word_overlap_count": overlap,
        "consecutive_duplicate_word": has_repeated_consecutive,
        "char_length": len(stripped),
    }


def main():
    print("=" * 70)
    print("RUNNING REPRODUCIBLE 30-PROMPT EVALUATION: LSTM v2 vs v3")
    print("=" * 70)

    # Initialize model service with both models loaded
    model_service.load_artifacts(preferred_version="v2")
    assert model_service.is_loaded, "ModelService failed to initialize"

    prompts_path = os.path.join("artifacts", "reports", "evaluation_prompts_suite.json")
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Evaluation prompt suite not found: {prompts_path}")

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_suite = json.load(f)

    results: List[Dict[str, Any]] = []
    csv_rows: List[List[Any]] = []

    # Fixed generation hyperparameter setup
    gen_config = {
        "max_new_tokens": 50,
        "temperature": 0.75,
        "top_k": 40,
        "top_p": 0.90,
        "repetition_penalty": 1.15,
        "seed": 42,
    }

    print(f"Loaded {len(prompts_suite)} evaluation prompts.")
    print(f"Settings: {gen_config}\n")

    for idx, item in enumerate(prompts_suite, start=1):
        prompt_text = item["prompt"]
        category = item.get("category", "General")
        prompt_id = item.get("id", idx)

        print(f"[{idx:02d}/30] ID {prompt_id} ({category}): \"{prompt_text[:45]}...\"")

        # Generate with v2
        t0 = time.time()
        v2_text, v2_ms = model_service.generate(
            prompt=prompt_text,
            max_new_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            top_k=gen_config["top_k"],
            top_p=gen_config["top_p"],
            repetition_penalty=gen_config["repetition_penalty"],
            seed=gen_config["seed"],
            model_version="v2",
        )
        v2_metrics = compute_diversity_metrics(v2_text)
        v2_heuristics = automated_quality_checks(prompt_text, v2_text)

        # Generate with v3
        v3_text, v3_ms = model_service.generate(
            prompt=prompt_text,
            max_new_tokens=gen_config["max_new_tokens"],
            temperature=gen_config["temperature"],
            top_k=gen_config["top_k"],
            top_p=gen_config["top_p"],
            repetition_penalty=gen_config["repetition_penalty"],
            seed=gen_config["seed"],
            model_version="v3",
        )
        v3_metrics = compute_diversity_metrics(v3_text)
        v3_heuristics = automated_quality_checks(prompt_text, v3_text)

        record = {
            "id": prompt_id,
            "category": category,
            "prompt": prompt_text,
            "settings": gen_config,
            "v2": {
                "generated_text": v2_text,
                "runtime_ms": v2_ms,
                "diversity": v2_metrics,
                "heuristics": v2_heuristics,
            },
            "v3": {
                "generated_text": v3_text,
                "runtime_ms": v3_ms,
                "diversity": v3_metrics,
                "heuristics": v3_heuristics,
            },
        }
        results.append(record)

        # Add to CSV evaluation sheet (blank human score columns 1-5)
        csv_rows.append([
            prompt_id,
            category,
            prompt_text,
            "v2",
            v2_text,
            "",  # Grammar (1-5)
            "",  # Prompt Relevance (1-5)
            "",  # Entity Consistency (1-5)
            "",  # Local Coherence (1-5)
            "",  # Causal Consistency (1-5)
            "",  # Repetition Free (1-5)
            "",  # Overall Readability (1-5)
            "",  # Evaluator Notes
        ])
        csv_rows.append([
            prompt_id,
            category,
            prompt_text,
            "v3",
            v3_text,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ])

    # Summary statistics
    v2_latencies = [r["v2"]["runtime_ms"] for r in results]
    v3_latencies = [r["v3"]["runtime_ms"] for r in results]
    v2_d1 = [r["v2"]["diversity"]["distinct_1"] for r in results]
    v3_d1 = [r["v3"]["diversity"]["distinct_1"] for r in results]
    v2_d2 = [r["v2"]["diversity"]["distinct_2"] for r in results]
    v3_d2 = [r["v3"]["diversity"]["distinct_2"] for r in results]
    v2_rep = [r["v2"]["diversity"]["repetition_4"] for r in results]
    v3_rep = [r["v3"]["diversity"]["repetition_4"] for r in results]

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompts_evaluated": len(results),
        "settings": gen_config,
        "v2_baseline_aggregates": {
            "mean_latency_ms": round(float(np.mean(v2_latencies)), 2),
            "mean_distinct_1": round(float(np.mean(v2_d1)), 4),
            "mean_distinct_2": round(float(np.mean(v2_d2)), 4),
            "mean_repetition_4": round(float(np.mean(v2_rep)), 4),
        },
        "v3_candidate_aggregates": {
            "mean_latency_ms": round(float(np.mean(v3_latencies)), 2),
            "mean_distinct_1": round(float(np.mean(v3_d1)), 4),
            "mean_distinct_2": round(float(np.mean(v3_d2)), 4),
            "mean_repetition_4": round(float(np.mean(v3_rep)), 4),
        },
    }

    full_output = {
        "summary": summary,
        "samples": results,
    }

    # Save JSON comparison artifact
    out_json = os.path.join("artifacts", "reports", "v2_vs_v3_generation_comparison.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] Saved JSON results to: {out_json}")

    # Save CSV human evaluation sheet
    out_csv = os.path.join("artifacts", "reports", "human_evaluation_sheet_v3.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt_id",
            "category",
            "prompt",
            "model_version",
            "generated_text",
            "grammar_1_to_5",
            "relevance_1_to_5",
            "entity_consistency_1_to_5",
            "local_coherence_1_to_5",
            "causal_consistency_1_to_5",
            "repetition_free_1_to_5",
            "overall_readability_1_to_5",
            "evaluator_notes",
        ])
        writer.writerows(csv_rows)
    print(f"[DONE] Saved blank human evaluation sheet to: {out_csv}")

    # Generate Markdown comparison report
    out_md = os.path.join("artifacts", "reports", "v2_vs_v3_generation_comparison.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Story LSTM v2 vs v3 Side-by-Side Generation Comparison\n\n")
        f.write("## 1. Evaluation Methodology\n")
        f.write("- **Prompts**: 30 fixed prompts from `artifacts/reports/evaluation_prompts_suite.json` across 8 narrative categories.\n")
        f.write(f"- **Sampling Configuration**: Temperature={gen_config['temperature']}, Top-K={gen_config['top_k']}, Top-P={gen_config['top_p']}, Repetition Penalty={gen_config['repetition_penalty']}, Seed={gen_config['seed']}.\n")
        f.write("- **Constraint & Sanitization**: PAD, BOS, and UNK tokens banned; generation halts on EOS.\n\n")
        
        f.write("## 2. Aggregate Comparative Metrics\n\n")
        f.write("| Metric | Story LSTM v2 (Baseline) | Story LSTM v3 (Colab Candidate) | Difference / Notes |\n")
        f.write("| :--- | :---: | :---: | :--- |\n")
        f.write(f"| **Mean Latency** | {summary['v2_baseline_aggregates']['mean_latency_ms']} ms | {summary['v3_candidate_aggregates']['mean_latency_ms']} ms | {summary['v3_candidate_aggregates']['mean_latency_ms'] - summary['v2_baseline_aggregates']['mean_latency_ms']:+.1f} ms |\n")
        f.write(f"| **Distinct-1 (Unigram)** | {summary['v2_baseline_aggregates']['mean_distinct_1']} | {summary['v3_candidate_aggregates']['mean_distinct_1']} | {summary['v3_candidate_aggregates']['mean_distinct_1'] - summary['v2_baseline_aggregates']['mean_distinct_1']:+.4f} |\n")
        f.write(f"| **Distinct-2 (Bigram)** | {summary['v2_baseline_aggregates']['mean_distinct_2']} | {summary['v3_candidate_aggregates']['mean_distinct_2']} | {summary['v3_candidate_aggregates']['mean_distinct_2'] - summary['v2_baseline_aggregates']['mean_distinct_2']:+.4f} |\n")
        f.write(f"| **Repetition-4 Rate** | {summary['v2_baseline_aggregates']['mean_repetition_4']} | {summary['v3_candidate_aggregates']['mean_repetition_4']} | {summary['v3_candidate_aggregates']['mean_repetition_4'] - summary['v2_baseline_aggregates']['mean_repetition_4']:+.4f} |\n\n")

        f.write("## 3. Detailed Side-by-Side Story Generations\n\n")
        for r in results:
            f.write(f"### Prompt {r['id']}: [{r['category']}]\n")
            f.write(f"> **Prompt**: *\"{r['prompt']}\"*\n\n")
            f.write(f"- **v2 Output** ({r['v2']['runtime_ms']} ms):\n")
            f.write(f"  > {r['v2']['generated_text']}\n\n")
            f.write(f"- **v3 Output** ({r['v3']['runtime_ms']} ms):\n")
            f.write(f"  > {r['v3']['generated_text']}\n\n")
            f.write("---\n\n")

    print(f"[DONE] Saved Markdown comparison report to: {out_md}")
    print("\nEvaluation successfully completed.")


if __name__ == "__main__":
    main()
