#!/usr/bin/env python3
"""Acquire and mix TinyStories and Project Gutenberg fairy tales for Story Continuation v2."""

import os
import sys
import json
import argparse
import random
from datetime import datetime, timezone
from datasets import load_dataset

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.src.prepare_corpus import (
    clean_text,
    extract_gutenberg_stories,
    deduplicate_documents,
    split_documents,
)

def prepare_corpus(
    tinystories_count: int = 10000,
    fairy_tales_dir: str = "data/raw",
    output_dir: str = "data/processed/v2",
    manifest_path: str = "data/manifests/dataset_v2_manifest.json",
    seed: int = 42,
    smoke_mode: bool = False
):
    if smoke_mode:
        tinystories_count = 200
        print(f"[SMOKE MODE] Restricting TinyStories count to {tinystories_count}")

    print("=" * 60)
    print("PHASE 2: PREPARING MULTI-SOURCE STORY CORPUS (v2)")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    # 1. Extract Project Gutenberg Fairy Tales
    print("1. Extracting Project Gutenberg Fairy Tales...")
    fairy_docs = []
    if os.path.exists(fairy_tales_dir):
        for fname in sorted(os.listdir(fairy_tales_dir)):
            if fname.endswith(".txt"):
                fpath = os.path.join(fairy_tales_dir, fname)
                extracted = extract_gutenberg_stories(fpath)
                fairy_docs.extend(extracted)
                print(f"   - {fname}: {len(extracted)} distinct story sections extracted")
    print(f"Total fairy tale documents extracted: {len(fairy_docs)}")

    # 2. Stream and Buffer Shuffle TinyStories
    print(f"2. Streaming {tinystories_count} stories from roneneldan/TinyStories...")
    dataset_stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    rng = random.Random(seed)
    buffer = []
    buffer_size = min(2000, tinystories_count)
    tiny_docs = []
    
    stream_iter = iter(dataset_stream)
    
    # Fill initial buffer
    for _ in range(buffer_size):
        try:
            item = next(stream_iter)
            buffer.append(item["text"])
        except StopIteration:
            break

    story_idx = 0
    while buffer and len(tiny_docs) < tinystories_count:
        # Pop random item from buffer
        pick_idx = rng.randint(0, len(buffer) - 1)
        raw_story = buffer.pop(pick_idx)
        
        # Refill buffer
        try:
            nxt = next(stream_iter)
            buffer.append(nxt["text"])
        except StopIteration:
            pass

        cleaned = clean_text(raw_story)
        words = cleaned.split()
        if len(words) >= 30:
            tiny_docs.append({
                "id": f"tinystories_{story_idx:06d}",
                "source": "roneneldan/TinyStories",
                "text": cleaned,
                "word_count": len(words)
            })
            story_idx += 1
            if story_idx % 2500 == 0:
                print(f"   - Acquired {story_idx} / {tinystories_count} TinyStories...")

    print(f"Total TinyStories documents collected: {len(tiny_docs)}")

    # 3. Deduplicate
    all_raw_docs = fairy_docs + tiny_docs
    unique_docs, duplicates_removed = deduplicate_documents(all_raw_docs)
    print(f"Deduplication complete: {len(all_raw_docs)} -> {len(unique_docs)} unique stories ({duplicates_removed} duplicates removed)")

    # 4. Document-Level Split (90/5/5)
    splits = split_documents(unique_docs, train_ratio=0.90, val_ratio=0.05, test_ratio=0.05, seed=seed)

    split_counts = {k: len(v) for k, v in splits.items()}
    split_words = {k: sum(d["word_count"] for d in v) for k, v in splits.items()}
    total_words = sum(split_words.values())
    
    fairy_words = sum(d["word_count"] for d in unique_docs if d["source"] == "Project Gutenberg")
    tiny_words = sum(d["word_count"] for d in unique_docs if d["source"] == "roneneldan/TinyStories")

    print("\nDataset Composition:")
    print(f"   - TinyStories Tokens: {tiny_words:,} ({tiny_words / total_words * 100:.1f}%)")
    print(f"   - Fairy Tale Tokens:  {fairy_words:,} ({fairy_words / total_words * 100:.1f}%)")
    print("\nSplit Distribution:")
    for s_name in ["train", "val", "test"]:
        print(f"   - {s_name.upper()}: {split_counts[s_name]} stories | {split_words[s_name]:,} words")

    # 5. Save Splits as JSONL
    for s_name, doc_list in splits.items():
        out_file = os.path.join(output_dir, f"{s_name}.jsonl")
        with open(out_file, "w", encoding="utf-8") as f:
            for doc in doc_list:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"Saved {s_name} split to: {out_file}")

    # 6. Save Reproducible Manifest
    manifest = {
        "dataset_name": "TinyStories + Grimm/Andersen Mixture v2",
        "version": "2.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "sources": [
            {
                "name": "roneneldan/TinyStories",
                "repository": "https://huggingface.co/datasets/roneneldan/TinyStories",
                "license": "CDLA-Permissive-2.0",
                "document_count": len([d for d in unique_docs if d["source"] == "roneneldan/TinyStories"]),
                "word_count": tiny_words,
            },
            {
                "name": "Project Gutenberg Classic Fairy Tales",
                "repository": "https://www.gutenberg.org",
                "license": "Public Domain",
                "document_count": len([d for d in unique_docs if d["source"] == "Project Gutenberg"]),
                "word_count": fairy_words,
            }
        ],
        "total_documents": len(unique_docs),
        "total_words": total_words,
        "mixture_ratio": {
            "tinystories_percent": round(tiny_words / total_words * 100, 2),
            "fairy_tales_percent": round(fairy_words / total_words * 100, 2)
        },
        "splits": {
            "train": {"documents": split_counts["train"], "words": split_words["train"]},
            "val": {"documents": split_counts["val"], "words": split_words["val"]},
            "test": {"documents": split_counts["test"], "words": split_words["test"]}
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest successfully written to: {manifest_path}")
    print("=" * 60)
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare v2 multi-source dataset")
    parser.add_argument("--count", type=int, default=10000, help="Number of TinyStories to sample")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test mode (200 stories)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed")
    args = parser.parse_args()

    prepare_corpus(tinystories_count=args.count, smoke_mode=args.smoke, seed=args.seed)
