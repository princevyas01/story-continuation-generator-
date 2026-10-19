"""Data acquisition, cleaning, deduplication, and splitting for Story Continuation v2."""

import os
import re
import json
import hashlib
import random
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone

def clean_text(text: str) -> str:
    """Normalize whitespace and line endings while strictly preserving casing and punctuation."""
    if not text:
        return ""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Normalize unicode quotation marks & dashes
    text = text.replace("“", "\"").replace("”", "\"")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", " - ").replace("–", " - ")
    text = text.replace("`", "'")
    # Collapse multiple blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    # Collapse intra-line multiple whitespace
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines).strip()
    return cleaned

def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer metadata blocks."""
    start_match = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*", text, re.IGNORECASE)
    if start_match:
        text = text[start_match.end():]

    end_match = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*", text, re.IGNORECASE)
    if end_match:
        text = text[:end_match.start()]
    else:
        end_alt = re.search(r"End of (the )?Project Gutenberg", text, re.IGNORECASE)
        if end_alt:
            text = text[:end_alt.start()]

    return text.strip()

def extract_gutenberg_stories(raw_filepath: str) -> List[Dict[str, Any]]:
    """Extract individual stories from a Project Gutenberg collection book."""
    if not os.path.exists(raw_filepath):
        return []

    with open(raw_filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    body = strip_gutenberg_boilerplate(content)
    # Fairy tales usually have uppercase chapter titles like "THE FROG-PRINCE" or Roman numerals
    story_blocks = re.split(r"\n{3,}(?=[A-Z\s]{4,}\n)", body)
    extracted = []
    base_name = os.path.basename(raw_filepath).replace(".txt", "")

    for idx, block in enumerate(story_blocks):
        cleaned = clean_text(block)
        words = cleaned.split()
        if len(words) >= 40:
            extracted.append({
                "id": f"{base_name}_story_{idx:03d}",
                "source": "Project Gutenberg",
                "text": cleaned,
                "word_count": len(words)
            })

    if not extracted and len(body.split()) >= 40:
        cleaned = clean_text(body)
        extracted.append({
            "id": f"{base_name}_full",
            "source": "Project Gutenberg",
            "text": cleaned,
            "word_count": len(cleaned.split())
        })

    return extracted

def deduplicate_documents(documents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Exact deduplication using SHA-256 hash of normalized text."""
    seen_hashes = set()
    unique_docs = []
    duplicates_removed = 0

    for doc in documents:
        norm_key = re.sub(r"\s+", " ", doc["text"].lower().strip())
        h = hashlib.sha256(norm_key.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            duplicates_removed += 1
            continue
        seen_hashes.add(h)
        unique_docs.append(doc)

    return unique_docs, duplicates_removed

def split_documents(
    documents: List[Dict[str, Any]],
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """Document-level split ensuring no cross-split leakage."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Ratios must sum to 1.0"
    rng = random.Random(seed)
    shuffled = list(documents)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))

    train_docs = shuffled[:n_train]
    val_docs = shuffled[n_train:n_train + n_val]
    test_docs = shuffled[n_train + n_val:]

    return {
        "train": train_docs,
        "val": val_docs,
        "test": test_docs
    }
