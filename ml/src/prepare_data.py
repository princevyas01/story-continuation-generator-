"""Data preparation module: cleans, normalizes, and splits story datasets."""

import os
import re
import json
import unicodedata
import random
from typing import List, Tuple, Dict, Any
from ml.src.tokenizer import clean_and_tokenize

def normalize_text(text: str) -> str:
    """Normalize text with consistent Unicode, punctuation, and whitespace handling."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("“", "\"").replace("”", "\"")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", " - ").replace("–", " - ")
    text = text.replace("`", "'")
    text = re.sub(r'([.,!?;:\"\'()\[\]{}])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer metadata blocks."""
    start_match = re.search(r'\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*', text, re.IGNORECASE)
    if start_match:
        text = text[start_match.end():]

    end_match = re.search(r'\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*', text, re.IGNORECASE)
    if end_match:
        text = text[:end_match.start()]
    else:
        end_alt = re.search(r'End of (the )?Project Gutenberg', text, re.IGNORECASE)
        if end_alt:
            text = text[:end_alt.start()]

    return text.strip()

def split_documents(
    documents: List[Dict[str, Any]], 
    train_ratio: float = 0.80, 
    val_ratio: float = 0.10, 
    test_ratio: float = 0.10, 
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Perform deterministic document-level splitting to prevent data leakage."""
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-4, f"Split ratios must sum to 1.0 (got {total})"

    docs_copy = list(documents)
    rng = random.Random(seed)
    rng.shuffle(docs_copy)

    n_total = len(docs_copy)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio)) if n_total >= 3 else 0
    
    train_docs = docs_copy[:n_train]
    val_docs = docs_copy[n_train:n_train + n_val]
    test_docs = docs_copy[n_train + n_val:]

    if not test_docs and len(train_docs) > 2:
        test_docs = [train_docs.pop()]
    if not val_docs and len(train_docs) > 2:
        val_docs = [train_docs.pop()]

    return train_docs, val_docs, test_docs

def load_and_preprocess_directory(raw_dir: str, min_words: int = 30) -> List[Dict[str, Any]]:
    """Load, clean, and tokenize all .txt files from raw directory."""
    documents = []
    if not os.path.exists(raw_dir):
        return documents

    for filename in sorted(os.listdir(raw_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(raw_dir, filename)
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            content = strip_gutenberg_boilerplate(content)
            norm = normalize_text(content)
            tokens = clean_and_tokenize(norm)

            if len(tokens) >= min_words:
                documents.append({
                    "id": filename,
                    "title": filename.replace(".txt", "").replace("_", " ").title(),
                    "text": norm,
                    "tokens": tokens,
                    "token_count": len(tokens)
                })

    return documents
