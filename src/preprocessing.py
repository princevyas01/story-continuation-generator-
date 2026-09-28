"""Text preprocessing, cleaning, normalization, and document splitting."""

import re
import unicodedata
from typing import List, Tuple, Dict, Any
import random

def normalize_text(text: str) -> str:
    """Normalize text with consistent Unicode, punctuation, and whitespace handling.
    
    Preserves sentence-level punctuation (.,!?;:\"') required for natural language flow.
    """
    if not text:
        return ""

    # Normalize Unicode (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # Normalize quotes and dashes
    text = text.replace("“", "\"").replace("”", "\"")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", " - ").replace("–", " - ")
    text = text.replace("`", "'")

    # Add spaces around punctuation for clean word-level tokenization
    # e.g., "word." -> "word ."
    text = re.sub(r'([.,!?;:\"\'()\[\]{}])', r' \1 ', text)

    # Replace multiple whitespaces and newlines with a single space
    text = re.sub(r'\s+', ' ', text)

    # Strip and lowercase for consistent vocabulary matching
    return text.strip().lower()

def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer metadata blocks."""
    # Find start marker
    start_match = re.search(r'\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*', text, re.IGNORECASE)
    if start_match:
        text = text[start_match.end():]

    # Find end marker
    end_match = re.search(r'\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*', text, re.IGNORECASE)
    if end_match:
        text = text[:end_match.start()]
    else:
        end_alt = re.search(r'End of (the )?Project Gutenberg', text, re.IGNORECASE)
        if end_alt:
            text = text[:end_alt.start()]

    return text.strip()

def tokenize_words(text: str) -> List[str]:
    """Split text into word-level tokens."""
    return [token for token in text.split() if token]

def split_documents(
    documents: List[Dict[str, Any]], 
    train_ratio: float = 0.80, 
    val_ratio: float = 0.10, 
    test_ratio: float = 0.10, 
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Perform deterministic document-level splitting to prevent data leakage.
    
    Splitting occurs at the story/document level before sliding windows are extracted.
    """
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

    # Ensure at least 1 document in val and test if possible
    if not test_docs and len(train_docs) > 2:
        test_docs = [train_docs.pop()]
    if not val_docs and len(train_docs) > 2:
        val_docs = [train_docs.pop()]

    return train_docs, val_docs, test_docs
