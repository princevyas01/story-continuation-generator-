"""Utility functions for ML pipeline: config loading, seed setting, and metrics."""

import os
import random
import math
from typing import Dict, Any, Optional, List
import yaml
import numpy as np

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "base.yaml"
)

def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    # ml/src/utils.py -> ml/src -> ml -> project_root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration and set project root."""
    target_path = config_path or DEFAULT_CONFIG_PATH
    if not os.path.exists(target_path):
        # Fallback to root configs/config.yaml if ml/configs/base.yaml is absent
        fallback = os.path.join(get_project_root(), "configs", "config.yaml")
        if os.path.exists(fallback):
            target_path = fallback
        else:
            raise FileNotFoundError(f"Configuration file not found: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["base_dir"] = get_project_root()
    return cfg

def resolve_path(base_dir: str, rel_path: str) -> str:
    """Resolve a relative path against base directory."""
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.normpath(os.path.join(base_dir, rel_path))

def set_seed(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and TensorFlow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def calculate_perplexity(loss: float, max_loss: float = 100.0) -> float:
    """Compute perplexity = exp(loss) with numerical stability."""
    if math.isnan(loss) or math.isinf(loss):
        return float("inf")
    clamped = min(loss, max_loss)
    try:
        return float(math.exp(clamped))
    except OverflowError:
        return float("inf")

def compute_distinct_n(tokens: List[str], n: int = 1) -> float:
    """Ratio of unique n-grams to total n-grams in tokens."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / float(len(ngrams))

def compute_repetition_ratio(tokens: List[str], n: int = 3) -> float:
    """Ratio of repeated n-grams: 1.0 - (unique n-grams / total n-grams)."""
    if len(tokens) < n:
        return 0.0
    return round(1.0 - compute_distinct_n(tokens, n=n), 4)

def evaluate_text_generation_quality(text: str) -> Dict[str, Any]:
    """Calculate token count, distinct-1, distinct-2, and repetition ratio."""
    tokens = [w.strip() for w in text.split() if w.strip()]
    if not tokens:
        return {
            "total_tokens": 0,
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "repetition_ratio_3gram": 0.0,
        }
    return {
        "total_tokens": len(tokens),
        "distinct_1": round(compute_distinct_n(tokens, n=1), 4),
        "distinct_2": round(compute_distinct_n(tokens, n=2), 4),
        "repetition_ratio_3gram": compute_repetition_ratio(tokens, n=3),
    }
