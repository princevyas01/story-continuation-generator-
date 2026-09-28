"""UI helper functions and caching utilities for Streamlit demonstration."""

import os
import sys
import json
import pandas as pd
import streamlit as st
from typing import Dict, Any, Optional, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import load_config, resolve_path
from src.tokenizer import StoryTokenizer
from src.model import load_trained_model

@st.cache_resource(show_spinner="Loading trained LSTM model and tokenizer...")
def load_cached_resources(config_path: str = "configs/config.yaml") -> Tuple[Optional[Any], Optional[StoryTokenizer], Dict[str, Any], Optional[str]]:
    """Load model, tokenizer, and config with Streamlit caching to prevent reloads on interaction."""
    try:
        cfg = load_config(config_path)
        bdir = cfg["base_dir"]
        
        model_path = resolve_path(bdir, cfg["training"]["best_model_path"])
        vocab_path = resolve_path(bdir, cfg["tokenization"]["vocab_path"])

        if not os.path.exists(model_path):
            return None, None, cfg, f"Model file not found at: {model_path}. Please run training first."
        if not os.path.exists(vocab_path):
            return None, None, cfg, f"Tokenizer artifact not found at: {vocab_path}."

        tokenizer = StoryTokenizer.load(vocab_path)
        model = load_trained_model(model_path)
        return model, tokenizer, cfg, None
    except Exception as e:
        return None, None, {}, f"Error loading resources: {str(e)}"

@st.cache_data(show_spinner=False)
def load_metrics_and_artifacts(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load precomputed metrics and dataset manifest."""
    try:
        cfg = load_config(config_path)
        bdir = cfg["base_dir"]
        metrics_p = resolve_path(bdir, cfg["artifacts"]["metrics_json"])
        manifest_p = resolve_path(bdir, cfg["data"]["manifest_path"])
        metrics_csv_p = resolve_path(bdir, cfg["artifacts"]["metrics_csv"])

        metrics = {}
        manifest = {}
        history_df = None

        if os.path.exists(metrics_p):
            with open(metrics_p, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        if os.path.exists(manifest_p):
            with open(manifest_p, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        if os.path.exists(metrics_csv_p):
            history_df = pd.read_csv(metrics_csv_p)

        return {
            "metrics": metrics,
            "manifest": manifest,
            "history_df": history_df,
            "cfg": cfg
        }
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(show_spinner=False)
def load_precomputed_samples(config_path: str = "configs/config.yaml"):
    """Load precomputed generation samples."""
    try:
        cfg = load_config(config_path)
        bdir = cfg["base_dir"]
        samples_p = resolve_path(bdir, cfg["artifacts"]["sample_generations_json"])
        if os.path.exists(samples_p):
            with open(samples_p, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception:
        return []
