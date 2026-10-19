"""Model Service managing the singleton LSTM model and tokenizer lifecycle.

Supports:
- v2 Candidate: Subword Byte-Level BPE with Many-to-Many Stacked LSTM.
- v1 Fallback: Word-level Many-to-One Stacked LSTM.
"""

import json
import os
import threading
from typing import Any, Dict, Optional, Tuple, Union
import keras
import tensorflow as tf

from backend.app.config import settings
from backend.app.generator import run_generation
from ml.src.masked_loss import (
    MaskedAccuracy,
    MaskedPerplexity,
    MaskedSparseCategoricalCrossentropy,
)
from ml.src.subword_tokenizer import SubwordStoryTokenizer
from ml.src.tokenizer import StoryTokenizer


class ModelService:
    """Thread-safe singleton managing the LSTM model and tokenizer."""

    def __init__(self):
        self._model: Optional[keras.Model] = None
        self._tokenizer: Optional[Union[SubwordStoryTokenizer, StoryTokenizer]] = None
        self._metrics: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self._initialized = False
        self._model_version_label = "2.0.0"
        self._tokenizer_type = "subword_bytelevel_bpe"

    @property
    def is_loaded(self) -> bool:
        return self._initialized and self._model is not None and self._tokenizer is not None

    def load_artifacts(self) -> None:
        """Load model, tokenizer, and metrics with v2 preference and v1 fallback."""
        with self._lock:
            if self._initialized:
                return

            # Check for v2 candidate first
            v2_tok_path = settings.TOKENIZER_V2_PATH
            v2_model_path = (
                settings.MODEL_V2_PATH
                if os.path.exists(settings.MODEL_V2_PATH)
                else settings.MODEL_V2_BEST_PATH
            )

            if os.path.exists(v2_tok_path) and os.path.exists(v2_model_path):
                print(f"[ModelService] Loading v2 Subword Tokenizer from: {v2_tok_path}")
                self._tokenizer = SubwordStoryTokenizer.load(v2_tok_path)
                self._tokenizer_type = "subword_bytelevel_bpe"

                print(f"[ModelService] Loading v2 Many-to-Many LSTM model from: {v2_model_path}")
                self._model = keras.models.load_model(v2_model_path)
                self._model_version_label = "2.0.0"

                # Warm up forward pass
                dummy_input = tf.zeros((1, 128), dtype=tf.int32)
                _ = self._model(dummy_input, training=False)
            else:
                # Fallback to v1 legacy
                tokenizer_path = settings.TOKENIZER_PATH
                if not os.path.exists(tokenizer_path):
                    tokenizer_path = settings.LEGACY_TOKENIZER_PATH

                if not os.path.exists(tokenizer_path):
                    raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

                print(f"[ModelService] Loading v1 Legacy Tokenizer from: {tokenizer_path}")
                self._tokenizer = StoryTokenizer.load(tokenizer_path)
                self._tokenizer_type = "word_level"

                model_path = settings.MODEL_PATH
                if not os.path.exists(model_path):
                    model_path = settings.LEGACY_MODEL_PATH

                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model weights not found at {model_path}")

                print(f"[ModelService] Loading v1 Legacy LSTM model from: {model_path}")
                self._model = keras.models.load_model(model_path)
                self._model_version_label = "1.0.0"

                # Warm up forward pass
                dummy_input = tf.zeros((1, 50), dtype=tf.int32)
                _ = self._model(dummy_input, training=False)

            # Load Metrics
            if os.path.exists(settings.METRICS_V2_PATH):
                with open(settings.METRICS_V2_PATH, "r", encoding="utf-8") as f:
                    self._metrics = json.load(f)
            elif os.path.exists(settings.METRICS_PATH):
                with open(settings.METRICS_PATH, "r", encoding="utf-8") as f:
                    self._metrics = json.load(f)
            else:
                self._metrics = {"status": "default_metrics_not_found"}

            self._initialized = True
            print(
                f"[ModelService] Successfully initialized model service ({self._model_version_label}, "
                f"vocab_size={self._tokenizer.vocab_size}, params={int(self._model.count_params()):,})."
            )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: Optional[float] = 0.90,
        repetition_penalty: float = 1.15,
        seed: Optional[int] = None,
    ) -> Tuple[str, int]:
        """Thread-safe generation endpoint."""
        if not self.is_loaded:
            raise RuntimeError("ModelService is not loaded. Call load_artifacts() first.")

        with self._lock:
            return run_generation(
                model=self._model,
                tokenizer=self._tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )

    def get_status(self) -> Dict[str, Any]:
        """Return model metadata and loaded status."""
        return {
            "status": "ready" if self.is_loaded else "not_loaded",
            "model_name": "story_continuation_lstm",
            "version": self._model_version_label,
            "vocab_size": self._tokenizer.vocab_size if self._tokenizer else 0,
            "device": "CPU",
            "parameters_count": int(self._model.count_params()) if self._model else 0,
            "tokenizer_type": self._tokenizer_type,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return precomputed training/evaluation metrics."""
        return {
            "status": "success",
            "version": self._model_version_label,
            "metrics": self._metrics or {},
        }


model_service = ModelService()
