"""Model Service managing the singleton LSTM model and tokenizer lifecycle."""

import os
import json
import threading
from typing import Optional, Dict, Any, Tuple
import tensorflow as tf
from tensorflow import keras

from backend.app.config import settings
from ml.src.tokenizer import StoryTokenizer
from backend.app.generator import run_generation

class ModelService:
    """Thread-safe singleton managing the LSTM model and tokenizer."""

    def __init__(self):
        self._model: Optional[keras.Model] = None
        self._tokenizer: Optional[StoryTokenizer] = None
        self._metrics: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self._initialized = False

    @property
    def is_loaded(self) -> bool:
        return self._initialized and self._model is not None and self._tokenizer is not None

    def load_artifacts(self) -> None:
        """Load model, tokenizer, and metrics into memory."""
        with self._lock:
            if self._initialized:
                return

            # 1. Load Tokenizer
            tokenizer_path = settings.TOKENIZER_PATH
            if not os.path.exists(tokenizer_path):
                tokenizer_path = settings.LEGACY_TOKENIZER_PATH

            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

            print(f"[ModelService] Loading tokenizer from: {tokenizer_path}")
            self._tokenizer = StoryTokenizer.load(tokenizer_path)

            # 2. Load Model
            model_path = settings.MODEL_PATH
            if not os.path.exists(model_path):
                model_path = settings.LEGACY_MODEL_PATH

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weights not found at {model_path}")

            print(f"[ModelService] Loading LSTM model from: {model_path}")
            self._model = keras.models.load_model(model_path)

            # Warm up forward pass
            dummy_input = tf.zeros((1, 50), dtype=tf.int32)
            _ = self._model(dummy_input, training=False)

            # 3. Load Metrics if present
            if os.path.exists(settings.METRICS_PATH):
                with open(settings.METRICS_PATH, "r", encoding="utf-8") as f:
                    self._metrics = json.load(f)
            else:
                self._metrics = {"status": "default_metrics_not_found"}

            self._initialized = True
            print("[ModelService] Successfully initialized model service.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: Optional[float] = 0.90,
        seed: Optional[int] = None
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
                seed=seed
            )

    def get_status(self) -> Dict[str, Any]:
        """Return model metadata and loaded status."""
        return {
            "status": "ready" if self.is_loaded else "not_loaded",
            "model_name": "story_continuation_lstm",
            "version": settings.VERSION,
            "vocab_size": self._tokenizer.vocab_size if self._tokenizer else 0,
            "device": "CPU",
            "parameters_count": int(self._model.count_params()) if self._model else 0
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return precomputed training/evaluation metrics."""
        return {
            "status": "success",
            "metrics": self._metrics or {}
        }

model_service = ModelService()
