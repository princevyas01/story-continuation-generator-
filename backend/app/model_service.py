"""Model Service managing the singleton LSTM model and tokenizer lifecycle.

Supports:
- v3 Candidate: Subword Byte-Level BPE (8,000 vocab) with 2-layer Stacked LSTM (raw logits, right-padded).
- v2 Baseline: Subword Byte-Level BPE (8,000 vocab) with Many-to-Many Stacked LSTM.
- v1 Fallback: Word-level Many-to-One Stacked LSTM.
"""

import hashlib
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
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

logger = logging.getLogger("model_service")


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def verify_v3_bundle(
    model_path: str, tokenizer_path: str, metadata_path: str
) -> Tuple[bool, str]:
    """Verify presence and SHA-256 checksums of v3 model and tokenizer against metadata."""
    if not os.path.exists(model_path):
        return False, f"v3 model file missing at {model_path}"
    if not os.path.exists(tokenizer_path):
        return False, f"v3 tokenizer file missing at {tokenizer_path}"
    if not os.path.exists(metadata_path):
        return False, f"v3 metadata file missing at {metadata_path}"

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        expected_model_hash = meta.get("model_sha256", "").lower()
        expected_tok_hash = meta.get("tokenizer_sha256", "").lower()

        actual_model_hash = compute_sha256(model_path)
        actual_tok_hash = compute_sha256(tokenizer_path)

        if actual_model_hash != expected_model_hash:
            return (
                False,
                f"v3 model SHA-256 mismatch: expected {expected_model_hash}, got {actual_model_hash}",
            )
        if actual_tok_hash != expected_tok_hash:
            return (
                False,
                f"v3 tokenizer SHA-256 mismatch: expected {expected_tok_hash}, got {actual_tok_hash}",
            )

        return True, "v3 checksums verified successfully"
    except Exception as exc:
        return False, f"Verification failed with exception: {str(exc)}"


class ModelService:
    """Thread-safe singleton managing the LSTM model and tokenizer registry."""

    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._initialized = False
        self._active_version: str = "v2"

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return (
                self._initialized
                and self._active_version in self._registry
                and self._registry[self._active_version]["model"] is not None
            )

    @property
    def active_version(self) -> str:
        return self._active_version

    def _normalize_version_key(self, version: Optional[str]) -> str:
        if not version:
            return self._active_version
        v = version.lower().strip()
        if "3" in v or "colab" in v:
            return "v3"
        if "2" in v:
            return "v2"
        if "1" in v:
            return "v1"
        return v

    def _load_v3(self) -> bool:
        """Attempt to load v3 candidate model and tokenizer pair."""
        model_path = settings.MODEL_V3_PATH
        tok_path = settings.TOKENIZER_V3_PATH
        meta_path = settings.METADATA_V3_PATH

        is_valid, msg = verify_v3_bundle(model_path, tok_path, meta_path)
        if not is_valid:
            logger.warning(f"[ModelService] v3 bundle verification failed: {msg}")
            return False

        logger.info(f"[ModelService] Loading v3 tokenizer from: {tok_path}")
        tokenizer = SubwordStoryTokenizer.load(tok_path)

        logger.info(f"[ModelService] Loading v3 LSTM model from: {model_path}")
        model = keras.models.load_model(model_path, compile=False)

        # Warm up forward pass
        dummy_input = tf.zeros((1, 128), dtype=tf.int32)
        _ = model(dummy_input, training=False)

        # Load metrics
        metrics: Dict[str, Any] = {}
        if os.path.exists(settings.METRICS_V3_PATH):
            with open(settings.METRICS_V3_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        self._registry["v3"] = {
            "model": model,
            "tokenizer": tokenizer,
            "metrics": metrics,
            "version_label": "3.0.0",
            "tokenizer_type": "subword_bytelevel_bpe_v3",
            "model_version": "v3",
        }
        logger.info("[ModelService] v3 model loaded successfully.")
        return True

    def _load_v2(self) -> bool:
        """Attempt to load v2 baseline model and tokenizer pair."""
        v2_tok_path = settings.TOKENIZER_V2_PATH
        v2_model_path = (
            settings.MODEL_V2_PATH
            if os.path.exists(settings.MODEL_V2_PATH)
            else settings.MODEL_V2_BEST_PATH
        )

        if not (os.path.exists(v2_tok_path) and os.path.exists(v2_model_path)):
            logger.warning("[ModelService] v2 artifacts not found.")
            return False

        logger.info(f"[ModelService] Loading v2 tokenizer from: {v2_tok_path}")
        tokenizer = SubwordStoryTokenizer.load(v2_tok_path)

        logger.info(f"[ModelService] Loading v2 LSTM model from: {v2_model_path}")
        model = keras.models.load_model(v2_model_path)

        # Warm up forward pass
        dummy_input = tf.zeros((1, 128), dtype=tf.int32)
        _ = model(dummy_input, training=False)

        metrics: Dict[str, Any] = {}
        if os.path.exists(settings.METRICS_V2_PATH):
            with open(settings.METRICS_V2_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        self._registry["v2"] = {
            "model": model,
            "tokenizer": tokenizer,
            "metrics": metrics,
            "version_label": "2.0.0",
            "tokenizer_type": "subword_bytelevel_bpe_v2",
            "model_version": "v2",
        }
        logger.info("[ModelService] v2 model loaded successfully.")
        return True

    def _load_v1(self) -> bool:
        """Fallback to v1 legacy word-level model and tokenizer."""
        tokenizer_path = settings.TOKENIZER_PATH
        if not os.path.exists(tokenizer_path):
            tokenizer_path = settings.LEGACY_TOKENIZER_PATH

        model_path = settings.MODEL_PATH
        if not os.path.exists(model_path):
            model_path = settings.LEGACY_MODEL_PATH

        if not (os.path.exists(tokenizer_path) and os.path.exists(model_path)):
            return False

        logger.info(f"[ModelService] Loading v1 tokenizer from: {tokenizer_path}")
        tokenizer = StoryTokenizer.load(tokenizer_path)

        logger.info(f"[ModelService] Loading v1 model from: {model_path}")
        model = keras.models.load_model(model_path)

        dummy_input = tf.zeros((1, 50), dtype=tf.int32)
        _ = model(dummy_input, training=False)

        metrics: Dict[str, Any] = {}
        if os.path.exists(settings.METRICS_PATH):
            with open(settings.METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)

        self._registry["v1"] = {
            "model": model,
            "tokenizer": tokenizer,
            "metrics": metrics,
            "version_label": "1.0.0",
            "tokenizer_type": "word_level",
            "model_version": "v1",
        }
        return True

    def load_artifacts(self, preferred_version: Optional[str] = None) -> None:
        """Load requested model version and preserve available rollback versions."""
        with self._lock:
            target = preferred_version or settings.ACTIVE_MODEL_VERSION
            target_key = self._normalize_version_key(target)

            # Try loading v2 first if present (ensures rollback baseline is always warm)
            v2_loaded = self._load_v2()

            # Try loading v3 candidate
            v3_loaded = self._load_v3()

            if not v2_loaded and not v3_loaded:
                # Try v1 fallback
                v1_loaded = self._load_v1()
                if not v1_loaded:
                    raise FileNotFoundError("No valid model/tokenizer pair could be loaded.")
                self._active_version = "v1"
            elif target_key == "v3":
                if v3_loaded:
                    self._active_version = "v3"
                else:
                    logger.warning("[ModelService] Requested v3 failed to load; rolling back to v2.")
                    self._active_version = "v2"
            else:
                if v2_loaded:
                    self._active_version = "v2"
                elif v3_loaded:
                    self._active_version = "v3"

            self._initialized = True
            active_info = self._registry.get(self._active_version, {})
            model = active_info.get("model")
            tok = active_info.get("tokenizer")
            params = int(model.count_params()) if model else 0
            vocab = tok.vocab_size if tok else 0
            logger.info(
                f"[ModelService] Initialized with active model {self._active_version} "
                f"({active_info.get('version_label')}, vocab={vocab}, params={params:,}). "
                f"Available versions: {list(self._registry.keys())}"
            )

    def set_active_version(self, version: str) -> bool:
        """Switch active model version at runtime."""
        key = self._normalize_version_key(version)
        with self._lock:
            if key in self._registry:
                self._active_version = key
                logger.info(f"[ModelService] Switched active model to: {key}")
                return True
            else:
                logger.error(f"[ModelService] Cannot switch to unknown version: {version}")
                return False

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: Optional[float] = 0.90,
        repetition_penalty: float = 1.15,
        seed: Optional[int] = None,
        model_version: Optional[str] = None,
    ) -> Tuple[str, int]:
        """Thread-safe generation endpoint with optional version override."""
        if not self.is_loaded:
            raise RuntimeError("ModelService is not loaded. Call load_artifacts() first.")

        v_key = self._normalize_version_key(model_version)
        if v_key not in self._registry:
            v_key = self._active_version

        entry = self._registry[v_key]

        with self._lock:
            return run_generation(
                model=entry["model"],
                tokenizer=entry["tokenizer"],
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                model_version=v_key,
            )

    def get_status(self) -> Dict[str, Any]:
        """Return model metadata and loaded status."""
        active_info = self._registry.get(self._active_version, {})
        model = active_info.get("model")
        tok = active_info.get("tokenizer")
        return {
            "status": "ready" if self.is_loaded else "not_loaded",
            "model_name": "story_continuation_lstm",
            "version": active_info.get("version_label", "unknown"),
            "vocab_size": tok.vocab_size if tok else 0,
            "device": "CPU",
            "parameters_count": int(model.count_params()) if model else 0,
            "tokenizer_type": active_info.get("tokenizer_type", "unknown"),
            "active_version": self._active_version,
            "available_versions": list(self._registry.keys()),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return precomputed training/evaluation metrics."""
        active_info = self._registry.get(self._active_version, {})
        return {
            "status": "success",
            "version": active_info.get("version_label", "unknown"),
            "metrics": active_info.get("metrics", {}),
        }


model_service = ModelService()
