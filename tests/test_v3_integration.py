"""Comprehensive integration and regression tests for Story LSTM v3 candidate."""

import os
import sys
import json
import numpy as np
import pytest
import tensorflow as tf
from fastapi.testclient import TestClient

# Ensure root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.config import settings
from backend.app.model_service import model_service, verify_v3_bundle, compute_sha256
from backend.app.generator import sample_next_token, run_generation
from backend.app.main import app
from ml.src.subword_tokenizer import SubwordStoryTokenizer


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestV3ArtifactIntegrity:
    """Artifact presence and SHA-256 hash validation."""

    def test_v3_artifacts_presence(self):
        assert os.path.exists(settings.MODEL_V3_PATH), f"Missing {settings.MODEL_V3_PATH}"
        assert os.path.exists(settings.TOKENIZER_V3_PATH), f"Missing {settings.TOKENIZER_V3_PATH}"
        assert os.path.exists(settings.METADATA_V3_PATH), f"Missing {settings.METADATA_V3_PATH}"
        assert os.path.exists(settings.METRICS_V3_PATH), f"Missing {settings.METRICS_V3_PATH}"

        assert os.path.getsize(settings.MODEL_V3_PATH) > 0
        assert os.path.getsize(settings.TOKENIZER_V3_PATH) > 0
        assert os.path.getsize(settings.METADATA_V3_PATH) > 0

    def test_v3_sha256_verification(self):
        with open(settings.METADATA_V3_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        expected_model_hash = metadata["model_sha256"].lower()
        expected_tok_hash = metadata["tokenizer_sha256"].lower()

        actual_model_hash = compute_sha256(settings.MODEL_V3_PATH)
        actual_tok_hash = compute_sha256(settings.TOKENIZER_V3_PATH)

        assert actual_model_hash == expected_model_hash, "v3 Model SHA-256 mismatch"
        assert actual_tok_hash == expected_tok_hash, "v3 Tokenizer SHA-256 mismatch"

        valid, msg = verify_v3_bundle(
            settings.MODEL_V3_PATH,
            settings.TOKENIZER_V3_PATH,
            settings.METADATA_V3_PATH,
        )
        assert valid is True
        assert "verified successfully" in msg


class TestV3Tokenizer:
    """Tokenizer loading, special token mapping, and vocabulary behavior."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return SubwordStoryTokenizer.load(settings.TOKENIZER_V3_PATH)

    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 8000

    def test_special_tokens_mapping(self, tokenizer):
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 1
        assert tokenizer.bos_token_id == 2
        assert tokenizer.eos_token_id == 3

        assert tokenizer.token_to_id("<PAD>") == 0
        assert tokenizer.token_to_id("<UNK>") == 1
        assert tokenizer.token_to_id("<BOS>") == 2
        assert tokenizer.token_to_id("<EOS>") == 3

    def test_encode_decode_roundtrip(self, tokenizer):
        prompt = "Once upon a time in a faraway kingdom, there lived a friendly dragon."
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        assert len(encoded) > 0
        assert all(0 <= tid < 8000 for tid in encoded)

        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        assert decoded == prompt


class TestV3ModelAndShapes:
    """Model architecture, logits output, and tensor shape contracts."""

    @pytest.fixture(scope="class")
    def model_and_tok(self):
        tok = SubwordStoryTokenizer.load(settings.TOKENIZER_V3_PATH)
        import keras
        model = keras.models.load_model(settings.MODEL_V3_PATH, compile=False)
        return model, tok

    def test_input_output_shapes(self, model_and_tok):
        model, _ = model_and_tok
        dummy_input = np.zeros((1, 128), dtype=np.int32)
        out = model(dummy_input, training=False)
        # Expected: raw logits shape [1, 128, 8000]
        assert out.shape == (1, 128, 8000)
        assert out.dtype == tf.float32

    def test_logits_range_and_finite(self, model_and_tok):
        model, tok = model_and_tok
        tokens = tok.encode("The brave adventurer stepped forward")
        arr = np.full((1, 128), tok.pad_token_id, dtype=np.int32)
        arr[0, :len(tokens)] = tokens
        last_valid = len(tokens) - 1

        out = model(arr, training=False).numpy()
        logits = out[0, last_valid, :]
        assert logits.shape == (8000,)
        assert np.all(np.isfinite(logits))
        # Logits should not be normalized probabilities (sum != 1.0)
        assert abs(float(np.sum(logits)) - 1.0) > 1.0


class TestV3InferenceAndSampling:
    """Prompt handling across lengths, padding, sampling math, and token constraints."""

    @pytest.fixture(scope="class")
    def v3_model_tok(self):
        tok = SubwordStoryTokenizer.load(settings.TOKENIZER_V3_PATH)
        import keras
        model = keras.models.load_model(settings.MODEL_V3_PATH, compile=False)
        return model, tok

    def test_single_token_prompt(self, v3_model_tok):
        model, tok = v3_model_tok
        text, ms = run_generation(
            model=model,
            tokenizer=tok,
            prompt="A",
            max_new_tokens=10,
            temperature=0.8,
            model_version="v3",
            seed=42,
        )
        assert len(text) > 0
        assert ms >= 0
        assert "<PAD>" not in text
        assert "<BOS>" not in text

    def test_normal_length_prompt(self, v3_model_tok):
        model, tok = v3_model_tok
        text, ms = run_generation(
            model=model,
            tokenizer=tok,
            prompt="Once upon a time, a young girl found a silver key in the forest.",
            max_new_tokens=15,
            temperature=0.7,
            top_k=40,
            model_version="v3",
            seed=42,
        )
        assert len(text.strip()) > 0
        assert "<PAD>" not in text

    def test_exact_128_tokens_prompt(self, v3_model_tok):
        model, tok = v3_model_tok
        # Build prompt that encodes to exactly 128 tokens
        base = "word "
        prompt = ""
        while len(tok.encode(prompt)) < 128:
            prompt += "story "
        tokens = tok.encode(prompt)[:128]
        prompt_128 = tok.decode(tokens)

        text, ms = run_generation(
            model=model,
            tokenizer=tok,
            prompt=prompt_128,
            max_new_tokens=10,
            temperature=0.7,
            model_version="v3",
            seed=42,
        )
        assert isinstance(text, str)
        assert ms >= 0

    def test_prompt_longer_than_128_tokens(self, v3_model_tok):
        model, tok = v3_model_tok
        long_prompt = "In the deep magical kingdom of Eldoria, " * 35
        assert len(tok.encode(long_prompt)) > 128

        text, ms = run_generation(
            model=model,
            tokenizer=tok,
            prompt=long_prompt,
            max_new_tokens=10,
            temperature=0.7,
            model_version="v3",
            seed=42,
        )
        assert isinstance(text, str)
        assert ms >= 0

    def test_special_tokens_excluded(self):
        # Verify sample_next_token never samples banned tokens
        logits = np.zeros(8000, dtype=np.float64)
        logits[0] = 100.0  # PAD
        logits[1] = 100.0  # UNK
        logits[2] = 100.0  # BOS
        logits[50] = 5.0   # Normal token

        sampled = sample_next_token(
            logits_or_probs=logits,
            temperature=0.01,
            banned_token_ids={0, 1, 2},
            from_logits=True,
        )
        assert sampled not in {0, 1, 2}
        assert sampled == 50

    def test_repetition_penalty_and_temperature(self):
        logits = np.ones(100, dtype=np.float64)
        logits[10] = 5.0

        # Repetition penalty should penalize token 10 if it appeared recently
        sampled_penalized = sample_next_token(
            logits_or_probs=logits,
            temperature=0.01,  # greedy
            generated_token_ids=[10, 10, 10],
            repetition_penalty=2.0,
            from_logits=True,
        )
        # Token 10 logit was 5.0 / 2.0 = 2.5; still highest among 1.0s, but test penalty applied
        assert sampled_penalized == 10

        # High repetition penalty should suppress it below 1.0
        sampled_suppressed = sample_next_token(
            logits_or_probs=logits,
            temperature=0.01,
            generated_token_ids=[10],
            repetition_penalty=10.0,  # 5.0 / 10.0 = 0.5 < 1.0
            from_logits=True,
        )
        assert sampled_suppressed != 10


class TestModelServiceAndRollback:
    """Model registry, fallback, version reporting, and rollback."""

    def test_v2_v3_coexistence_in_service(self):
        model_service.load_artifacts(preferred_version="v2")
        assert model_service.is_loaded
        status = model_service.get_status()
        assert status["status"] == "ready"
        assert "v2" in status["available_versions"]
        assert "v3" in status["available_versions"]

    def test_active_version_switching(self):
        # Switch to v3
        ok = model_service.set_active_version("v3")
        assert ok is True
        assert model_service.active_version == "v3"
        status_v3 = model_service.get_status()
        assert status_v3["version"] == "3.0.0"
        assert status_v3["parameters_count"] == 4577088

        # Generate with v3
        text_v3, ms_v3 = model_service.generate("Once upon a time", max_new_tokens=5, seed=42)
        assert len(text_v3) > 0

        # Switch back to v2 (rollback)
        ok = model_service.set_active_version("v2")
        assert ok is True
        assert model_service.active_version == "v2"
        status_v2 = model_service.get_status()
        assert status_v2["version"] == "2.0.0"

        # Generate with v2
        text_v2, ms_v2 = model_service.generate("Once upon a time", max_new_tokens=5, seed=42)
        assert len(text_v2) > 0

    def test_corrupt_v3_fallback_to_v2(self, tmp_path):
        # Test verify_v3_bundle with mismatched hash
        fake_meta = tmp_path / "model_v3.json"
        fake_meta.write_text(json.dumps({
            "model_sha256": "wronghash",
            "tokenizer_sha256": "wronghash"
        }), encoding="utf-8")

        valid, msg = verify_v3_bundle(
            settings.MODEL_V3_PATH,
            settings.TOKENIZER_V3_PATH,
            str(fake_meta),
        )
        assert valid is False
        assert "mismatch" in msg


class TestAPIEndpointsV3:
    """FastAPI route compatibility with v3 candidate."""

    def test_status_endpoint(self, client):
        resp = client.get("/api/v1/model/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert "available_versions" in data

    def test_metrics_endpoint(self, client):
        resp = client.get("/api/v1/model/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

    def test_generate_v3_override(self, client):
        resp = client.post("/api/v1/generate", json={
            "prompt": "The little bear wanted to climb the mountain",
            "max_new_tokens": 10,
            "temperature": 0.7,
            "model_version": "v3",
            "seed": 42
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "generated_text" in data
        assert len(data["generated_text"]) > 0
        assert data["settings"]["model_version"] == "v3"

    def test_generate_v2_regression(self, client):
        resp = client.post("/api/v1/generate", json={
            "prompt": "Once upon a time",
            "max_new_tokens": 10,
            "temperature": 0.8,
            "model_version": "v2",
            "seed": 42
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "generated_text" in data
        assert data["settings"]["model_version"] == "v2"
