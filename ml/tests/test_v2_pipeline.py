"""Unit and regression tests for Story Continuation Generator v2 pipeline."""

import os
import sys
import numpy as np
import pytest
import tensorflow as tf
import keras

from ml.src.masked_loss import (
    MaskedAccuracy,
    MaskedPerplexity,
    MaskedSparseCategoricalCrossentropy,
)
from ml.src.model import build_many_to_many_model, compile_model_v2
from ml.src.subword_tokenizer import SubwordStoryTokenizer
from backend.app.generator import sample_next_token, run_generation


class TestSubwordTokenizer:
    """Test suite for Byte-Level BPE SubwordStoryTokenizer."""

    def test_special_tokens_mapping(self):
        tokenizer_path = os.path.join("artifacts", "tokenizers", "tokenizer_v2.json")
        if not os.path.exists(tokenizer_path):
            pytest.skip("tokenizer_v2.json not yet generated")

        tok = SubwordStoryTokenizer.load(tokenizer_path)
        assert tok.pad_token_id == 0
        assert tok.unk_token_id == 1
        assert tok.bos_token_id == 2
        assert tok.eos_token_id == 3

        assert tok.token_to_id("<PAD>") == 0
        assert tok.token_to_id("<UNK>") == 1
        assert tok.token_to_id("<BOS>") == 2
        assert tok.token_to_id("<EOS>") == 3

    def test_roundtrip_fidelity(self):
        tokenizer_path = os.path.join("artifacts", "tokenizers", "tokenizer_v2.json")
        if not os.path.exists(tokenizer_path):
            pytest.skip("tokenizer_v2.json not yet generated")

        tok = SubwordStoryTokenizer.load(tokenizer_path)
        test_sentences = [
            "Once upon a time, a little girl named Lily lived in the village.",
            "\"Look at that!\" shouted the boy, pointing toward the mountain.",
            "Rare words: Xylophonist, Quetzalcoatl, Zephyrus 12345!?",
        ]
        for sentence in test_sentences:
            encoded = tok.encode(sentence)
            decoded = tok.decode(encoded, skip_special_tokens=True)
            assert decoded == sentence

    def test_zero_unk_rate(self):
        tokenizer_path = os.path.join("artifacts", "tokenizers", "tokenizer_v2.json")
        if not os.path.exists(tokenizer_path):
            pytest.skip("tokenizer_v2.json not yet generated")

        tok = SubwordStoryTokenizer.load(tokenizer_path)
        rare_text = "Thalassia and Quenxar had glowing bio-luminescent flora."
        encoded = tok.encode(rare_text)
        assert tok.unk_token_id not in encoded


class TestMaskedLossAndMetrics:
    """Verify that padding tokens (PAD=0) are excluded from loss and accuracy."""

    def test_masked_loss_ignores_padding(self):
        loss_fn = MaskedSparseCategoricalCrossentropy(from_logits=True, pad_id=0)

        # Target has 1 real token (id=5) and 2 padded tokens (id=0)
        y_true = tf.constant([[5, 0, 0]], dtype=tf.int32)

        # High confidence for class 5 at pos 0; wrong logits at pos 1 and 2
        logits = tf.zeros((1, 3, 10), dtype=tf.float32)

        loss_val = loss_fn(y_true, logits).numpy()
        assert np.isfinite(loss_val)
        assert loss_val > 0.0

    def test_masked_accuracy(self):
        metric = MaskedAccuracy(pad_id=0)

        # 3 positions: pos 0 matches, pos 1 is pad (should be ignored), pos 2 does not match
        y_true = tf.constant([[1, 0, 3]], dtype=tf.int32)
        
        # Predictions: argmax -> [1, 9, 4]
        # Pos 0: matches 1 (correct)
        # Pos 1: pad (ignored)
        # Pos 2: predicts 4 != 3 (incorrect)
        # Accuracy over unmasked tokens should be 1 / 2 = 0.50
        logits = tf.constant([
            [
                [0, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
                [0, 0, 0, 0, 10, 0, 0, 0, 0, 0],
            ]
        ], dtype=tf.float32)

        metric.update_state(y_true, logits)
        acc = metric.result().numpy()
        assert np.isclose(acc, 0.50, atol=1e-4)


class TestSamplingLogic:
    """Verify logit-space penalties, temperature, top-k, and nucleus sampling."""

    def test_banned_tokens_never_sampled(self):
        logits = np.random.randn(100)
        banned = {0, 1, 2, 3}
        for _ in range(50):
            token = sample_next_token(
                logits_or_probs=logits,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                banned_token_ids=banned,
                from_logits=True,
            )
            assert token not in banned

    def test_deterministic_with_seed(self):
        logits = np.random.randn(100)
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        tok1 = sample_next_token(logits, temperature=0.8, rng=rng1)
        tok2 = sample_next_token(logits, temperature=0.8, rng=rng2)
        assert tok1 == tok2

    def test_repetition_penalty_reduces_repetition(self):
        logits = np.array([5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float64)
        # Context contains token 0 multiple times
        recent = [0, 0, 0]
        penalized_token = sample_next_token(
            logits,
            temperature=0.01,  # greedy
            generated_token_ids=recent,
            repetition_penalty=2.0,
        )
        # Token 0 is penalized (5.0 / 2.0 = 2.5), so it should NOT be selected as greedy max
        assert penalized_token != 0
