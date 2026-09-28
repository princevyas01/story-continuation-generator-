"""Tests for word tokenizer, special tokens, encoding/decoding, and serialization."""

import os
import tempfile
import pytest
from src.tokenizer import StoryTokenizer

def test_special_tokens_presence():
    tok = StoryTokenizer(max_vocab_size=100)
    assert tok.pad_token in tok.word2idx
    assert tok.unk_token in tok.word2idx
    assert tok.start_token in tok.word2idx
    assert tok.end_token in tok.word2idx
    assert tok.word2idx[tok.pad_token] == tok.pad_idx
    assert tok.word2idx[tok.unk_token] == tok.unk_idx

def test_encode_decode_deterministic():
    tok = StoryTokenizer(max_vocab_size=100)
    texts = [["the", "dragon", "slept", "in", "the", "cave"]]
    tok.fit_on_texts(texts)

    encoded = tok.encode(["the", "dragon", "slept"])
    assert len(encoded) == 3
    decoded = tok.decode(encoded)
    assert decoded == "the dragon slept"

def test_unknown_tokens_mapping():
    tok = StoryTokenizer(max_vocab_size=10)
    texts = [["apple", "banana", "cherry"]]
    tok.fit_on_texts(texts)

    encoded = tok.encode(["alien_word_xyz"])
    assert encoded == [tok.unk_idx]
    assert tok.decode(encoded) == tok.unk_token

def test_tokenizer_save_reload():
    tok = StoryTokenizer(max_vocab_size=50)
    texts = [["once", "upon", "a", "time", "in", "a", "faraway", "land"]]
    tok.fit_on_texts(texts)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "vocab.json")
        tok.save(filepath)

        reloaded = StoryTokenizer.load(filepath)
        assert reloaded.vocab_size == tok.vocab_size
        assert reloaded.word2idx == tok.word2idx
        assert reloaded.idx2word == tok.idx2word
        assert reloaded.encode(["once", "time"]) == tok.encode(["once", "time"])
