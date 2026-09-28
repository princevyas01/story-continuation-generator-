"""Tests for sequence creation, sliding window logic, and array shapes."""

import numpy as np
import pytest
from src.tokenizer import StoryTokenizer
from src.sequences import create_sequences_from_token_ids, build_dataset_arrays

def test_sequence_shapes_and_alignment():
    token_ids = [10, 11, 12, 13, 14, 15, 16]
    seq_len = 3
    X, y = create_sequences_from_token_ids(token_ids, seq_len=seq_len, step=1)
    
    # Windows:
    # [10, 11, 12] -> 13
    # [11, 12, 13] -> 14
    # [12, 13, 14] -> 15
    # [13, 14, 15] -> 16
    assert X.shape == (4, 3)
    assert y.shape == (4,)
    np.testing.assert_array_equal(X[0], [10, 11, 12])
    assert y[0] == 13
    np.testing.assert_array_equal(X[-1], [13, 14, 15])
    assert y[-1] == 16

def test_short_sequence_padding():
    token_ids = [20, 21]
    seq_len = 5
    pad_idx = 0
    X, y = create_sequences_from_token_ids(token_ids, seq_len=seq_len, step=1, pad_idx=pad_idx)
    assert X.shape == (1, 5)
    assert y.shape == (1,)
    # Padded sequence: [0, 0, 0, 20] -> 21
    assert list(X[0]) == [0, 0, 0, 0, 20]
    assert y[0] == 21

def test_build_dataset_arrays():
    tok = StoryTokenizer(max_vocab_size=50)
    tok.fit_on_texts([["the", "sun", "rose", "high", "in", "the", "clear", "blue", "sky"]])
    
    docs = [["the", "sun", "rose", "high"], ["clear", "blue", "sky"]]
    X, y = build_dataset_arrays(docs, tok, seq_len=4, step=1)
    assert len(X) > 0
    assert X.shape[1] == 4
    assert len(y) == len(X)
