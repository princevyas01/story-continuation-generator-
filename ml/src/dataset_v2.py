"""Dataset pipeline for Many-to-Many Language Model (v2).

Constructs autoregressive sequences (X = tokens[:-1], Y = tokens[1:])
with PAD masking (PAD_ID = 0) and zero inter-story contamination.
"""

import json
import os
from typing import Tuple, List, Optional
import numpy as np
import tensorflow as tf

from ml.src.subword_tokenizer import SubwordStoryTokenizer


def build_sequences_from_jsonl(
    jsonl_path: str,
    tokenizer: SubwordStoryTokenizer,
    seq_len: int = 128,
    stride: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract shifted input-target pairs from distinct story documents.

    Never concatenates unrelated stories. Stories shorter than seq_len+1
    are padded to seq_len. Longer stories are chunked with the given stride.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    x_list: List[List[int]] = []
    y_list: List[List[int]] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item.get("text", "")
            if not text:
                continue

            # BOS and EOS ensure model learns narrative opening and closure
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) <= 1:
                continue

            if len(tokens) <= seq_len + 1:
                # Pad to seq_len
                x_seq = tokens[:-1]
                y_seq = tokens[1:]

                pad_len = seq_len - len(x_seq)
                if pad_len > 0:
                    x_seq = x_seq + [tokenizer.PAD_ID] * pad_len
                    y_seq = y_seq + [tokenizer.PAD_ID] * pad_len

                x_list.append(x_seq)
                y_list.append(y_seq)
            else:
                # Chunk with stride
                for start in range(0, len(tokens) - 1, stride):
                    chunk = tokens[start : start + seq_len + 1]
                    if len(chunk) <= 1:
                        break

                    x_seq = chunk[:-1]
                    y_seq = chunk[1:]

                    pad_len = seq_len - len(x_seq)
                    if pad_len > 0:
                        x_seq = x_seq + [tokenizer.PAD_ID] * pad_len
                        y_seq = y_seq + [tokenizer.PAD_ID] * pad_len

                    x_list.append(x_seq)
                    y_list.append(y_seq)

                    if len(chunk) < seq_len + 1:
                        break

    x_arr = np.array(x_list, dtype=np.int32)
    y_arr = np.array(y_list, dtype=np.int32)
    return x_arr, y_arr


def create_tf_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """Create an optimized tf.data.Dataset from NumPy sequence arrays."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 10000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
