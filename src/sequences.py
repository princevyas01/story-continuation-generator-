"""Sequence creation and tf.data pipeline builders for autoregressive language modeling."""

from typing import List, Tuple, Generator
import numpy as np
from src.tokenizer import StoryTokenizer

def create_sequences_from_token_ids(
    token_ids: List[int],
    seq_len: int = 50,
    step: int = 1,
    pad_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (X, y) sliding window pairs from a single tokenized document.
    
    If token_ids is shorter than seq_len + 1, left-pads the sequence.
    """
    if len(token_ids) < 2:
        return np.empty((0, seq_len), dtype=np.int32), np.empty((0,), dtype=np.int32)

    if len(token_ids) <= seq_len:
        # Left-pad to fit seq_len
        pad_count = seq_len + 1 - len(token_ids)
        padded = [pad_idx] * pad_count + token_ids
        x = np.array([padded[:-1]], dtype=np.int32)
        y = np.array([padded[-1]], dtype=np.int32)
        return x, y

    x_list = []
    y_list = []
    for i in range(0, len(token_ids) - seq_len, step):
        x_list.append(token_ids[i : i + seq_len])
        y_list.append(token_ids[i + seq_len])

    return np.array(x_list, dtype=np.int32), np.array(y_list, dtype=np.int32)

def build_dataset_arrays(
    documents_tokens: List[List[str]],
    tokenizer: StoryTokenizer,
    seq_len: int = 50,
    step: int = 1,
    include_boundaries: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a collection of tokenized documents into concatenated (X, y) numpy arrays."""
    all_x = []
    all_y = []

    for tokens in documents_tokens:
        if not tokens:
            continue
        if include_boundaries:
            doc_tokens = [tokenizer.start_token] + tokens + [tokenizer.end_token]
        else:
            doc_tokens = tokens

        token_ids = tokenizer.encode(doc_tokens)
        x_doc, y_doc = create_sequences_from_token_ids(
            token_ids=token_ids,
            seq_len=seq_len,
            step=step,
            pad_idx=tokenizer.pad_idx
        )
        if len(x_doc) > 0:
            all_x.append(x_doc)
            all_y.append(y_doc)

    if not all_x:
        return np.empty((0, seq_len), dtype=np.int32), np.empty((0,), dtype=np.int32)

    return np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)

def sequence_generator(
    documents_tokens: List[List[str]],
    tokenizer: StoryTokenizer,
    seq_len: int = 50,
    step: int = 1,
    include_boundaries: bool = True
) -> Generator[Tuple[np.ndarray, int], None, None]:
    """Memory-efficient generator yielding single (X, y) samples one by one."""
    for tokens in documents_tokens:
        if not tokens:
            continue
        if include_boundaries:
            doc_tokens = [tokenizer.start_token] + tokens + [tokenizer.end_token]
        else:
            doc_tokens = tokens

        token_ids = tokenizer.encode(doc_tokens)
        x_doc, y_doc = create_sequences_from_token_ids(
            token_ids=token_ids,
            seq_len=seq_len,
            step=step,
            pad_idx=tokenizer.pad_idx
        )
        for i in range(len(x_doc)):
            yield x_doc[i], y_doc[i]
