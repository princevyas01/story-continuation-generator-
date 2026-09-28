"""Word-level Tokenizer with deterministic vocabulary mapping and JSON persistence."""

import json
import os
from collections import Counter
from typing import List, Dict, Optional, Iterable

class StoryTokenizer:
    """Deterministic word-level tokenizer with special tokens and frequency pruning."""

    def __init__(
        self,
        max_vocab_size: int = 10000,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        start_token: str = "<START>",
        end_token: str = "<END>"
    ):
        self.max_vocab_size = max_vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.start_token = start_token
        self.end_token = end_token

        # Reserved special tokens with fixed indexing
        self.special_tokens = [self.pad_token, self.unk_token, self.start_token, self.end_token]
        self.pad_idx = 0
        self.unk_idx = 1
        self.start_idx = 2
        self.end_idx = 3

        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._initialize_specials()

    def _initialize_specials(self):
        """Map special tokens to fixed indices."""
        self.word2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.idx2word = {idx: token for idx, token in enumerate(self.special_tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def fit_on_texts(self, token_sequences: Iterable[List[str]]):
        """Build vocabulary from iterable of token lists based on frequency."""
        counter = Counter()
        for tokens in token_sequences:
            counter.update(tokens)

        # Remove special tokens from counter if they were in input
        for st in self.special_tokens:
            if st in counter:
                del counter[st]

        # Take top most frequent words up to max_vocab_size
        available_slots = self.max_vocab_size - len(self.special_tokens)
        most_common = counter.most_common(max(0, available_slots))

        self._initialize_specials()
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert list of word tokens into integer token IDs."""
        return [self.word2idx.get(w, self.unk_idx) for w in tokens]

    def decode(self, indices: List[int], skip_special: bool = False) -> str:
        """Convert token IDs back into text, with optional special token filtering."""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.unk_token)
            if skip_special and word in self.special_tokens:
                continue
            words.append(word)
        return " ".join(words)

    def save(self, filepath: str):
        """Save vocabulary and tokenizer metadata to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "max_vocab_size": self.max_vocab_size,
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "pad_idx": self.pad_idx,
            "unk_idx": self.unk_idx,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "StoryTokenizer":
        """Load tokenizer from JSON artifact."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tokenizer artifact not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(
            max_vocab_size=data["max_vocab_size"],
            pad_token=data["pad_token"],
            unk_token=data["unk_token"],
            start_token=data["start_token"],
            end_token=data["end_token"]
        )
        tokenizer.word2idx = data["word2idx"]
        tokenizer.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        return tokenizer
