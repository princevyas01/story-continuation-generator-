"""Subword Tokenizer using Byte-Level BPE for open-vocabulary story generation."""

import json
import os
from typing import List, Dict, Optional, Iterable, Union
from tokenizers import ByteLevelBPETokenizer, Tokenizer


class SubwordStoryTokenizer:
    """Byte-Level BPE subword tokenizer for robust vocabulary coverage and zero OOV rate.

    Special Token Mapping:
        <PAD>: 0
        <UNK>: 1
        <BOS>: 2
        <EOS>: 3
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, tokenizer: Optional[Tokenizer] = None, vocab_size: int = 8000):
        self._tokenizer = tokenizer
        self.target_vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is None:
            return 0
        return self._tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self.PAD_ID

    @property
    def unk_token_id(self) -> int:
        return self.UNK_ID

    @property
    def bos_token_id(self) -> int:
        return self.BOS_ID

    @property
    def eos_token_id(self) -> int:
        return self.EOS_ID

    @classmethod
    def train(
        cls,
        text_iterator: Iterable[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
    ) -> "SubwordStoryTokenizer":
        """Train a byte-level BPE tokenizer on an iterator of story texts."""
        bpe_trainer = ByteLevelBPETokenizer()
        bpe_trainer.train_from_iterator(
            text_iterator,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=cls.SPECIAL_TOKENS,
        )
        return cls(tokenizer=bpe_trainer._tokenizer, vocab_size=vocab_size)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs.

        If add_special_tokens is True, prepends BOS and appends EOS.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded.")

        encoding = self._tokenizer.encode(text)
        ids = list(encoding.ids)
        if add_special_tokens:
            ids = [self.BOS_ID] + ids + [self.EOS_ID]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to string."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not trained or loaded.")

        if skip_special_tokens:
            ids = [i for i in ids if i not in (self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID)]

        return self._tokenizer.decode(ids)

    def token_to_id(self, token: str) -> Optional[int]:
        if self._tokenizer is None:
            return None
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, idx: int) -> Optional[str]:
        if self._tokenizer is None:
            return None
        return self._tokenizer.id_to_token(idx)

    def save(self, filepath: str) -> None:
        """Save tokenizer configuration and model weights/vocab to file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        if self._tokenizer is None:
            raise RuntimeError("Cannot save uninitialized tokenizer.")

        self._tokenizer.save(filepath)

        # Also save sidecar metadata JSON
        meta_path = filepath.replace(".json", "_metadata.json")
        meta = {
            "type": "subword_bytelevel_bpe",
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "pad": {"token": self.PAD_TOKEN, "id": self.PAD_ID},
                "unk": {"token": self.UNK_TOKEN, "id": self.UNK_ID},
                "bos": {"token": self.BOS_TOKEN, "id": self.BOS_ID},
                "eos": {"token": self.EOS_TOKEN, "id": self.EOS_ID},
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "SubwordStoryTokenizer":
        """Load tokenizer from JSON artifact."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tokenizer file not found: {filepath}")

        raw_tokenizer = Tokenizer.from_file(filepath)
        return cls(tokenizer=raw_tokenizer, vocab_size=raw_tokenizer.get_vocab_size())

    def evaluate_unknown_rate(self, texts: Iterable[str]) -> float:
        """Calculate proportion of UNK tokens on given texts."""
        total_tokens = 0
        unk_tokens = 0
        for text in texts:
            ids = self.encode(text)
            total_tokens += len(ids)
            unk_tokens += sum(1 for i in ids if i == self.UNK_ID)
        return (unk_tokens / total_tokens) if total_tokens > 0 else 0.0
