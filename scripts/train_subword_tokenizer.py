"""Train and evaluate the Subword Byte-Level BPE Tokenizer on train.jsonl only."""

import json
import os
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from ml.src.subword_tokenizer import SubwordStoryTokenizer


def text_stream(jsonl_path: str):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                yield data["text"]


def main():
    train_path = os.path.join("data", "processed", "v2", "train.jsonl")
    val_path = os.path.join("data", "processed", "v2", "val.jsonl")
    test_path = os.path.join("data", "processed", "v2", "test.jsonl")

    output_dir = os.path.join("artifacts", "tokenizers")
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer_v2.json")

    print(f"Training Byte-Level BPE Tokenizer on {train_path}...")
    tokenizer = SubwordStoryTokenizer.train(
        text_iterator=text_stream(train_path),
        vocab_size=8000,
        min_frequency=2,
    )

    actual_vocab_size = tokenizer.vocab_size
    print(f"Training complete. Actual vocabulary size: {actual_vocab_size}")

    # Verify special tokens
    assert tokenizer.pad_token_id == 0, f"Expected PAD ID 0, got {tokenizer.pad_token_id}"
    assert tokenizer.unk_token_id == 1, f"Expected UNK ID 1, got {tokenizer.unk_token_id}"
    assert tokenizer.bos_token_id == 2, f"Expected BOS ID 2, got {tokenizer.bos_token_id}"
    assert tokenizer.eos_token_id == 3, f"Expected EOS ID 3, got {tokenizer.eos_token_id}"

    # Verify token lookup
    assert tokenizer.token_to_id("<PAD>") == 0
    assert tokenizer.token_to_id("<UNK>") == 1
    assert tokenizer.token_to_id("<BOS>") == 2
    assert tokenizer.token_to_id("<EOS>") == 3

    print("Special tokens verified: PAD=0, UNK=1, BOS=2, EOS=3")

    # Roundtrip test
    sample_texts = [
        "Once upon a time, a brave girl named Maya explored the enchanted forest.",
        "\"Look at that sparkling butterfly!\" she whispered with excitement.",
        "Don't worry, even rare names like Xylander and Zhorian won't break byte-level BPE.",
    ]

    print("\n--- Roundtrip Encoding/Decoding Verification ---")
    for s in sample_texts:
        encoded = tokenizer.encode(s)
        decoded = tokenizer.decode(encoded)
        print(f"Original: {s}")
        print(f"Tokens count: {len(encoded)} | Decoded: {decoded}")
        assert s == decoded, f"Mismatch in roundtrip!\nExpected: {s}\nGot: {decoded}"

    # Evaluate UNK rate on Val and Test
    val_unk_rate = tokenizer.evaluate_unknown_rate(text_stream(val_path))
    test_unk_rate = tokenizer.evaluate_unknown_rate(text_stream(test_path))

    print(f"\nValidation set UNK rate: {val_unk_rate:.6f}")
    print(f"Test set UNK rate: {test_unk_rate:.6f}")

    # Save tokenizer
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")

    # Also verify loading
    loaded_tok = SubwordStoryTokenizer.load(tokenizer_path)
    assert loaded_tok.vocab_size == tokenizer.vocab_size
    assert loaded_tok.decode(loaded_tok.encode("Testing load")) == "Testing load"
    print("Tokenizer load verification succeeded!")

    # Save stats
    stats = {
        "model_type": "Byte-Level BPE",
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        },
        "train_source": train_path,
        "val_unk_rate": val_unk_rate,
        "test_unk_rate": test_unk_rate,
    }
    stats_path = os.path.join(output_dir, "tokenizer_v2_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
