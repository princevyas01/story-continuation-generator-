#!/usr/bin/env python3
"""End-to-end smoke test validating dataset, sequence creation, training, reload, and generation."""

import os
import sys
import shutil
import tempfile
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import normalize_text, tokenize_words
from src.tokenizer import StoryTokenizer
from src.sequences import build_dataset_arrays
from src.model import build_model, compile_model, load_trained_model
from src.generate import generate_continuation
from src.metrics import calculate_perplexity

def run_smoke_test():
    print("=" * 60)
    print("RUNNING END-TO-END SMOKE TEST")
    print("=" * 60)

    # 1. Create a tiny synthetic mini-corpus
    tiny_stories = [
        "Once upon a time in a deep forest, a brave little rabbit found a golden carrot. The rabbit was full of joy and shared it with his friends.",
        "There was once a wise old king who lived in a grand castle. Every morning he looked out of the window and greeted the sun with a smile.",
        "In a quaint cottage near the river, a young boy discovered a magical stone that sparkled in the dark night and brought good fortune."
    ]

    temp_dir = tempfile.mkdtemp(prefix="lstm_smoke_")
    try:
        # 2. Preprocess
        docs_tokens = []
        for s in tiny_stories:
            cleaned = normalize_text(s)
            tokens = tokenize_words(cleaned)
            docs_tokens.append(tokens)

        # 3. Fit tokenizer
        tokenizer = StoryTokenizer(max_vocab_size=200)
        tokenizer.fit_on_texts(docs_tokens)
        vocab_file = os.path.join(temp_dir, "vocab.json")
        tokenizer.save(vocab_file)
        reloaded_tok = StoryTokenizer.load(vocab_file)
        assert reloaded_tok.vocab_size == tokenizer.vocab_size, "Tokenizer reload size mismatch"
        print(f"[PASS] Tokenizer built and verified. Vocab size: {reloaded_tok.vocab_size}")

        # 4. Create sequences
        seq_len = 10
        X, y = build_dataset_arrays(docs_tokens, reloaded_tok, seq_len=seq_len, step=1)
        assert len(X) > 0, "No sequences created"
        assert X.shape[1] == seq_len, f"Sequence length mismatch: {X.shape}"
        print(f"[PASS] Sequence creation verified: {len(X)} sequences of shape {X.shape}")

        # 5. Build and compile model
        model = build_model(
            vocab_size=reloaded_tok.vocab_size,
            seq_len=seq_len,
            embedding_dim=32,
            lstm_units=64,
            num_layers=1,
            dropout_rate=0.0
        )
        compile_model(model, learning_rate=0.01)
        print("[PASS] Model build and compilation verified.")

        # 6. Train for 2 quick epochs
        history = model.fit(X, y, epochs=2, batch_size=8, verbose=0)
        loss = history.history["loss"][-1]
        ppl = calculate_perplexity(loss)
        print(f"[PASS] Model training step verified. Loss: {loss:.4f}, Perplexity: {ppl:.2f}")

        # 7. Save and reload model
        model_file = os.path.join(temp_dir, "smoke_model.keras")
        model.save(model_file)
        reloaded_model = load_trained_model(model_file)
        print("[PASS] Model save and reload verified.")

        # 8. Generate continuation
        result = generate_continuation(
            model=reloaded_model,
            tokenizer=reloaded_tok,
            prompt="once upon a time",
            seq_len=seq_len,
            max_new_tokens=15,
            temperature=0.8,
            top_k=5,
            top_p=0.9
        )
        assert result["tokens_generated"] > 0, "No tokens generated"
        print(f"[PASS] Autoregressive generation verified. Prompt: '{result['prompt']}' -> Story: '{result['full_story']}'")
        print("=" * 60)
        print("END-TO-END SMOKE TEST PASSED SUCCESSFULLY!")
        print("=" * 60)
        return True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
