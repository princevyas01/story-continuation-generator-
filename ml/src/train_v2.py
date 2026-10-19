"""Training pipeline for Many-to-Many Subword Stacked LSTM (v2 Candidate)."""

import argparse
import json
import os
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import keras
import numpy as np
import tensorflow as tf

from ml.src.dataset_v2 import build_sequences_from_jsonl, create_tf_dataset
from ml.src.masked_loss import (
    MaskedSparseCategoricalCrossentropy,
    MaskedAccuracy,
    MaskedPerplexity,
)
from ml.src.model import (
    build_many_to_many_model,
    compile_model_v2,
    save_model_summary,
)
from ml.src.subword_tokenizer import SubwordStoryTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Story Continuation LSTM v2")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Context sequence length")
    parser.add_argument("--stride", type=int, default=96, help="Sequence window stride")
    parser.add_argument("--lr", type=float, default=0.0015, help="Initial learning rate")
    parser.add_argument("--embedding-dim", type=int, default=192, help="Embedding dimension")
    parser.add_argument("--lstm-units", type=int, default=256, help="LSTM hidden units")
    parser.add_argument("--dropout", type=float, default=0.20, help="Dropout rate")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch on tiny subset")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Starting Story Continuation Stacked LSTM v2 Training")
    print(f"Parameters: {vars(args)}")
    print("=" * 60)

    tokenizer_path = os.path.join("artifacts", "tokenizers", "tokenizer_v2.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. Run scripts/train_subword_tokenizer.py first."
        )

    print(f"Loading Subword Tokenizer from {tokenizer_path}...")
    tokenizer = SubwordStoryTokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer loaded. Vocab size: {vocab_size}")

    train_path = os.path.join("data", "processed", "v2", "train.jsonl")
    val_path = os.path.join("data", "processed", "v2", "val.jsonl")

    print(f"Building training sequences from {train_path} (seq_len={args.seq_len}, stride={args.stride})...")
    x_train, y_train = build_sequences_from_jsonl(
        train_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        stride=args.stride,
    )
    print(f"Building validation sequences from {val_path}...")
    x_val, y_val = build_sequences_from_jsonl(
        val_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        stride=args.seq_len,  # Non-overlapping for validation
    )

    if args.smoke_test:
        print("[SMOKE TEST] Truncating data to 128 train and 32 val samples...")
        x_train, y_train = x_train[:128], y_train[:128]
        x_val, y_val = x_val[:32], y_val[:32]
        args.epochs = 1

    print(f"Train samples: {len(x_train):,} sequences")
    print(f"Val samples:   {len(x_val):,} sequences")

    train_ds = create_tf_dataset(x_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_ds = create_tf_dataset(x_val, y_val, batch_size=args.batch_size, shuffle=False)

    print("\nInstantiating Stacked LSTM v2 Architecture...")
    model = build_many_to_many_model(
        vocab_size=vocab_size,
        seq_len=args.seq_len,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        num_layers=2,
        dropout_rate=args.dropout,
    )
    model = compile_model_v2(model, learning_rate=args.lr, clipnorm=1.0, pad_id=tokenizer.PAD_ID)
    model.summary()

    # Model save paths
    models_dir = os.path.join("artifacts", "models")
    best_dir = os.path.join("models", "best")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    model_best_artifact = os.path.join(models_dir, "story_lstm_v2.keras")
    model_best_repo = os.path.join(best_dir, "story_lstm_v2.keras")
    summary_path = os.path.join(models_dir, "model_summary_v2.txt")
    config_path = os.path.join(models_dir, "model_v2_config.json")

    save_model_summary(model, summary_path)

    config_data = {
        "architecture": "StackedLSTM-ManyToMany",
        "vocab_size": vocab_size,
        "embedding_dim": args.embedding_dim,
        "lstm_units": args.lstm_units,
        "num_layers": 2,
        "dropout_rate": args.dropout,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "pad_id": tokenizer.PAD_ID,
        "unk_id": tokenizer.UNK_ID,
        "bos_id": tokenizer.BOS_ID,
        "eos_id": tokenizer.EOS_ID,
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "clipnorm": 1.0,
        "loss": "MaskedSparseCategoricalCrossentropy",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_best_artifact,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    print("\nStarting Training...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )
    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time:.1f} seconds ({(total_training_time/60):.2f} minutes).")

    # Also sync best model to models/best/
    if os.path.exists(model_best_artifact):
        import shutil
        shutil.copy2(model_best_artifact, model_best_repo)
        print(f"Synced best model to {model_best_repo}")

    # Serialize history
    reports_dir = os.path.join("artifacts", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    history_path = os.path.join(reports_dir, "training_history_v2.json")
    
    # Convert numpy types in history to native Python floats
    clean_history = {}
    for k, v in history.history.items():
        clean_history[k] = [float(val) for val in v]
    clean_history["total_training_time_seconds"] = total_training_time
    clean_history["epochs_completed"] = len(history.history["loss"])

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(clean_history, f, indent=2)
    print(f"Saved training history to {history_path}")

    # Print best metrics
    best_epoch = int(np.argmin(clean_history["val_loss"]))
    print("\n" + "=" * 60)
    print(f"Best Validation Epoch: {best_epoch + 1}")
    print(f"  Train Loss:       {clean_history['loss'][best_epoch]:.4f}")
    print(f"  Train Accuracy:   {clean_history['accuracy'][best_epoch]:.4f}")
    print(f"  Train Perplexity: {clean_history['perplexity'][best_epoch]:.4f}")
    print(f"  Val Loss:         {clean_history['val_loss'][best_epoch]:.4f}")
    print(f"  Val Accuracy:     {clean_history['val_accuracy'][best_epoch]:.4f}")
    print(f"  Val Perplexity:   {clean_history['val_perplexity'][best_epoch]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
