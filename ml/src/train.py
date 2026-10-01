#!/usr/bin/env python3
"""Training pipeline for Story Continuation LSTM model."""

import os
import sys
import time
import json
import argparse
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.src.utils import load_config, resolve_path, set_seed, calculate_perplexity
from ml.src.tokenizer import StoryTokenizer
from ml.src.build_sequences import build_dataset_arrays
from ml.src.model import build_model, compile_model, save_model_summary

def plot_and_save_loss_curve(history: keras.callbacks.History, save_path: str):
    """Plot training and validation loss curves and save to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    epochs_range = range(1, len(history.history["loss"]) + 1)
    
    plt.figure(figsize=(9, 5))
    plt.plot(epochs_range, history.history["loss"], "b-o", label="Training Loss")
    if "val_loss" in history.history:
        plt.plot(epochs_range, history.history["val_loss"], "r--s", label="Validation Loss")
        
    plt.title("LSTM Language Model: Training & Validation Loss Curve", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Cross-Entropy Loss", fontsize=11)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def train_lstm(
    config_path: str = "ml/configs/base.yaml",
    smoke_mode: bool = False,
    override_epochs: int = None,
    override_batch_size: int = None
):
    print("=" * 60)
    mode_str = "SMOKE TRAINING RUN" if smoke_mode else "FULL TRAINING RUN"
    print(f"STARTING {mode_str}")
    print("=" * 60)

    cfg = load_config(config_path)
    bdir = cfg["base_dir"]
    seed = cfg["project"]["random_seed"]
    set_seed(seed)

    start_train_time = time.time()

    manifest_path = resolve_path(bdir, cfg["data"]["manifest_path"])
    vocab_path = resolve_path(bdir, cfg["tokenization"]["vocab_path"])
    legacy_vocab = resolve_path(bdir, cfg["tokenization"].get("legacy_vocab_path", "artifacts/tokenizer_vocab.json"))
    
    # Load tokenizer
    if os.path.exists(vocab_path):
        tokenizer = StoryTokenizer.load(vocab_path)
    elif os.path.exists(legacy_vocab):
        tokenizer = StoryTokenizer.load(legacy_vocab)
    else:
        raise FileNotFoundError(f"Tokenizer not found at {vocab_path} or {legacy_vocab}")

    # Output paths
    best_model_path = resolve_path(bdir, cfg["training"]["best_model_path"])
    final_model_path = resolve_path(bdir, cfg["training"]["final_model_path"])
    loss_curve_path = resolve_path(bdir, cfg["artifacts"]["loss_curve_png"])
    metrics_json_path = resolve_path(bdir, cfg["artifacts"]["metrics_json"])
    metrics_csv_path = resolve_path(bdir, cfg["artifacts"]["metrics_csv"])
    summary_path = resolve_path(bdir, cfg["artifacts"]["model_summary_txt"])

    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_json_path), exist_ok=True)

    seq_len = cfg["tokenization"]["sequence_length"]
    emb_dim = cfg["model"]["embedding_dim"]
    lstm_units = cfg["model"]["lstm_units"]
    dropout = cfg["model"]["dropout_rate"]
    num_layers = cfg["model"]["num_layers"]
    lr = cfg["model"]["learning_rate"]
    clipnorm = cfg["model"]["clipnorm"]
    batch_size = override_batch_size or cfg["training"]["batch_size"]
    epochs = override_epochs or (cfg["training"]["smoke_epochs"] if smoke_mode else cfg["training"]["epochs"])

    print(f"Building LSTM Model (Vocab: {tokenizer.vocab_size}, SeqLen: {seq_len}, Units: {lstm_units})...")
    model = build_model(
        vocab_size=tokenizer.vocab_size,
        seq_len=seq_len,
        embedding_dim=emb_dim,
        lstm_units=lstm_units,
        num_layers=num_layers,
        dropout_rate=dropout
    )
    compile_model(model, learning_rate=lr, clipnorm=clipnorm)
    save_model_summary(model, summary_path)

    # Check for processed data
    train_x_path = resolve_path(bdir, "data/processed/train_x.npy")
    train_y_path = resolve_path(bdir, "data/processed/train_y.npy")
    val_x_path = resolve_path(bdir, "data/processed/val_x.npy")
    val_y_path = resolve_path(bdir, "data/processed/val_y.npy")

    if os.path.exists(train_x_path) and os.path.exists(train_y_path):
        print("Loading precomputed sequences from data/processed/...")
        x_train = np.load(train_x_path)
        y_train = np.load(train_y_path)
        x_val = np.load(val_x_path) if os.path.exists(val_x_path) else None
        y_val = np.load(val_y_path) if os.path.exists(val_y_path) else None
    else:
        print("Precomputed sequences not found; generating from raw stories...")
        from ml.src.prepare_data import load_and_preprocess_directory, split_documents
        docs = load_and_preprocess_directory(resolve_path(bdir, "data/raw"))
        train_docs, val_docs, _ = split_documents(docs, seed=seed)
        x_train, y_train = build_dataset_arrays([d["tokens"] for d in train_docs], tokenizer, seq_len=seq_len)
        x_val, y_val = build_dataset_arrays([d["tokens"] for d in val_docs], tokenizer, seq_len=seq_len)

    if smoke_mode:
        x_train = x_train[:500]
        y_train = y_train[:500]
        if x_val is not None:
            x_val = x_val[:100]
            y_val = y_val[:100]

    val_data = (x_val, y_val) if x_val is not None and len(x_val) > 0 else None

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss" if val_data else "loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss" if val_data else "loss",
            patience=cfg["training"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if val_data else "loss",
            factor=cfg["training"]["reduce_lr_factor"],
            patience=cfg["training"]["reduce_lr_patience"],
            min_lr=cfg["training"]["min_lr"],
            verbose=1
        )
    ]

    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=val_data,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    model.save(final_model_path)
    plot_and_save_loss_curve(history, loss_curve_path)

    elapsed = time.time() - start_train_time
    best_loss = min(history.history["loss"])
    best_val_loss = min(history.history["val_loss"]) if val_data else None
    val_ppl = calculate_perplexity(best_val_loss) if best_val_loss else calculate_perplexity(best_loss)

    metrics = {
        "training_duration_seconds": round(elapsed, 2),
        "epochs_completed": len(history.history["loss"]),
        "final_loss": round(float(history.history["loss"][-1]), 4),
        "best_train_loss": round(float(best_loss), 4),
        "best_val_loss": round(float(best_val_loss), 4) if best_val_loss else None,
        "perplexity": round(float(val_ppl), 4),
        "train_samples": len(x_train),
        "val_samples": len(x_val) if x_val is not None else 0,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 60)
    print(f"TRAINING COMPLETE in {elapsed:.1f}s | Best Val Loss: {best_val_loss} | PPL: {val_ppl:.2f}")
    print("=" * 60)
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()
    train_lstm(smoke_mode=args.smoke, override_epochs=args.epochs, override_batch_size=args.batch_size)
