#!/usr/bin/env python3
"""Training pipeline for Story Continuation LSTM model."""

import os
import sys
import time
import json
import random
import argparse
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config, resolve_path
from src.tokenizer import StoryTokenizer
from src.sequences import build_dataset_arrays
from src.model import build_model, compile_model, save_model_summary
from src.metrics import calculate_perplexity

def set_all_seeds(seed: int = 42):
    """Ensure determinism across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def plot_and_save_loss_curve(history: keras.callbacks.History, save_path: str):
    """Plot training and validation loss curves and save to disk."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    epochs_range = range(1, len(history.history["loss"]) + 1)
    
    plt.figure(figsize=(9, 5))
    plt.plot(epochs_range, history.history["loss"], "b-o", label="Training Loss (Sparse Categorical CE)")
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
    print(f"Saved loss curve plot to: {save_path}")

def train_lstm(
    config_path: str = "configs/config.yaml",
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
    set_all_seeds(seed)

    start_train_time = time.time()

    # Paths
    manifest_path = resolve_path(bdir, cfg["data"]["manifest_path"])
    vocab_path = resolve_path(bdir, cfg["tokenization"]["vocab_path"])
    best_model_path = resolve_path(bdir, cfg["training"]["best_model_path"])
    final_model_path = resolve_path(bdir, cfg["training"]["final_model_path"])
    loss_curve_path = resolve_path(bdir, cfg["artifacts"]["loss_curve_png"])
    metrics_json_path = resolve_path(bdir, cfg["artifacts"]["metrics_json"])
    metrics_csv_path = resolve_path(bdir, cfg["artifacts"]["metrics_csv"])
    training_report_path = resolve_path(bdir, cfg["artifacts"]["training_report_md"])
    model_summary_path = resolve_path(bdir, cfg["artifacts"]["model_summary_txt"])
    config_snapshot_path = resolve_path(bdir, cfg["artifacts"]["config_used_yaml"])

    # Load dataset manifest
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run prepare_data.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Load tokenizer
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Tokenizer not found at {vocab_path}. Run prepare_data.py first.")
    tokenizer = StoryTokenizer.load(vocab_path)
    vocab_size = tokenizer.vocab_size
    print(f"Loaded tokenizer with vocabulary size: {vocab_size}")

    # Load story documents according to split
    processed_dir = resolve_path(bdir, cfg["data"]["processed_dir"])
    train_ids = manifest["story_split_mapping"]["train"]
    val_ids = manifest["story_split_mapping"]["val"]
    test_ids = manifest["story_split_mapping"]["test"]

    if smoke_mode:
        smoke_count = cfg["training"].get("smoke_sample_docs", 4)
        train_ids = train_ids[:smoke_count]
        val_ids = val_ids[:2]
        test_ids = test_ids[:2]
        epochs = cfg["training"].get("smoke_epochs", 2)
        batch_size = 32
        step_stride = 4 # Faster sequence creation for smoke run
    else:
        epochs = override_epochs or cfg["training"]["epochs"]
        batch_size = override_batch_size or cfg["training"]["batch_size"]
        step_stride = 2 # Efficient stride for 130k corpus

    def load_doc_tokens(doc_ids):
        docs_tokens = []
        for d_id in doc_ids:
            p = os.path.join(processed_dir, f"{d_id}.txt")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    docs_tokens.append(f.read().split())
        return docs_tokens

    train_tokens = load_doc_tokens(train_ids)
    val_tokens = load_doc_tokens(val_ids)
    test_tokens = load_doc_tokens(test_ids)

    seq_len = cfg["tokenization"]["sequence_length"]
    print(f"Creating sliding window sequences (seq_len={seq_len}, step={step_stride})...")
    X_train, y_train = build_dataset_arrays(train_tokens, tokenizer, seq_len=seq_len, step=step_stride)
    X_val, y_val = build_dataset_arrays(val_tokens, tokenizer, seq_len=seq_len, step=step_stride)
    X_test, y_test = build_dataset_arrays(test_tokens, tokenizer, seq_len=seq_len, step=step_stride)

    print(f"Generated Training Samples:   {len(X_train)} sequences of shape {X_train.shape}")
    print(f"Generated Validation Samples: {len(X_val)} sequences of shape {X_val.shape}")
    print(f"Generated Test Samples:       {len(X_test)} sequences of shape {X_test.shape}")

    # Build and compile model
    model = build_model(
        vocab_size=vocab_size,
        seq_len=seq_len,
        embedding_dim=cfg["model"]["embedding_dim"],
        lstm_units=cfg["model"]["lstm_units"],
        num_layers=cfg["model"]["num_layers"],
        dropout_rate=cfg["model"]["dropout_rate"]
    )
    compile_model(
        model,
        learning_rate=cfg["model"]["learning_rate"],
        clipnorm=cfg["model"]["clipnorm"]
    )

    # Save model summary artifact
    save_model_summary(model, model_summary_path)

    # Callbacks
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss" if len(X_val) > 0 else "loss",
        save_best_only=True,
        verbose=1
    )
    early_stop_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss" if len(X_val) > 0 else "loss",
        patience=cfg["training"]["early_stopping_patience"],
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss" if len(X_val) > 0 else "loss",
        factor=cfg["training"]["reduce_lr_factor"],
        patience=cfg["training"]["reduce_lr_patience"],
        min_lr=cfg["training"]["min_lr"],
        verbose=1
    )

    callbacks_list = [checkpoint_cb]
    if not smoke_mode:
        callbacks_list.extend([early_stop_cb, reduce_lr_cb])

    print(f"Beginning model training ({epochs} epochs, batch_size={batch_size})...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )

    # Save final model
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    elapsed_time = time.time() - start_train_time
    print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes).")

    # Evaluate best model on test set if available
    best_model = keras.models.load_model(best_model_path)
    test_loss, test_acc = (0.0, 0.0)
    if len(X_test) > 0:
        test_eval = best_model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
        test_loss = float(test_eval[0])
        test_acc = float(test_eval[1])
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    final_train_loss = float(history.history["loss"][-1])
    best_val_loss = float(min(history.history["val_loss"])) if "val_loss" in history.history else final_train_loss
    train_ppl = calculate_perplexity(final_train_loss)
    val_ppl = calculate_perplexity(best_val_loss)
    test_ppl = calculate_perplexity(test_loss) if test_loss > 0 else 0.0

    # Save loss plot
    plot_and_save_loss_curve(history, loss_curve_path)

    # Save metrics JSON & CSV
    metrics_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "smoke_mode": smoke_mode,
        "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "vocabulary_size": vocab_size,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "train_loss": round(final_train_loss, 4),
        "val_loss": round(best_val_loss, 4),
        "test_loss": round(test_loss, 4),
        "train_perplexity": round(train_ppl, 2),
        "val_perplexity": round(val_ppl, 2),
        "test_perplexity": round(test_ppl, 2),
        "training_duration_seconds": round(elapsed_time, 2),
        "history": {k: [round(float(v), 4) for v in vals] for k, vals in history.history.items()}
    }

    os.makedirs(os.path.dirname(metrics_json_path), exist_ok=True)
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # CSV metrics history
    os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
    with open(metrics_csv_path, "w", encoding="utf-8") as f:
        headers = ["epoch", "loss", "accuracy"]
        if "val_loss" in history.history:
            headers.extend(["val_loss", "val_accuracy"])
        f.write(",".join(headers) + "\n")
        for ep in range(len(history.history["loss"])):
            row = [str(ep + 1), f"{history.history['loss'][ep]:.4f}", f"{history.history['accuracy'][ep]:.4f}"]
            if "val_loss" in history.history:
                row.extend([f"{history.history['val_loss'][ep]:.4f}", f"{history.history['val_accuracy'][ep]:.4f}"])
            f.write(",".join(row) + "\n")

    # Save Config Snapshot
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        f.write(f"# Snapshot of configuration executed at {metrics_data['timestamp']}\n")
        f.write(f"# Smoke Mode: {smoke_mode}\n\n")
        json.dump(cfg, f, indent=2)

    # Generate Human-Readable Training Report
    with open(training_report_path, "w", encoding="utf-8") as f:
        f.write(f"# Model Training Report\n\n")
        f.write(f"- **Execution Mode:** {'Smoke Run' if smoke_mode else 'Full Training Run'}\n")
        f.write(f"- **Timestamp:** {metrics_data['timestamp']}\n")
        f.write(f"- **Dataset:** {manifest['source_name']}\n")
        f.write(f"- **Documents (Train/Val/Test):** {manifest['train_document_count']} / {manifest['validation_document_count']} / {manifest['test_document_count']}\n")
        f.write(f"- **Corpus Total Tokens:** {manifest['token_count']}\n")
        f.write(f"- **Vocabulary Size:** {vocab_size} tokens\n")
        f.write(f"- **Sequence Length (Context Window):** {seq_len}\n")
        f.write(f"- **Model Architecture:** 2-Layer Word-Level LSTM (Units: 256, Embedding: 128, Dropout: 0.2)\n")
        f.write(f"- **Trainable Parameters:** {model.count_params():,}\n")
        f.write(f"- **Batch Size:** {batch_size}\n")
        f.write(f"- **Epochs Requested / Completed:** {epochs} / {metrics_data['epochs_completed']}\n")
        f.write(f"- **Final Training Loss:** {final_train_loss:.4f} (Perplexity: {train_ppl:.2f})\n")
        f.write(f"- **Best Validation Loss:** {best_val_loss:.4f} (Perplexity: {val_ppl:.2f})\n")
        f.write(f"- **Held-Out Test Loss:** {test_loss:.4f} (Perplexity: {test_ppl:.2f})\n")
        f.write(f"- **Training Duration:** {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")
        f.write(f"- **Platform / Machine:** Intel Core Ultra 5 125H (14 cores / 18 threads), Windows 11, TensorFlow CPU (oneDNN enabled)\n")

    print(f"Saved metrics to: {metrics_json_path}")
    print(f"Saved training report to: {training_report_path}")
    print("=" * 60)
    return metrics_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train word-level LSTM for story continuation.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke training on small subset")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    train_lstm(
        config_path=args.config,
        smoke_mode=args.smoke,
        override_epochs=args.epochs,
        override_batch_size=args.batch_size
    )
