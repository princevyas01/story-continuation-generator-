"""LSTM Language Model definition, compilation, and serialization.

Supports:
- v1: Word-level many-to-one architecture (legacy baseline).
- v2: Subword-level many-to-many autoregressive architecture with masked loss.
"""

import os
from typing import Optional
import keras
from keras import layers
import tensorflow as tf

from ml.src.masked_loss import (
    MaskedSparseCategoricalCrossentropy,
    MaskedAccuracy,
    MaskedPerplexity,
)


def build_model(
    vocab_size: int,
    seq_len: int = 50,
    embedding_dim: int = 128,
    lstm_units: int = 256,
    num_layers: int = 2,
    dropout_rate: float = 0.20,
) -> keras.Model:
    """Build a word-level LSTM many-to-one language model (v1 Baseline)."""
    inputs = layers.Input(shape=(seq_len,), dtype="int32", name="token_input")

    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="word_embedding",
    )(inputs)

    if num_layers > 1:
        x = layers.LSTM(
            units=lstm_units,
            return_sequences=True,
            dropout=dropout_rate,
            name="lstm_layer_1",
        )(x)

        x = layers.LSTM(
            units=lstm_units,
            return_sequences=False,
            dropout=dropout_rate,
            name="lstm_layer_2",
        )(x)
    else:
        x = layers.LSTM(
            units=lstm_units,
            return_sequences=False,
            dropout=dropout_rate,
            name="lstm_layer_1",
        )(x)

    outputs = layers.Dense(
        units=vocab_size,
        activation="softmax",
        name="next_token_softmax",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="story_continuation_lstm_v1")
    return model


def build_many_to_many_model(
    vocab_size: int = 8000,
    seq_len: Optional[int] = None,
    embedding_dim: int = 192,
    lstm_units: int = 256,
    num_layers: int = 2,
    dropout_rate: float = 0.20,
) -> keras.Model:
    """Build a subword-level stacked LSTM many-to-many language model (v2 Candidate).

    Every timestep outputs linear logits for next-token prediction:
        Input: (batch, time)
        Output: (batch, time, vocab_size)
    """
    inputs = layers.Input(shape=(seq_len,), dtype="int32", name="token_input")

    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="subword_embedding",
    )(inputs)

    for i in range(num_layers):
        x = layers.LSTM(
            units=lstm_units,
            return_sequences=True,
            dropout=dropout_rate,
            name=f"lstm_layer_{i+1}",
        )(x)

    # Output logits at every timestep (many-to-many)
    outputs = layers.Dense(
        units=vocab_size,
        activation=None,  # from_logits=True for numerical stability
        name="token_logits",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="story_continuation_lstm_v2")
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.002,
    clipnorm: float = 1.0,
) -> keras.Model:
    """Compile v1 baseline model with Adam optimizer and unmasked crossentropy."""
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=clipnorm,
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def compile_model_v2(
    model: keras.Model,
    learning_rate: float = 0.0015,
    clipnorm: float = 1.0,
    pad_id: int = 0,
) -> keras.Model:
    """Compile v2 candidate model with masked loss and masked evaluation metrics."""
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=clipnorm,
    )
    loss = MaskedSparseCategoricalCrossentropy(from_logits=True, pad_id=pad_id)
    metrics = [
        MaskedAccuracy(pad_id=pad_id, name="accuracy"),
        MaskedPerplexity(from_logits=True, pad_id=pad_id, name="perplexity"),
    ]
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    return model


def get_model_summary_string(model: keras.Model) -> str:
    """Generate a clean string representation of the model architecture and parameters."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


def save_model_summary(model: keras.Model, filepath: str) -> None:
    """Save model architecture summary to a text file artifact."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    summary_str = get_model_summary_string(model)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(summary_str)


def load_trained_model(filepath: Optional[str] = None) -> keras.Model:
    """Load a trained Keras model with v2 preference and v1 fallback paths."""
    candidates = []
    if filepath:
        candidates.append(filepath)

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates.extend([
        # Preferred v2 candidate
        os.path.join(base, "artifacts", "models", "story_lstm_v2.keras"),
        os.path.join(base, "models", "best", "story_lstm_v2.keras"),
        # Fallback to v1 baseline
        os.path.join(base, "artifacts", "models", "story_lstm.keras"),
        os.path.join(base, "models", "best", "story_lstm.keras"),
        os.path.join(base, "models", "final", "story_lstm_final.keras"),
    ])

    for path in candidates:
        if os.path.exists(path):
            return keras.models.load_model(path)

    raise FileNotFoundError(f"No trained model found in candidates: {candidates}")
