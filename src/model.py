"""LSTM Language Model definition, compilation, and serialization."""

import os
from typing import Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(
    vocab_size: int,
    seq_len: int = 50,
    embedding_dim: int = 128,
    lstm_units: int = 256,
    num_layers: int = 2,
    dropout_rate: float = 0.2
) -> keras.Model:
    """Build a word-level LSTM autoregressive language model.
    
    Architecture:
      Input (seq_len,)
      -> Embedding(vocab_size, embedding_dim)
      -> LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)
      -> LSTM(lstm_units, return_sequences=False, dropout=dropout_rate)
      -> Dense(vocab_size, activation='softmax')
    """
    inputs = layers.Input(shape=(seq_len,), dtype="int32", name="token_input")
    
    # Embedding layer mapping discrete token IDs to dense vectors
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="word_embedding"
    )(inputs)

    # First LSTM layer (returns full sequence of hidden states)
    if num_layers > 1:
        x = layers.LSTM(
            units=lstm_units,
            return_sequences=True,
            dropout=dropout_rate,
            name="lstm_layer_1"
        )(x)
        
        # Second LSTM layer (returns final hidden state vector)
        x = layers.LSTM(
            units=lstm_units,
            return_sequences=False,
            dropout=dropout_rate,
            name="lstm_layer_2"
        )(x)
    else:
        x = layers.LSTM(
            units=lstm_units,
            return_sequences=False,
            dropout=dropout_rate,
            name="lstm_layer_1"
        )(x)

    # Dense projection head producing next-token probability distribution over vocabulary
    outputs = layers.Dense(
        units=vocab_size,
        activation="softmax",
        name="next_token_softmax"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="story_continuation_lstm")
    return model

def compile_model(
    model: keras.Model,
    learning_rate: float = 0.002,
    clipnorm: float = 1.0
) -> keras.Model:
    """Compile model with Adam optimizer and Sparse Categorical Crossentropy."""
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=clipnorm
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"]
    )
    return model

def get_model_summary_string(model: keras.Model) -> str:
    """Generate a clean string representation of the model architecture and parameters."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)

def save_model_summary(model: keras.Model, filepath: str):
    """Save model architecture summary to a text file artifact."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    summary_str = get_model_summary_string(model)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(summary_str)

def load_trained_model(filepath: str) -> keras.Model:
    """Load a trained Keras model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")
    return keras.models.load_model(filepath)
