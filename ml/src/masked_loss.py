"""Masked Loss and Metrics for Many-to-Many Language Modeling.

Ignores PAD tokens (id=0) so padded sequence positions do not affect gradients or evaluation.
"""

import keras
from keras import ops
import tensorflow as tf


@keras.saving.register_keras_serializable(package="story_generator")
class MaskedSparseCategoricalCrossentropy(keras.losses.Loss):
    """Sparse Categorical Crossentropy that completely ignores PAD tokens."""

    def __init__(
        self,
        from_logits: bool = True,
        pad_id: int = 0,
        name: str = "masked_sparse_categorical_crossentropy",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.pad_id = pad_id

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        # Mask where target is not PAD
        mask = tf.not_equal(y_true, self.pad_id)
        
        # Compute element-wise crossentropy
        raw_loss = keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )
        mask_f = tf.cast(mask, raw_loss.dtype)
        
        # Sum over masked positions and divide by number of active tokens
        loss_sum = tf.reduce_sum(raw_loss * mask_f)
        token_count = tf.reduce_sum(mask_f)
        return loss_sum / (token_count + 1e-8)

    def get_config(self):
        config = super().get_config()
        config.update({
            "from_logits": self.from_logits,
            "pad_id": self.pad_id,
        })
        return config


@keras.saving.register_keras_serializable(package="story_generator")
class MaskedAccuracy(keras.metrics.Metric):
    """Token-level accuracy computed exclusively over non-pad positions."""

    def __init__(self, pad_id: int = 0, name: str = "masked_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pad_id = pad_id
        self.total_correct = self.add_weight(name="total_correct", initializer="zeros")
        self.total_tokens = self.add_weight(name="total_tokens", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.not_equal(y_true, self.pad_id)
        
        pred_ids = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        matches = tf.logical_and(tf.equal(y_true, pred_ids), mask)
        
        matches_f = tf.cast(matches, tf.float32)
        mask_f = tf.cast(mask, tf.float32)
        
        self.total_correct.assign_add(tf.reduce_sum(matches_f))
        self.total_tokens.assign_add(tf.reduce_sum(mask_f))

    def result(self):
        return tf.math.divide_no_nan(self.total_correct, self.total_tokens)

    def reset_state(self):
        self.total_correct.assign(0.0)
        self.total_tokens.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"pad_id": self.pad_id})
        return config


@keras.saving.register_keras_serializable(package="story_generator")
class MaskedPerplexity(keras.metrics.Metric):
    """Perplexity computed as exp(cross_entropy) over non-pad tokens."""

    def __init__(self, from_logits: bool = True, pad_id: int = 0, name: str = "masked_perplexity", **kwargs):
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.pad_id = pad_id
        self.total_loss = self.add_weight(name="total_loss", initializer="zeros")
        self.total_tokens = self.add_weight(name="total_tokens", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.not_equal(y_true, self.pad_id)
        raw_loss = keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )
        mask_f = tf.cast(mask, raw_loss.dtype)
        
        self.total_loss.assign_add(tf.reduce_sum(raw_loss * mask_f))
        self.total_tokens.assign_add(tf.reduce_sum(mask_f))

    def result(self):
        mean_cross_entropy = tf.math.divide_no_nan(self.total_loss, self.total_tokens)
        return tf.math.exp(mean_cross_entropy)

    def reset_state(self):
        self.total_loss.assign(0.0)
        self.total_tokens.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "from_logits": self.from_logits,
            "pad_id": self.pad_id,
        })
        return config
