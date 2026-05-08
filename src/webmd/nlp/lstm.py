# ============================================================
# nlp/lstm.py — BiLSTM model: build, train, evaluate, save, load
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from webmd.config import (
    EMBED_DIM,
    EPOCHS,
    LSTM_MODEL_PATH,
    LSTM_TOKENIZER_PATH,
    MAX_LEN,
    VOCAB_SIZE,
)
from webmd.nlp.tfidf import NlpResult


def build_lstm_model() -> Any:
    """Construct the BiLSTM architecture.

    Architecture (identical to original dl_nlp.py):
        Embedding(30k, 128) → SpatialDropout(0.5) → BiLSTM(32)
        → Dense(16, ReLU) → Dropout(0.6) → Sigmoid output

    Heavy TF imports are kept inside the function to avoid loading the GPU
    runtime at CLI startup.
    """
    from tensorflow.keras.layers import (
        Bidirectional,
        Dense,
        Dropout,
        Embedding,
        LSTM,
        SpatialDropout1D,
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    model = Sequential([
        Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN, name="embedding"),
        SpatialDropout1D(0.5, name="spatial_dropout"),
        Bidirectional(LSTM(32, dropout=0.4, recurrent_dropout=0.4), name="bilstm"),
        Dense(16, activation="relu", kernel_regularizer=l2(1e-3), name="dense_1"),
        Dropout(0.6),
        Dense(1, activation="sigmoid", name="output", dtype="float32"),
    ])
    model.compile(optimizer=Adam(5e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _gpu_batch_size() -> int:
    """Return 1024 if a GPU is available, else 256."""
    import tensorflow as tf
    return 1024 if tf.config.list_physical_devices("GPU") else 256


def _configure_gpu() -> None:
    """Enable memory growth on all GPUs and activate mixed precision if available."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print(f"GPU detected: {[g.name for g in gpus]}  |  Mixed precision: ON")
    else:
        print("No GPU detected — running on CPU")


def train_lstm(
    model: Any,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Any, Any]:
    """Train the BiLSTM with EarlyStopping and ReduceLROnPlateau.

    Returns (trained_model, history).
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    _configure_gpu()
    print("\n-- Training BiLSTM (constrained) --")
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=2,
            restore_best_weights=True, verbose=1, min_delta=1e-3,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=_gpu_batch_size(),
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


def evaluate_lstm(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> NlpResult:
    """Evaluate the trained BiLSTM on the held-out test set."""
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc    = float((y_pred == y_test).mean())
    f1     = float(f1_score(y_test, y_pred, average="weighted"))
    report = classification_report(
        y_test, y_pred, target_names=["Negative", "Positive"], output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n-- BiLSTM Test Results --")
    print(f"Test Accuracy: {acc:.4f}  |  F1 (weighted): {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    return NlpResult(accuracy=acc, f1=f1, report=report, cm=cm, y_pred=y_pred, y_prob=y_prob)


def save_lstm(model: Any, tokenizer: Any, model_path: Path = LSTM_MODEL_PATH) -> None:
    """Save the Keras model and the fitted Keras Tokenizer to disk."""
    import joblib

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    joblib.dump(tokenizer, LSTM_TOKENIZER_PATH)
    print(f"BiLSTM model saved to '{model_path}'")
    print(f"Tokenizer saved to '{LSTM_TOKENIZER_PATH}'")


def load_lstm(model_path: Path = LSTM_MODEL_PATH) -> tuple[Any, Any]:
    """Load the Keras model and tokenizer from disk.

    Returns (model, tokenizer).
    Raises FileNotFoundError with a helpful message if either artifact is missing.
    """
    import joblib
    from tensorflow.keras.models import load_model

    for path in (model_path, LSTM_TOKENIZER_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"LSTM artifact not found: {path}\n"
                "Run `uv run webmd-nlp` first to train and save it."
            )

    model     = load_model(model_path, compile=False)
    tokenizer = joblib.load(LSTM_TOKENIZER_PATH)
    return model, tokenizer
