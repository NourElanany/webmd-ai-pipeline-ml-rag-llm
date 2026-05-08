# ============================================================
# nlp/ensemble.py — TF-IDF + BiLSTM weighted ensemble
# ============================================================

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import hstack

from webmd.config import MAX_LEN, TFIDF_LSTM_WEIGHT
from webmd.nlp.tfidf import NlpResult, TfidfArtifacts


def evaluate_ensemble(
    lstm_probs: np.ndarray,
    tfidf_probs: np.ndarray,
    y_test: np.ndarray,
) -> NlpResult:
    """Combine LSTM and TF-IDF probability arrays into a weighted ensemble.

    Weights: LSTM = TFIDF_LSTM_WEIGHT (0.20), TF-IDF = 1 - TFIDF_LSTM_WEIGHT (0.80).
    These are defined in config.py as the single source of truth.
    """
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    y_prob = TFIDF_LSTM_WEIGHT * lstm_probs + (1 - TFIDF_LSTM_WEIGHT) * tfidf_probs
    y_pred = (y_prob >= 0.5).astype(int)

    acc    = float((y_pred == y_test).mean())
    f1     = float(f1_score(y_test, y_pred, average="weighted"))
    report = classification_report(
        y_test, y_pred, target_names=["Negative", "Positive"], output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    lstm_pct  = int(TFIDF_LSTM_WEIGHT * 100)
    tfidf_pct = 100 - lstm_pct
    print(f"\n-- Ensemble (LSTM {lstm_pct}% + TF-IDF {tfidf_pct}%) --")
    print(f"Test Accuracy: {acc:.4f}  |  F1 (weighted): {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    return NlpResult(accuracy=acc, f1=f1, report=report, cm=cm, y_pred=y_pred, y_prob=y_prob)


def predict_ensemble(
    text: str,
    tfidf_artifacts: TfidfArtifacts,
    lstm_model: Any,
    tokenizer: Any,
) -> tuple[float, float, float]:
    """Run a single review through the full ensemble pipeline.

    Returns (ensemble_prob, tfidf_prob, lstm_prob) — all in [0, 1].
    Values >= 0.5 indicate Positive sentiment.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # TF-IDF branch
    X_te = hstack([
        tfidf_artifacts.word_vec.transform([text]),
        tfidf_artifacts.char_vec.transform([text]),
    ])
    tfidf_prob = float(
        0.5 * tfidf_artifacts.svc.predict_proba(X_te)[0][1]
        + 0.5 * tfidf_artifacts.lr.predict_proba(X_te)[0][1]
    )

    # LSTM branch
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    lstm_prob = float(lstm_model.predict(padded, verbose=0)[0][0])

    # Weighted ensemble
    ensemble_prob = TFIDF_LSTM_WEIGHT * lstm_prob + (1 - TFIDF_LSTM_WEIGHT) * tfidf_prob

    return ensemble_prob, tfidf_prob, lstm_prob
