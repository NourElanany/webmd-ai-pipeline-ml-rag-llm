# ============================================================
# nlp/tfidf.py — TF-IDF + LinearSVC/LR ensemble: train, evaluate, save, load
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

from webmd.config import (
    TFIDF_CHAR_VEC_PATH,
    TFIDF_LR_PATH,
    TFIDF_SVC_PATH,
    TFIDF_WORD_VEC_PATH,
)


@dataclass
class TfidfArtifacts:
    """All fitted TF-IDF objects needed for inference."""
    svc:      Any   # CalibratedClassifierCV(LinearSVC)
    lr:       Any   # LogisticRegression
    word_vec: TfidfVectorizer
    char_vec: TfidfVectorizer


@dataclass
class NlpResult:
    """Evaluation metrics for a single model or ensemble."""
    accuracy: float
    f1:       float
    report:   dict[str, Any]
    cm:       np.ndarray
    y_pred:   np.ndarray
    y_prob:   np.ndarray


def build_tfidf_model(
    train_texts: np.ndarray,
    y_train: np.ndarray,
    val_texts: np.ndarray,
    y_val: np.ndarray,
) -> TfidfArtifacts:
    """Fit word + char TF-IDF vectorisers and train SVC + LR classifiers.

    Architecture:
    - Word n-grams (1-3), 150k features
    - Char n-grams (2-4), 150k features
    - CalibratedClassifierCV(LinearSVC) — high precision on sparse features
    - LogisticRegression (SAGA) — better calibrated probabilities
    Both classifiers are ensembled at 50/50 for validation reporting.
    """
    print("\n-- Training TF-IDF models --")

    word_vec = TfidfVectorizer(
        ngram_range=(1, 3), max_features=150_000,
        sublinear_tf=True, min_df=2, analyzer="word",
        strip_accents="unicode", token_pattern=r"\b\w+\b",
    )
    char_vec = TfidfVectorizer(
        ngram_range=(2, 4), max_features=150_000,
        sublinear_tf=True, min_df=3, analyzer="char_wb",
    )

    X_tr = hstack([word_vec.fit_transform(train_texts), char_vec.fit_transform(train_texts)])
    X_vl = hstack([word_vec.transform(val_texts),       char_vec.transform(val_texts)])

    svc = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=3000), cv=3)
    svc.fit(X_tr, y_train)

    lr = LogisticRegression(C=5.0, max_iter=1000, solver="saga", n_jobs=-1)
    lr.fit(X_tr, y_train)

    val_prob = 0.5 * svc.predict_proba(X_vl)[:, 1] + 0.5 * lr.predict_proba(X_vl)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)
    print(
        f"TF-IDF Ensemble Val  Acc: {(val_pred == y_val).mean():.4f}  "
        f"F1: {f1_score(y_val, val_pred, average='weighted'):.4f}"
    )

    return TfidfArtifacts(svc=svc, lr=lr, word_vec=word_vec, char_vec=char_vec)


def evaluate_tfidf(
    artifacts: TfidfArtifacts,
    test_texts: np.ndarray,
    y_test: np.ndarray,
) -> NlpResult:
    """Evaluate the TF-IDF ensemble on the held-out test set."""
    X_te   = hstack([
        artifacts.word_vec.transform(test_texts),
        artifacts.char_vec.transform(test_texts),
    ])
    y_prob = (
        0.5 * artifacts.svc.predict_proba(X_te)[:, 1]
        + 0.5 * artifacts.lr.predict_proba(X_te)[:, 1]
    )
    y_pred = (y_prob >= 0.5).astype(int)

    acc    = float((y_pred == y_test).mean())
    f1     = float(f1_score(y_test, y_pred, average="weighted"))
    report = classification_report(
        y_test, y_pred, target_names=["Negative", "Positive"], output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n-- TF-IDF Test Results --")
    print(f"Test Accuracy: {acc:.4f}  |  F1 (weighted): {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    return NlpResult(accuracy=acc, f1=f1, report=report, cm=cm, y_pred=y_pred, y_prob=y_prob)


def save_tfidf(artifacts: TfidfArtifacts, out_dir: Path | None = None) -> None:
    """Persist all four TF-IDF artifacts to disk with joblib.

    Paths default to the values in config.py (ARTIFACTS_DIR).
    Pass *out_dir* to override the directory (used in tests / custom runs).
    """
    import joblib  # heavy import — kept local

    svc_path  = (out_dir / TFIDF_SVC_PATH.name)      if out_dir else TFIDF_SVC_PATH
    lr_path   = (out_dir / TFIDF_LR_PATH.name)       if out_dir else TFIDF_LR_PATH
    wv_path   = (out_dir / TFIDF_WORD_VEC_PATH.name) if out_dir else TFIDF_WORD_VEC_PATH
    cv_path   = (out_dir / TFIDF_CHAR_VEC_PATH.name) if out_dir else TFIDF_CHAR_VEC_PATH

    for path in (svc_path, lr_path, wv_path, cv_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifacts.svc,      svc_path)
    joblib.dump(artifacts.lr,       lr_path)
    joblib.dump(artifacts.word_vec, wv_path)
    joblib.dump(artifacts.char_vec, cv_path)
    print(f"TF-IDF artifacts saved to '{svc_path.parent}/'")


def load_tfidf(out_dir: Path | None = None) -> TfidfArtifacts:
    """Load all four TF-IDF artifacts from disk.

    Raises FileNotFoundError with a helpful message if any artifact is missing.
    """
    import joblib  # heavy import — kept local

    svc_path = (out_dir / TFIDF_SVC_PATH.name)      if out_dir else TFIDF_SVC_PATH
    lr_path  = (out_dir / TFIDF_LR_PATH.name)       if out_dir else TFIDF_LR_PATH
    wv_path  = (out_dir / TFIDF_WORD_VEC_PATH.name) if out_dir else TFIDF_WORD_VEC_PATH
    cv_path  = (out_dir / TFIDF_CHAR_VEC_PATH.name) if out_dir else TFIDF_CHAR_VEC_PATH

    for path in (svc_path, lr_path, wv_path, cv_path):
        if not path.exists():
            raise FileNotFoundError(
                f"TF-IDF artifact not found: {path}\n"
                "Run `uv run webmd-nlp` first to train and save it."
            )

    return TfidfArtifacts(
        svc=joblib.load(svc_path),
        lr=joblib.load(lr_path),
        word_vec=joblib.load(wv_path),
        char_vec=joblib.load(cv_path),
    )
