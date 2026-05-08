# ============================================================
# nlp/preprocess.py — Text cleaning, data loading, train/val/test split
# ============================================================

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from webmd.config import MAX_LEN, ML_RANDOM_STATE, NLP_SAMPLE_SIZE, VOCAB_SIZE


@dataclass
class NlpSplits:
    """All arrays produced by a single unified split.

    Sequences (X_seq_*) are used by the BiLSTM.
    Texts (X_txt_*) are used by TF-IDF.
    Both sets index the same rows so models are always evaluated on identical data.
    """
    tokenizer:   object          # fitted Keras Tokenizer
    X_seq_train: np.ndarray
    X_seq_val:   np.ndarray
    X_seq_test:  np.ndarray
    X_txt_train: np.ndarray
    X_txt_val:   np.ndarray
    X_txt_test:  np.ndarray
    y_train:     np.ndarray
    y_val:       np.ndarray
    y_test:      np.ndarray


def clean_text(text: str) -> str:
    """Normalise a raw review string for NLP modelling.

    Steps:
    - Lowercase
    - Strip HTML tags
    - Expand common contractions (preserves sentiment-bearing negations)
    - Remove non-alphabetic characters
    - Collapse whitespace
    - Drop single-character tokens

    Stopwords are intentionally kept — "not", "very", "but" carry signal.
    """
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"n't",  " not",  text)
    text = re.sub(r"'s",   " is",   text)
    text = re.sub(r"'re",  " are",  text)
    text = re.sub(r"'ve",  " have", text)
    text = re.sub(r"'ll",  " will", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(w for w in text.split() if len(w) > 1)


def load_nlp_data(path: Path, sample_size: int = NLP_SAMPLE_SIZE) -> pd.DataFrame:
    """Load cleaned CSV and prepare a balanced binary-sentiment dataset.

    - Drops reviews shorter than 10 characters.
    - Drops neutral rating (Satisfaction == 3) — it is noise for binary classification.
    - Labels: Satisfaction >= 4 → Positive (1), <= 2 → Negative (0).
    - Balances to sample_size // 2 per class.
    - Applies clean_text() to produce the Clean_Review column.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {path}\n"
            "Run `uv run webmd-eda` first to generate it."
        )

    df = pd.read_csv(path)
    df = df[df["Reviews"].str.strip().str.len() > 10].copy()
    df = df[df["Satisfaction"] != 3].copy()
    df["Sentiment"] = (df["Satisfaction"] >= 4).astype(int)

    df = (
        df.groupby("Sentiment", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), sample_size // 2), random_state=ML_RANDOM_STATE))
        .reset_index(drop=True)
    )

    df["Clean_Review"] = df["Reviews"].apply(clean_text)
    df = df[df["Clean_Review"].str.len() > 5]

    pos = (df["Sentiment"] == 1).sum()
    neg = (df["Sentiment"] == 0).sum()
    print(f"NLP dataset: {len(df):,} samples  |  Positive: {pos:,}  Negative: {neg:,}")
    return df


def prepare_data(df: pd.DataFrame) -> NlpSplits:
    """Fit a Keras Tokenizer and produce a single unified 70/15/15 split.

    Both padded sequences (LSTM) and raw texts (TF-IDF) are sliced from the
    same index array so both models always train and evaluate on identical rows.
    """
    # Heavy import kept local — avoids loading TF at CLI startup
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>", lower=True)
    tokenizer.fit_on_texts(df["Clean_Review"])

    sequences = tokenizer.texts_to_sequences(df["Clean_Review"])
    X_seq = pad_sequences(sequences, maxlen=MAX_LEN, truncating="pre", padding="pre")
    X_txt = df["Clean_Review"].values
    y     = df["Sentiment"].values

    idx = np.arange(len(y))
    idx_train, idx_temp = train_test_split(
        idx, test_size=0.30, random_state=ML_RANDOM_STATE, stratify=y
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, random_state=ML_RANDOM_STATE, stratify=y[idx_temp]
    )

    print(
        f"Split — Train: {len(idx_train):,} | "
        f"Val: {len(idx_val):,} | Test: {len(idx_test):,}"
    )

    return NlpSplits(
        tokenizer=tokenizer,
        X_seq_train=X_seq[idx_train], X_seq_val=X_seq[idx_val], X_seq_test=X_seq[idx_test],
        X_txt_train=X_txt[idx_train], X_txt_val=X_txt[idx_val], X_txt_test=X_txt[idx_test],
        y_train=y[idx_train],         y_val=y[idx_val],         y_test=y[idx_test],
    )
