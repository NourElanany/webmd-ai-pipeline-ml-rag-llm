# ============================================================
# nlp/side_effects.py — Side-effect keyword detection
# Single source of truth for KEYWORDS and NEGATIONS
# ============================================================

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

# Full list of known medical side-effect terms.
# Used by both the training pipeline (corpus analysis) and the live analyzer.
SIDE_EFFECT_KEYWORDS: list[str] = [
    "drowsiness", "dizziness", "nausea", "headache", "fatigue", "vomiting",
    "diarrhea", "constipation", "rash", "insomnia", "anxiety", "depression",
    "weight gain", "weight loss", "dry mouth", "blurred vision", "sweating",
    "tremor", "palpitations", "shortness of breath", "chest pain", "itching",
    "swelling", "fever", "chills", "muscle pain", "joint pain", "hair loss",
    "stomach pain", "upset stomach", "bloating", "gas", "heartburn", "cramps",
    "confusion", "memory loss", "mood swings", "irritability", "numbness",
    "tingling", "weakness", "loss of appetite", "increased appetite",
    "dry skin", "acne", "bruising", "bleeding", "infection", "back pain",
]

# Union of all negation terms from both original scripts.
NEGATIONS: frozenset[str] = frozenset({
    "no", "not", "never", "without", "neither", "nor",
    "hardly", "barely", "scarcely",
    "didn't", "don't", "doesn't", "wasn't", "weren't", "haven't", "hasn't",
    "free", "absence", "absent", "lack", "lacking",
})


def detect(text: str) -> list[str]:
    """Negation-aware side-effect detection for a single review string.

    Scans for each keyword; if any of the 5 words immediately preceding it
    is a negation term the keyword is skipped.

    Returns a list of detected (non-negated) side-effect keywords.
    """
    text_lower = text.lower()
    found: list[str] = []
    for kw in SIDE_EFFECT_KEYWORDS:
        idx = text_lower.find(kw)
        if idx == -1:
            continue
        before = text_lower[:idx].split()[-5:]
        if not any(neg in before for neg in NEGATIONS):
            found.append(kw)
    return found


def extract_corpus(df: pd.DataFrame, top_n: int = 30) -> list[tuple[str, int]]:
    """Count keyword mentions across the Reviews and Sides columns of *df*.

    Returns the top-*n* (keyword, count) pairs sorted by frequency.
    """
    freq: Counter[str] = Counter()

    for text in df["Reviews"].str.lower():
        for kw in SIDE_EFFECT_KEYWORDS:
            if kw in text:
                freq[kw] += 1

    for sides_text in df["Sides"].dropna():
        parts = str(sides_text).lower().split(",")
        for part in parts:
            part = re.sub(r"\s+", " ", part.strip().strip("."))
            if 3 < len(part) < 40 and part not in ("", " ", "may occur"):
                for kw in SIDE_EFFECT_KEYWORDS:
                    if kw in part:
                        freq[kw] += 1

    return freq.most_common(top_n)


def by_sentiment(
    df: pd.DataFrame,
    top_n: int = 15,
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Compute normalised side-effect frequency split by sentiment label.

    Expects *df* to have a 'Sentiment' column (0 = negative, 1 = positive).

    Returns:
        keywords:  Top keywords sorted by combined frequency.
        neg_norm:  Keyword → mentions-per-review for negative reviews.
        pos_norm:  Keyword → mentions-per-review for positive reviews.
    """
    neg_df = df[df["Sentiment"] == 0]
    pos_df = df[df["Sentiment"] == 1]

    neg_freq: Counter[str] = Counter()
    pos_freq: Counter[str] = Counter()

    for text in neg_df["Reviews"].str.lower():
        for kw in SIDE_EFFECT_KEYWORDS:
            if kw in text:
                neg_freq[kw] += 1

    for text in pos_df["Reviews"].str.lower():
        for kw in SIDE_EFFECT_KEYWORDS:
            if kw in text:
                pos_freq[kw] += 1

    neg_norm = {k: v / len(neg_df) for k, v in neg_freq.items()}
    pos_norm = {k: v / len(pos_df) for k, v in pos_freq.items()}

    all_kws = sorted(
        set(list(neg_norm)[:top_n] + list(pos_norm)[:top_n]),
        key=lambda k: neg_norm.get(k, 0) + pos_norm.get(k, 0),
        reverse=True,
    )[:top_n]

    return all_kws, neg_norm, pos_norm
