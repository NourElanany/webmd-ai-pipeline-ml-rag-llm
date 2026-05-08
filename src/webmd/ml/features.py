# ============================================================
# ml/features.py — Feature engineering for effectiveness prediction
# ============================================================

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from webmd.config import AGE_MAP, TOP_CONDITIONS

# Human-readable names aligned with the feature column order used in training.
# Used by ml/plots.py for axis labels.
FEATURE_NAMES: list[str] = [
    "Age", "Sex", "Condition", "Ease of Use",
    "Satisfaction", "Useful Count", "Year", "Review Length",
]

_FEATURE_COLS: list[str] = [
    "Age_Num", "Sex_Enc", "Condition_Enc",
    "EaseofUse", "Satisfaction", "UsefulCount",
    "Year", "Review_Length",
]


@dataclass
class FeatureArtifacts:
    """Holds the fitted LabelEncoder so the GUI can encode live inputs."""
    label_encoder: LabelEncoder


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, FeatureArtifacts]:
    """Engineer all features from the cleaned DataFrame.

    Filters rows, encodes categorical columns, and returns X, y, and the
    fitted LabelEncoder needed for live prediction in the GUI.

    Returns:
        X:         Feature DataFrame (8 columns).
        y:         Target Series (Effectiveness 1-5).
        artifacts: FeatureArtifacts with the fitted LabelEncoder.
    """
    df = df.copy()

    # Map age groups to numeric midpoints; drop rows with unknown age
    df["Age_Num"] = df["Age"].map(AGE_MAP)
    df = df[df["Age_Num"].notna()]

    # Keep only known sex values
    df = df[df["Sex"].str.strip().isin(["Male", "Female"])]

    # Drop rows with unknown condition
    df = df[df["Condition"].str.strip() != "Unknown"]

    # Binary-encode sex
    df["Sex_Enc"] = (df["Sex"].str.strip() == "Female").astype(int)

    # Label-encode top-N conditions; remainder → "Other"
    top_conditions = df["Condition"].value_counts().head(TOP_CONDITIONS).index
    df["Condition_Clean"] = df["Condition"].where(
        df["Condition"].isin(top_conditions), other="Other"
    )
    le = LabelEncoder()
    df["Condition_Enc"] = le.fit_transform(df["Condition_Clean"])

    # Keep only valid target values
    df = df[df["Effectiveness"].between(1, 5)]

    X = df[_FEATURE_COLS].copy()
    y = df["Effectiveness"].copy()

    print(f"Prepared dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"Target distribution:\n{y.value_counts().sort_index().to_string()}")

    return X, y, FeatureArtifacts(label_encoder=le)


def encode_input(
    age_group: str,
    sex: str,
    condition: str,
    ease_of_use: int,
    satisfaction: int,
    useful_count: int,
    year: int,
    review_length: int,
    artifacts: FeatureArtifacts,
) -> np.ndarray:
    """Encode a single live prediction input into a (1, 8) feature array.

    Used by the GUI live predictor. Condition is encoded with the fitted
    LabelEncoder; unknown conditions fall back to the median class index.
    """
    age_num = AGE_MAP[age_group]
    sex_enc = 1 if sex.strip() == "Female" else 0

    le = artifacts.label_encoder
    if condition in le.classes_:
        cond_enc = int(le.transform([condition])[0])
    else:
        cond_enc = int(len(le.classes_) // 2)  # median fallback

    return np.array([[
        age_num, sex_enc, cond_enc,
        ease_of_use, satisfaction, useful_count,
        year, review_length,
    ]])
