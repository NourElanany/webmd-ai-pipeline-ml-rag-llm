# ============================================================
# data/cleaner.py — Clean raw WebMD DataFrame and persist result
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# Columns that must exist in the raw CSV
_TEXT_COLS    = ["Age", "Condition", "Sex", "Sides", "Reviews"]
_NUMERIC_COLS = ["EaseofUse", "Effectiveness", "Satisfaction"]


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply all cleaning steps to the raw DataFrame.

    Steps (identical to original analysis.py):
    1. Fill missing text columns with "Unknown".
    2. Fill missing numeric ratings with column median.
    3. Parse Date → extract Year.
    4. Strip review text, compute Review_Length.
    5. Drop duplicate rows.
    6. Drop rows with out-of-range ratings (outside 1–5).

    Returns:
        df:     Cleaned DataFrame.
        report: Dict with cleaning statistics for display in the GUI.
    """
    report: dict[str, Any] = {}

    # 1. Record missing values before filling
    missing_before = df.isnull().sum()
    report["missing_before"] = missing_before[missing_before > 0].to_dict()

    # 2. Fill missing text columns
    for col in _TEXT_COLS:
        df[col] = df[col].fillna("Unknown")

    # 3. Fill missing numeric columns with median
    for col in _NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].median())

    report["missing_after"] = int(df.isnull().sum().sum())

    # 4. Parse dates and extract year
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year

    # 5. Clean review text and compute character length
    df["Reviews"] = df["Reviews"].astype(str).str.strip()
    df["Review_Length"] = df["Reviews"].apply(len)

    # 6. Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)

    # 7. Remove out-of-range ratings
    for col in _NUMERIC_COLS:
        df = df[df[col].between(1, 5)]

    report["final_shape"] = df.shape

    # 8. Descriptive statistics for the report
    report["stats"] = (
        df[["EaseofUse", "Effectiveness", "Satisfaction", "UsefulCount", "Review_Length"]]
        .describe()
        .round(2)
    )

    print(f"Cleaning done. Final shape: {df.shape[0]:,} rows")
    return df, report


def save_cleaned(df: pd.DataFrame, path: Path) -> None:
    """Persist the cleaned DataFrame to CSV.

    Creates parent directories if they do not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Cleaned data saved to '{path}' — ready for Phase 2")
