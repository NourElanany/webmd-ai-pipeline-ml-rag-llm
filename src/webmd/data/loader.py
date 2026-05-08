# ============================================================
# data/loader.py — Load raw and cleaned CSV files
# ============================================================

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw(path: Path) -> pd.DataFrame:
    """Load the raw WebMD CSV from disk.

    Raises FileNotFoundError with a helpful message if the file is missing
    (it is not committed — must be downloaded from Kaggle).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {path}\n"
            "Download 'webmd.csv' from Kaggle and place it at that path."
        )
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def load_cleaned(path: Path) -> pd.DataFrame:
    """Load the cleaned CSV produced by Phase 1.

    Raises FileNotFoundError if the file is missing (run `uv run webmd-eda` first).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {path}\n"
            "Run `uv run webmd-eda` first to generate it."
        )
    return pd.read_csv(path)
