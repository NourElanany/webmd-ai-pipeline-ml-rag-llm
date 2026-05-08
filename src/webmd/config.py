# ============================================================
# config.py — Single source of truth for all project constants
# Paths, hyperparameters, GUI theme, plot registries
# ============================================================

from __future__ import annotations

import os
from pathlib import Path

# ── Root & directory layout ──────────────────────────────────────────────────
# All paths are relative to the project root (where pyproject.toml lives).
# Resolved at import time so every module gets absolute paths.

_HERE = Path(__file__).resolve()          # src/webmd/config.py
PROJECT_ROOT = _HERE.parents[2]           # project root

DATA_DIR      = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR     = PROJECT_ROOT / "assets"
CHROMA_DIR    = PROJECT_ROOT / "chroma_db"

# Input data
RAW_CSV     = DATA_DIR / "src" / "webmd.csv"
CLEANED_CSV = DATA_DIR / "webmd_cleaned.csv"

# Artifact files
ML_MODEL_PATH       = ARTIFACTS_DIR / "rf_effectiveness_model.pkl"
TFIDF_SVC_PATH      = ARTIFACTS_DIR / "tfidf_svc.pkl"
TFIDF_LR_PATH       = ARTIFACTS_DIR / "tfidf_lr.pkl"
TFIDF_WORD_VEC_PATH = ARTIFACTS_DIR / "tfidf_word_vectorizer.pkl"
TFIDF_CHAR_VEC_PATH = ARTIFACTS_DIR / "tfidf_char_vectorizer.pkl"
LSTM_MODEL_PATH     = ARTIFACTS_DIR / "lstm_sentiment_model.keras"
LSTM_TOKENIZER_PATH = ARTIFACTS_DIR / "lstm_tokenizer.pkl"

# Plot output directories
EDA_PLOTS_DIR = PLOTS_DIR / "analysis"
ML_PLOTS_DIR  = PLOTS_DIR / "ml"
NLP_PLOTS_DIR = PLOTS_DIR / "nlp"

# ── Plot registries (name → path) ────────────────────────────────────────────
EDA_PLOT_NAMES: dict[str, Path] = {
    "Ratings Distribution": EDA_PLOTS_DIR / "plot1_ratings_distribution.png",
    "Gender Distribution":  EDA_PLOTS_DIR / "plot2_gender_distribution.png",
    "Age Distribution":     EDA_PLOTS_DIR / "plot3_age_distribution.png",
    "Top Conditions":       EDA_PLOTS_DIR / "plot4_top_conditions.png",
    "Top Drugs Ratings":    EDA_PLOTS_DIR / "plot5_top_drugs_ratings.png",
    "Reviews Over Years":   EDA_PLOTS_DIR / "plot6_reviews_over_years.png",
    "Correlation Heatmap":  EDA_PLOTS_DIR / "plot7_correlation_heatmap.png",
    "Review Length Dist.":  EDA_PLOTS_DIR / "plot8_review_length_dist.png",
}

ML_PLOT_NAMES: dict[str, Path] = {
    "Model Comparison":  ML_PLOTS_DIR / "plot1_model_comparison.png",
    "Confusion Matrix":  ML_PLOTS_DIR / "plot2_confusion_matrix.png",
    "Feature Importance":ML_PLOTS_DIR / "plot3_feature_importance.png",
    "Per-Class Metrics": ML_PLOTS_DIR / "plot4_per_class_metrics.png",
    "Actual vs Predicted":ML_PLOTS_DIR / "plot5_actual_vs_predicted.png",
}

NLP_PLOT_NAMES: dict[str, Path] = {
    "Training History":          NLP_PLOTS_DIR / "plot1_training_history.png",
    "Model Comparison":          NLP_PLOTS_DIR / "plot2_model_comparison.png",
    "Confusion Matrix":          NLP_PLOTS_DIR / "plot3_confusion_matrix.png",
    "Top Side Effects":          NLP_PLOTS_DIR / "plot4_top_side_effects.png",
    "Side Effects by Sentiment": NLP_PLOTS_DIR / "plot5_side_effects_by_sentiment.png",
    "Confidence Distribution":   NLP_PLOTS_DIR / "plot6_confidence_distribution.png",
}

# ── ML hyperparameters ───────────────────────────────────────────────────────
TOP_CONDITIONS   = 50       # number of top conditions kept; rest → "Other"
ML_RANDOM_STATE  = 42
ML_TEST_SIZE     = 0.30     # first split: 70% train, 30% temp
ML_VAL_RATIO     = 0.50     # second split: 50% of temp → val, 50% → test

AGE_MAP: dict[str, int] = {
    "0-2": 1, "3-6": 4, "7-12": 9, "13-18": 15,
    "19-24": 21, "25-34": 29, "35-44": 39,
    "45-54": 49, "55-64": 59, "65-74": 69, "75 or over": 80,
}

# ── NLP hyperparameters ──────────────────────────────────────────────────────
VOCAB_SIZE      = 30_000
MAX_LEN         = 150
EMBED_DIM       = 128
EPOCHS          = 10
NLP_SAMPLE_SIZE = 80_000    # total balanced samples (40k pos + 40k neg)
TFIDF_LSTM_WEIGHT = 0.20    # LSTM weight in final ensemble (TF-IDF gets 1 - this)

# ── RAG / LLM config ─────────────────────────────────────────────────────────
COLLECTION_NAME = "webmd_reviews"
EMBED_MODEL     = "all-MiniLM-L6-v2"
RAG_SAMPLE_SIZE = 50_000
RAG_TOP_K       = 7
LLM_DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# Read from environment — never hardcoded
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL: str = os.environ.get("OPENROUTER_MODEL", LLM_DEFAULT_MODEL)

# ── GUI theme ────────────────────────────────────────────────────────────────
COLORS: dict[str, str] = {
    "dark_bg":  "#1e1e2e",
    "panel_bg": "#2a2a3e",
    "accent":   "#7c6af7",
    "accent2":  "#cba6f7",
    "text":     "#cdd6f4",
    "subtext":  "#a6adc8",
    "card_bg":  "#313244",
    "green":    "#a6e3a1",
    "red":      "#f38ba8",
    "yellow":   "#f9e2af",
    "border":   "#45475a",
    "blue":     "#89b4fa",
    "orange":   "#fab387",
    "pink":     "#f5c2e7",
    "teal":     "#94e2d5",
    "accent_active": "#6355d4",
}

FONTS: dict[str, tuple] = {
    "title":  ("Segoe UI", 14, "bold"),
    "bold":   ("Segoe UI", 10, "bold"),
    "normal": ("Segoe UI", 10),
    "small":  ("Segoe UI", 9),
    "mono":   ("Consolas", 9),
    "large":  ("Segoe UI", 15, "bold"),
    "xlarge": ("Segoe UI", 18, "bold"),
    "h2":     ("Segoe UI", 12, "bold"),
    "h3":     ("Segoe UI", 11, "bold"),
}
