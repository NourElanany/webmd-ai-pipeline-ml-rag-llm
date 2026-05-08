# ============================================================
# nlp/plots.py — Generate and save all 6 NLP evaluation plots
# ============================================================

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Any

import pandas as pd

from webmd.nlp.side_effects import by_sentiment, extract_corpus
from webmd.nlp.tfidf import NlpResult

matplotlib.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid")


def generate_nlp_plots(
    history: Any,
    lstm_result: NlpResult,
    tfidf_result: NlpResult,
    ensemble_result: NlpResult,
    df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    """Generate all 6 NLP evaluation plots and save them to *out_dir*.

    Returns a mapping of plot name → saved file path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, Path] = {}

    plots["Training History"]          = _plot_training_history(history, out_dir)
    plots["Model Comparison"]          = _plot_model_comparison(lstm_result, tfidf_result, ensemble_result, out_dir)
    plots["Confusion Matrix"]          = _plot_confusion_matrix(ensemble_result, out_dir)
    plots["Top Side Effects"]          = _plot_top_side_effects(df, out_dir)
    plots["Side Effects by Sentiment"] = _plot_side_effects_by_sentiment(df, out_dir)
    plots["Confidence Distribution"]   = _plot_confidence_distribution(ensemble_result, out_dir)

    print(f"All {len(plots)} NLP plots saved to '{out_dir}/'")
    return plots


# ── Private plot helpers ─────────────────────────────────────────────────────

def _plot_training_history(history: Any, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("BiLSTM Training History", fontsize=15, fontweight="bold")
    for ax, metric, title in zip(axes, ["loss", "accuracy"], ["Loss", "Accuracy"]):
        ax.plot(history.history[metric],          label="Train", color="#7c6af7", marker="o")
        ax.plot(history.history[f"val_{metric}"], label="Val",   color="#DD8452", marker="s")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    path = out_dir / "plot1_training_history.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_model_comparison(
    lstm_result: NlpResult,
    tfidf_result: NlpResult,
    ensemble_result: NlpResult,
    out_dir: Path,
) -> Path:
    model_names = ["BiLSTM", "TF-IDF+SVC", "Ensemble"]
    accs = [lstm_result.accuracy, tfidf_result.accuracy, ensemble_result.accuracy]
    f1s  = [lstm_result.f1,       tfidf_result.f1,       ensemble_result.f1]

    x, w = np.arange(len(model_names)), 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, accs, w, label="Accuracy", color="#7c6af7", edgecolor="white")
    ax.bar(x + w / 2, f1s,  w, label="F1",       color="#a6e3a1", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0.7, 1.0)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w / 2, a + 0.003, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + w / 2, f + 0.003, f"{f:.3f}", ha="center", fontsize=9)
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "plot2_model_comparison.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_confusion_matrix(result: NlpResult, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        result.cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    ax.set_title("Confusion Matrix — Ensemble", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    path = out_dir / "plot3_confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_top_side_effects(df: pd.DataFrame, out_dir: Path) -> Path:
    side_effects = extract_corpus(df, top_n=20)
    if not side_effects:
        # Produce an empty placeholder so the registry key always exists
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("Top 20 Side Effects — No data", fontsize=13)
        path = out_dir / "plot4_top_side_effects.png"
        plt.savefig(path)
        plt.close()
        return path

    labels, counts = zip(*side_effects)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels[::-1], counts[::-1], color="#7c6af7", edgecolor="white")
    ax.set_title("Top 20 Side Effects Mentioned in Reviews", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mention Count")
    for bar, val in zip(bars, counts[::-1]):
        ax.text(
            bar.get_width() + 50,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=8,
        )
    plt.tight_layout()
    path = out_dir / "plot4_top_side_effects.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_side_effects_by_sentiment(df: pd.DataFrame, out_dir: Path) -> Path:
    kws, neg_norm, pos_norm = by_sentiment(df, top_n=15)
    x, width = np.arange(len(kws)), 0.38
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, [neg_norm.get(k, 0) for k in kws], width,
           label="Negative Reviews", color="#f38ba8", edgecolor="white")
    ax.bar(x + width / 2, [pos_norm.get(k, 0) for k in kws], width,
           label="Positive Reviews", color="#a6e3a1", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(kws, rotation=35, ha="right")
    ax.set_title(
        "Side-Effect Frequency: Negative vs Positive (Normalized)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylabel("Mentions per Review")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "plot5_side_effects_by_sentiment.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_confidence_distribution(result: NlpResult, out_dir: Path) -> Path:
    pos_probs = result.y_prob[result.y_pred == 1]
    neg_probs = result.y_prob[result.y_pred == 0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pos_probs, bins=40, alpha=0.7, color="#a6e3a1",
            label="Predicted Positive", edgecolor="white")
    ax.hist(neg_probs, bins=40, alpha=0.7, color="#f38ba8",
            label="Predicted Negative", edgecolor="white")
    ax.axvline(0.5, color="white", linestyle="--", linewidth=1.5, label="Decision Boundary")
    ax.set_title(
        "Prediction Confidence Distribution (Ensemble)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Predicted Probability (Positive)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "plot6_confidence_distribution.png"
    plt.savefig(path)
    plt.close()
    return path
