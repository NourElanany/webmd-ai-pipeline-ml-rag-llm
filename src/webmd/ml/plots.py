# ============================================================
# ml/plots.py — Generate and save all 5 ML evaluation plots
# ============================================================

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

from webmd.ml.features import FEATURE_NAMES
from webmd.ml.train import MlSplits, ModelResult

matplotlib.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid")


def generate_ml_plots(
    results: dict[str, ModelResult],
    best_name: str,
    splits: MlSplits,
    out_dir: Path,
) -> dict[str, Path]:
    """Generate all 5 ML evaluation plots and save them to *out_dir*.

    Returns a mapping of plot name → saved file path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, Path] = {}

    plots["Model Comparison"]   = _plot_model_comparison(results, best_name, out_dir)
    plots["Confusion Matrix"]   = _plot_confusion_matrix(results, best_name, out_dir)
    plots["Feature Importance"] = _plot_feature_importance(results, out_dir)
    plots["Per-Class Metrics"]  = _plot_per_class_metrics(results, best_name, out_dir)
    plots["Actual vs Predicted"]= _plot_actual_vs_predicted(results, best_name, splits, out_dir)

    print(f"All {len(plots)} ML plots saved to '{out_dir}/'")
    return plots


# ── Private plot helpers ─────────────────────────────────────────────────────

def _plot_model_comparison(
    results: dict[str, ModelResult],
    best_name: str,
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Comparison", fontsize=15, fontweight="bold")

    names   = list(results.keys())
    accs    = [results[n].accuracy for n in names]
    f1s     = [results[n].f1       for n in names]
    val_f1s = [results[n].val_f1   for n in names]
    colors  = ["#7c6af7" if n == best_name else "#4C72B0" for n in names]

    for ax, vals, title in zip(
        axes,
        [accs, val_f1s, f1s],
        ["Test Accuracy", "Val F1 (Weighted)", "Test F1 (Weighted)"],
    ):
        ax.barh(names, vals, color=colors, edgecolor="white")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    path = out_dir / "plot1_model_comparison.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_confusion_matrix(
    results: dict[str, ModelResult],
    best_name: str,
    out_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        results[best_name].cm,
        annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=[1, 2, 3, 4, 5],
        yticklabels=[1, 2, 3, 4, 5],
    )
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    path = out_dir / "plot2_confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_feature_importance(
    results: dict[str, ModelResult],
    out_dir: Path,
) -> Path:
    # Prefer XGBoost; fall back to Random Forest
    fi_name = next(
        (n for n in ["XGBoost", "Random Forest"] if n in results), None
    )
    if fi_name is None:
        raise ValueError("Neither XGBoost nor Random Forest found in results.")

    importances = results[fi_name].model.feature_importances_
    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        range(len(importances)),
        importances[idx],
        color="#7c6af7", edgecolor="white",
    )
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in idx], rotation=30, ha="right")
    ax.set_title(f"Feature Importance — {fi_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Importance Score")
    for bar, val in zip(bars, importances[idx]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}", ha="center", fontsize=8,
        )
    plt.tight_layout()
    path = out_dir / "plot3_feature_importance.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_per_class_metrics(
    results: dict[str, ModelResult],
    best_name: str,
    out_dir: Path,
) -> Path:
    report  = results[best_name].report
    classes = [str(i) for i in range(1, 6)]
    prec    = [report[c]["precision"] for c in classes if c in report]
    f1s     = [report[c]["f1-score"]  for c in classes if c in report]
    rec     = [report[c]["recall"]    for c in classes if c in report]

    x     = np.arange(len(classes))
    width = 0.28
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, prec, width, label="Precision", color="#4C72B0", edgecolor="white")
    ax.bar(x,         f1s,  width, label="F1",        color="#7c6af7", edgecolor="white")
    ax.bar(x + width, rec,  width, label="Recall",    color="#55A868", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {c}" for c in classes])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"Per-Class Metrics — {best_name}", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "plot4_per_class_metrics.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_actual_vs_predicted(
    results: dict[str, ModelResult],
    best_name: str,
    splits: MlSplits,
    out_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Actual vs Predicted Distribution — {best_name}",
        fontsize=13, fontweight="bold",
    )
    y_pred = results[best_name].y_pred
    for ax, data, title, color in zip(
        axes,
        [splits.y_test, y_pred],
        ["Actual", "Predicted"],
        ["#4C72B0", "#7c6af7"],
    ):
        vals, cnts = np.unique(data, return_counts=True)
        ax.bar(vals, cnts, color=color, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("Effectiveness Rating")
        ax.set_ylabel("Count")
        ax.set_xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    path = out_dir / "plot5_actual_vs_predicted.png"
    plt.savefig(path)
    plt.close()
    return path
