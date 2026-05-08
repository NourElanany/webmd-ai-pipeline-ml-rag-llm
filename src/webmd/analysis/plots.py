# ============================================================
# analysis/plots.py — Generate and save all 8 EDA plots
# ============================================================

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import pandas as pd

matplotlib.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid", palette="muted")


def generate_eda_plots(df: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Generate all 8 EDA visualisations and save them to *out_dir*.

    Returns a mapping of plot name → saved file path (mirrors EDA_PLOT_NAMES
    in config.py so the GUI can resolve paths without re-running this function).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, Path] = {}

    plots["Ratings Distribution"]  = _plot_ratings_distribution(df, out_dir)
    plots["Gender Distribution"]   = _plot_gender_distribution(df, out_dir)
    plots["Age Distribution"]      = _plot_age_distribution(df, out_dir)
    plots["Top Conditions"]        = _plot_top_conditions(df, out_dir)
    plots["Top Drugs Ratings"]     = _plot_top_drugs_ratings(df, out_dir)
    plots["Reviews Over Years"]    = _plot_reviews_over_years(df, out_dir)
    plots["Correlation Heatmap"]   = _plot_correlation_heatmap(df, out_dir)
    plots["Review Length Dist."]   = _plot_review_length_dist(df, out_dir)

    print(f"All {len(plots)} EDA plots saved to '{out_dir}/'")
    return plots


# ── Private plot helpers ─────────────────────────────────────────────────────

def _plot_ratings_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Ratings Distribution", fontsize=16, fontweight="bold")
    for ax, col, color, label in zip(
        axes,
        ["Satisfaction", "Effectiveness", "EaseofUse"],
        ["#4C72B0", "#DD8452", "#55A868"],
        ["Satisfaction", "Effectiveness", "Ease of Use"],
    ):
        counts = df[col].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=color, edgecolor="white")
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Rating (1-5)")
        ax.set_ylabel("Number of Reviews")
        ax.set_xticks([1, 2, 3, 4, 5])
        for i, v in zip(counts.index, counts.values):
            ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=8)
    plt.tight_layout()
    path = out_dir / "plot1_ratings_distribution.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_gender_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    sex_counts = df["Sex"].value_counts()
    sex_counts = sex_counts[sex_counts.index != "Unknown"]
    ax.pie(
        sex_counts.values,
        labels=sex_counts.index,
        autopct="%1.1f%%",
        colors=["#4C72B0", "#DD8452", "#55A868"],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax.set_title("Gender Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "plot2_gender_distribution.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_age_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    age_order = [
        "7-12", "13-18", "19-24", "25-34", "35-44",
        "45-54", "55-64", "65-74", "75 or over",
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    age_counts = df["Age"].value_counts()
    age_counts = age_counts.reindex([a for a in age_order if a in age_counts.index])
    age_df = age_counts.reset_index()
    age_df.columns = ["Age", "Count"]
    sns.barplot(data=age_df, x="Age", y="Count", hue="Age", palette="Blues_d",
                legend=False, ax=ax)
    ax.set_title("Age Group Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Number of Reviews")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(age_counts.values):
        ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=8)
    plt.tight_layout()
    path = out_dir / "plot3_age_distribution.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_top_conditions(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    top_cond = df["Condition"].value_counts().head(15)
    top_cond = top_cond[top_cond.index != "Unknown"]
    top_cond_df = top_cond.reset_index()
    top_cond_df.columns = ["Condition", "Count"]
    sns.barplot(data=top_cond_df, x="Count", y="Condition", hue="Condition",
                palette="viridis", legend=False, ax=ax)
    ax.set_title("Top 15 Medical Conditions", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Condition")
    for i, v in enumerate(top_cond.values):
        ax.text(v + 100, i, f"{v:,}", va="center", fontsize=8)
    plt.tight_layout()
    path = out_dir / "plot4_top_conditions.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_top_drugs_ratings(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    top_drugs = df["Drug"].value_counts().head(10).index
    drug_ratings = (
        df[df["Drug"].isin(top_drugs)]
        .groupby("Drug")[["Satisfaction", "Effectiveness", "EaseofUse"]]
        .mean()
        .round(2)
    )
    drug_ratings.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
    ax.set_title("Avg Ratings for Top 10 Most-Reviewed Drugs", fontsize=14, fontweight="bold")
    ax.set_xlabel("Drug")
    ax.set_ylabel("Average Rating")
    ax.set_ylim(0, 5.5)
    ax.legend(["Satisfaction", "Effectiveness", "Ease of Use"])
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    path = out_dir / "plot5_top_drugs_ratings.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_reviews_over_years(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    yearly = df.groupby("Year").size().reset_index(name="Count").dropna()
    ax.plot(yearly["Year"], yearly["Count"], marker="o", linewidth=2,
            color="#4C72B0", markersize=6)
    ax.fill_between(yearly["Year"], yearly["Count"], alpha=0.15, color="#4C72B0")
    ax.set_title("Number of Reviews Over the Years", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Reviews")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    path = out_dir / "plot6_reviews_over_years.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    corr_cols = ["EaseofUse", "Effectiveness", "Satisfaction", "UsefulCount", "Review_Length"]
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, square=True)
    ax.set_title("Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "plot7_correlation_heatmap.png"
    plt.savefig(path)
    plt.close()
    return path


def _plot_review_length_dist(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    lengths = df["Review_Length"].clip(upper=2000)
    ax.hist(lengths, bins=50, color="#55A868", edgecolor="white", linewidth=0.5)
    ax.axvline(
        lengths.mean(), color="red", linestyle="--", linewidth=1.5,
        label=f"Mean: {lengths.mean():.0f} chars",
    )
    ax.axvline(
        lengths.median(), color="orange", linestyle="--", linewidth=1.5,
        label=f"Median: {lengths.median():.0f} chars",
    )
    ax.set_title("Review Length Distribution (Unstructured Text)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Review Length (chars)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "plot8_review_length_dist.png"
    plt.savefig(path)
    plt.close()
    return path
