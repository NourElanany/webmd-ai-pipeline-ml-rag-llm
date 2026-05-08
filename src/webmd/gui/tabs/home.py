# ============================================================
# gui/tabs/home.py — Overview / Home tab
# ============================================================

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk

from webmd.config import CLEANED_CSV
from webmd.gui.theme import (
    ACCENT, BLUE, CARD_BG, DARK_BG, FONT_BOLD,
    FONT_NORMAL, FONT_SMALL, ORANGE, RED, SUBTEXT, YELLOW,
)
from webmd.gui.widgets import card_label, section_label


def build_home_tab(nb: ttk.Notebook) -> None:
    """Add the Overview tab to *nb*."""
    tab = ttk.Frame(nb)
    nb.add(tab, text="  🏠  Overview  ")

    canvas = tk.Canvas(tab, bg=DARK_BG, highlightthickness=0)
    sb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    canvas.pack(fill="both", expand=True)

    inner = tk.Frame(canvas, bg=DARK_BG)
    canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    # Title
    tk.Label(inner, text="WebMD Drug Reviews — Final Project",
             bg=DARK_BG, fg=ACCENT, font=("Segoe UI", 18, "bold")).pack(pady=(24, 4))
    tk.Label(inner, text="ML + DL + LLM + RAG Integration Pipeline",
             bg=DARK_BG, fg=SUBTEXT, font=("Segoe UI", 12)).pack(pady=(0, 24))

    # Phase cards
    phases = [
        ("📊", "Phase 1 — EDA",
         "Python · NumPy · Pandas · Matplotlib · Seaborn\n"
         "Data cleaning, descriptive stats, 8 visualizations",
         BLUE),
        ("🤖", "Phase 2 — Machine Learning",
         "Random Forest · Gradient Boosting · Logistic Regression · XGBoost\n"
         "Effectiveness prediction (1-5) · 70/15/15 split · Feature importance",
         "#a6e3a1"),
        ("🧠", "Phase 3 — Deep Learning & NLP",
         "BiLSTM · TF-IDF + LinearSVC Ensemble · Sentiment Analysis\n"
         "Side-effect extraction · Negation-aware detection · 6 plots",
         YELLOW),
        ("🔍", "Phase 4 — RAG + LLM",
         "ChromaDB · SentenceTransformers (all-MiniLM-L6-v2)\n"
         "Semantic retrieval · OpenRouter LLM · Arabic medical summaries",
         ORANGE),
    ]

    grid = tk.Frame(inner, bg=DARK_BG)
    grid.pack(padx=30, pady=10, fill="x")
    grid.columnconfigure(0, weight=1)
    grid.columnconfigure(1, weight=1)

    for i, (icon, title, desc, color) in enumerate(phases):
        f = tk.Frame(grid, bg=CARD_BG, padx=20, pady=16)
        f.grid(row=i // 2, column=i % 2, padx=10, pady=10, sticky="nsew")
        tk.Label(f, text=f"{icon}  {title}", bg=CARD_BG, fg=color,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(f, text=desc, bg=CARD_BG, fg=SUBTEXT,
                 font=FONT_SMALL, justify="left").pack(anchor="w", pady=(6, 0))

    # Dataset stats (loaded in background thread)
    section_label(inner, "Dataset")
    stats_frame = tk.Frame(inner, bg=DARK_BG)
    stats_frame.pack(padx=30, fill="x")
    for i in range(4):
        stats_frame.columnconfigure(i, weight=1)

    def _load_stats() -> None:
        try:
            import pandas as pd
            df = pd.read_csv(CLEANED_CSV)
            card_label(stats_frame, "Total Reviews",     f"{len(df):,}",                         ACCENT, 0, 0)
            card_label(stats_frame, "Unique Drugs",       f"{df['Drug'].nunique():,}",            BLUE,   1, 0)
            card_label(stats_frame, "Unique Conditions",  f"{df['Condition'].nunique():,}",       YELLOW, 2, 0)
            card_label(stats_frame, "Date Range",
                       f"{int(df['Year'].min())} – {int(df['Year'].max())}",                      ORANGE, 3, 0)
        except Exception as exc:
            tk.Label(stats_frame, text=f"Stats unavailable: {exc}",
                     bg=DARK_BG, fg=RED, font=FONT_SMALL).grid(row=0, column=0, columnspan=4)

    threading.Thread(target=_load_stats, daemon=True).start()
    tk.Label(inner, text="", bg=DARK_BG).pack(pady=20)
