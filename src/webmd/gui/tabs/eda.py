# ============================================================
# gui/tabs/eda.py — EDA tab (Phase 1)
# ============================================================

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk

from webmd.config import CLEANED_CSV, EDA_PLOT_NAMES
from webmd.gui.theme import (
    ACCENT, BLUE, DARK_BG, GREEN, ORANGE, RED, YELLOW,
)
from webmd.gui.widgets import card_label, plot_viewer


def build_eda_tab(nb: ttk.Notebook) -> None:
    """Add the EDA tab to *nb*."""
    tab = ttk.Frame(nb)
    nb.add(tab, text="  📊  EDA  ")

    # Stats cards row
    top = tk.Frame(tab, bg=DARK_BG)
    top.pack(fill="x", padx=12, pady=10)
    for i in range(5):
        top.columnconfigure(i, weight=1)

    def _load_stats() -> None:
        try:
            import pandas as pd
            df = pd.read_csv(CLEANED_CSV)
            card_label(top, "Total Reviews",     f"{len(df):,}",                        ACCENT,  0, 0)
            card_label(top, "Unique Drugs",       f"{df['Drug'].nunique():,}",           BLUE,    1, 0)
            card_label(top, "Unique Conditions",  f"{df['Condition'].nunique():,}",      YELLOW,  2, 0)
            card_label(top, "Avg Satisfaction",   f"{df['Satisfaction'].mean():.2f}/5",  GREEN,   3, 0)
            card_label(top, "Avg Effectiveness",  f"{df['Effectiveness'].mean():.2f}/5", ORANGE,  4, 0)
        except Exception as exc:
            tk.Label(top, text=f"Load error: {exc}", bg=DARK_BG, fg=RED).grid(row=0, column=0)

    threading.Thread(target=_load_stats, daemon=True).start()

    # Plot viewer
    viz = tk.Frame(tab, bg=DARK_BG)
    viz.pack(fill="both", expand=True, padx=8)
    plot_viewer(viz, EDA_PLOT_NAMES)
