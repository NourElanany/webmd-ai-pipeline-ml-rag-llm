# ============================================================
# gui/tabs/ml.py — ML tab (Phase 2): plots + live predictor
# ============================================================

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np

from webmd.config import AGE_MAP, ML_MODEL_PATH, ML_PLOT_NAMES
from webmd.gui.theme import (
    ACCENT, CARD_BG, DARK_BG, FONT_BOLD, FONT_NORMAL,
    FONT_SMALL, FONT_TITLE, GREEN, RED, SUBTEXT, TEXT, YELLOW,
)
from webmd.gui.widgets import btn, plot_viewer

# Lazy-loaded cache — populated on first Predict click
_ml_cache: dict = {}


def _get_ml_artifacts():
    """Load the ML model and LabelEncoder on first call; cache for reuse."""
    if "model" not in _ml_cache:
        import joblib
        from sklearn.preprocessing import LabelEncoder
        import pandas as pd
        from webmd.config import CLEANED_CSV, TOP_CONDITIONS

        df = pd.read_csv(CLEANED_CSV)
        top_conditions = df["Condition"].value_counts().head(TOP_CONDITIONS).index
        df["Condition_Clean"] = df["Condition"].where(
            df["Condition"].isin(top_conditions), other="Other"
        )
        le = LabelEncoder()
        le.fit(df["Condition_Clean"])
        _ml_cache["le"]    = le
        _ml_cache["model"] = joblib.load(ML_MODEL_PATH)

    return _ml_cache["model"], _ml_cache["le"]


def build_ml_tab(nb: ttk.Notebook) -> None:
    """Add the ML tab (with Plots and Live Predictor sub-tabs) to *nb*."""
    tab = ttk.Frame(nb)
    nb.add(tab, text="  🤖  ML  ")

    inner_nb = ttk.Notebook(tab)
    inner_nb.pack(fill="both", expand=True, padx=6, pady=6)

    # ── Sub-tab: Plots ───────────────────────────────────────
    plots_tab = ttk.Frame(inner_nb)
    inner_nb.add(plots_tab, text="  Plots  ")
    plot_viewer(plots_tab, ML_PLOT_NAMES)

    # ── Sub-tab: Live Predictor ──────────────────────────────
    pred_tab = ttk.Frame(inner_nb)
    inner_nb.add(pred_tab, text="  Live Predictor  ")
    _build_predictor(pred_tab)


def _build_predictor(parent: ttk.Frame) -> None:
    outer = tk.Frame(parent, bg=DARK_BG)
    outer.pack(expand=True, pady=20)

    tk.Label(outer, text="Predict Drug Effectiveness",
             bg=DARK_BG, fg=TEXT, font=FONT_TITLE).grid(
        row=0, column=0, columnspan=2, pady=(0, 16))

    fields: dict[str, tk.StringVar] = {}
    defs = [
        ("Age Group",     "combo", list(AGE_MAP.keys())),
        ("Sex",           "combo", ["Male", "Female"]),
        ("Ease of Use",   "combo", ["1", "2", "3", "4", "5"]),
        ("Satisfaction",  "combo", ["1", "2", "3", "4", "5"]),
        ("Useful Count",  "entry", "5"),
        ("Year",          "entry", "2020"),
        ("Review Length", "entry", "200"),
    ]

    for i, (label, ftype, opts) in enumerate(defs):
        tk.Label(outer, text=label, bg=DARK_BG, fg=SUBTEXT,
                 font=FONT_NORMAL, width=14, anchor="e").grid(
            row=i + 1, column=0, padx=(0, 10), pady=5, sticky="e")

        var = tk.StringVar(value=opts[0] if ftype == "combo" else opts)
        if ftype == "combo":
            ttk.Combobox(outer, textvariable=var, values=opts,
                         state="readonly", width=22,
                         font=FONT_NORMAL).grid(row=i + 1, column=1, pady=5, sticky="w")
        else:
            tk.Entry(outer, textvariable=var, width=24, bg=CARD_BG, fg=TEXT,
                     insertbackground=TEXT, relief="flat",
                     font=FONT_NORMAL).grid(row=i + 1, column=1, pady=5, sticky="w")
        fields[label] = var

    res_var  = tk.StringVar(value="—")
    conf_var = tk.StringVar(value="")
    res_lbl  = tk.Label(outer, textvariable=res_var, bg=CARD_BG, fg=ACCENT,
                        font=("Segoe UI", 20, "bold"), width=28, pady=10)
    res_lbl.grid(row=len(defs) + 2, column=0, columnspan=2, pady=(14, 4))
    tk.Label(outer, textvariable=conf_var, bg=DARK_BG, fg=SUBTEXT,
             font=FONT_SMALL).grid(row=len(defs) + 3, column=0, columnspan=2)

    def _predict() -> None:
        res_var.set("⏳ Loading model...")
        conf_var.set("")
        outer.update()
        try:
            model, le = _get_ml_artifacts()
            age_num  = AGE_MAP[fields["Age Group"].get()]
            sex_enc  = 1 if fields["Sex"].get() == "Female" else 0
            cond_enc = int(len(le.classes_) // 2)   # median fallback (no condition field)
            ease     = int(fields["Ease of Use"].get())
            sat      = int(fields["Satisfaction"].get())
            useful   = int(fields["Useful Count"].get())
            year     = int(fields["Year"].get())
            rev_len  = int(fields["Review Length"].get())

            X     = np.array([[age_num, sex_enc, cond_enc, ease, sat, useful, year, rev_len]])
            pred  = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            conf  = proba.max() * 100
            stars = "★" * pred + "☆" * (5 - pred)

            res_var.set(f"Effectiveness: {pred}/5  {stars}")
            conf_var.set(f"Confidence: {conf:.1f}%")
            res_lbl.configure(fg=GREEN if pred >= 4 else (YELLOW if pred == 3 else RED))
        except Exception as exc:
            res_var.set("Error")
            conf_var.set(str(exc))
            res_lbl.configure(fg=RED)

    btn(outer, "  Predict  ", _predict).grid(
        row=len(defs) + 1, column=0, columnspan=2, pady=12)
