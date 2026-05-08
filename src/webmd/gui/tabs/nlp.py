# ============================================================
# gui/tabs/nlp.py — NLP/DL tab (Phase 3): plots + live analyzer
# Loads pre-trained artifacts from disk — never retrains.
# ============================================================

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk

from webmd.config import NLP_PLOT_NAMES
from webmd.gui.theme import (
    ACCENT, CARD_BG, DARK_BG, FONT_BOLD, FONT_NORMAL,
    FONT_SMALL, FONT_TITLE, ORANGE, RED, SUBTEXT, TEXT,
)
from webmd.gui.widgets import btn, plot_viewer

# Lazy-loaded cache — populated on first Analyze click
_nlp_cache: dict = {}


def _get_nlp_artifacts():
    """Load all NLP artifacts from disk on first call; cache for reuse.

    Loads: LSTM model, Keras tokenizer, TF-IDF SVC, LR, word vec, char vec.
    Raises FileNotFoundError (from load_tfidf / load_lstm) if not yet trained.
    """
    if "lstm" not in _nlp_cache:
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        from webmd.nlp.lstm import load_lstm
        from webmd.nlp.tfidf import load_tfidf

        tfidf = load_tfidf()
        lstm_model, tokenizer = load_lstm()

        _nlp_cache["lstm"]      = lstm_model
        _nlp_cache["tokenizer"] = tokenizer
        _nlp_cache["tfidf"]     = tfidf

    return _nlp_cache["lstm"], _nlp_cache["tokenizer"], _nlp_cache["tfidf"]


def build_nlp_tab(nb: ttk.Notebook) -> None:
    """Add the NLP/DL tab (Plots + Live Analyzer sub-tabs) to *nb*."""
    tab = ttk.Frame(nb)
    nb.add(tab, text="  🧠  NLP / DL  ")

    inner_nb = ttk.Notebook(tab)
    inner_nb.pack(fill="both", expand=True, padx=6, pady=6)

    # ── Sub-tab: Plots ───────────────────────────────────────
    plots_tab = ttk.Frame(inner_nb)
    inner_nb.add(plots_tab, text="  Plots  ")
    plot_viewer(plots_tab, NLP_PLOT_NAMES)

    # ── Sub-tab: Live Analyzer ───────────────────────────────
    live_tab = ttk.Frame(inner_nb)
    inner_nb.add(live_tab, text="  Live Analyzer  ")
    _build_analyzer(live_tab)


def _build_analyzer(parent: ttk.Frame) -> None:
    outer = tk.Frame(parent, bg=DARK_BG)
    outer.pack(fill="both", expand=True, padx=30, pady=16)

    tk.Label(outer, text="Sentiment & Side-Effect Analyzer",
             bg=DARK_BG, fg=TEXT, font=FONT_TITLE).pack(anchor="w", pady=(0, 10))

    tk.Label(outer, text="Enter a drug review:", bg=DARK_BG, fg=SUBTEXT,
             font=FONT_NORMAL).pack(anchor="w")

    review_box = tk.Text(outer, height=5, bg=CARD_BG, fg=TEXT,
                         insertbackground=TEXT, font=("Segoe UI", 11),
                         relief="flat", wrap="word")
    review_box.pack(fill="x", pady=(4, 10))
    review_box.insert("end",
        "This medication worked great for my condition. "
        "I experienced some drowsiness and dry mouth but overall very satisfied.")

    # Result card
    res_card = tk.Frame(outer, bg=CARD_BG, padx=20, pady=14)
    res_card.pack(fill="x", pady=(0, 10))

    sent_var   = tk.StringVar(value="—")
    conf_var   = tk.StringVar(value="")
    sides_var  = tk.StringVar(value="")
    status_var = tk.StringVar(value="")

    for row_i, (label, var, color) in enumerate([
        ("Sentiment:",    sent_var,  ACCENT),
        ("Confidence:",   conf_var,  TEXT),
        ("Side Effects:", sides_var, ORANGE),
    ]):
        tk.Label(res_card, text=label, bg=CARD_BG, fg=SUBTEXT,
                 font=FONT_NORMAL, width=14, anchor="e").grid(
            row=row_i, column=0, sticky="e", pady=4)
        tk.Label(res_card, textvariable=var, bg=CARD_BG, fg=color,
                 font=FONT_BOLD, wraplength=650, justify="left").grid(
            row=row_i, column=1, sticky="w", padx=12)

    tk.Label(outer, textvariable=status_var, bg=DARK_BG, fg=SUBTEXT,
             font=FONT_SMALL).pack(anchor="w")

    def _analyze() -> None:
        text = review_box.get("1.0", "end").strip()
        if not text:
            return
        sent_var.set("⏳ Loading models...")
        status_var.set("")
        outer.update()

        def _run() -> None:
            try:
                from webmd.nlp.ensemble import predict_ensemble
                from webmd.nlp.preprocess import clean_text
                from webmd.nlp.side_effects import detect

                lstm_model, tokenizer, tfidf = _get_nlp_artifacts()
                cleaned = clean_text(text)
                ensemble_prob, tfidf_prob, lstm_prob = predict_ensemble(
                    cleaned, tfidf, lstm_model, tokenizer
                )

                label  = "Positive 😊" if ensemble_prob >= 0.5 else "Negative 😞"
                conf   = ensemble_prob if ensemble_prob >= 0.5 else 1 - ensemble_prob
                found  = detect(text)

                sent_var.set(label)
                conf_var.set(
                    f"{conf * 100:.1f}%  "
                    f"(TF-IDF: {tfidf_prob:.2f} | LSTM: {lstm_prob:.2f})"
                )
                sides_var.set(", ".join(found) if found else "None detected")
                status_var.set("")
            except Exception as exc:
                sent_var.set("Error")
                status_var.set(str(exc))

        threading.Thread(target=_run, daemon=True).start()

    btn(outer, "  Analyze  ", _analyze).pack(anchor="w", pady=6)
