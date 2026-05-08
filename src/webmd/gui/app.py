# ============================================================
# gui/app.py — Unified dashboard: root window + notebook assembly
# ============================================================

from __future__ import annotations

import os
import warnings

import tkinter as tk
from tkinter import ttk

from webmd.gui.theme import ACCENT, DARK_BG, FONTS, apply_style
from webmd.gui.tabs.home import build_home_tab
from webmd.gui.tabs.eda  import build_eda_tab
from webmd.gui.tabs.ml   import build_ml_tab
from webmd.gui.tabs.nlp  import build_nlp_tab
from webmd.gui.tabs.rag  import build_rag_tab


def main() -> None:
    """Launch the unified WebMD AI Pipeline dashboard."""
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"

    # Load .env so OPENROUTER_API_KEY is available before any tab opens
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    root = tk.Tk()
    root.title("WebMD Drug Reviews — Final Project Dashboard")
    root.geometry("1350x820")
    root.minsize(1000, 650)
    root.configure(bg=DARK_BG)

    style = ttk.Style()
    apply_style(style)

    # ── Header ───────────────────────────────────────────────
    hdr = tk.Frame(root, bg=ACCENT, height=58)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)
    tk.Label(hdr, text="  🧬  WebMD Drug Reviews  |  ML + DL + LLM + RAG",
             bg=ACCENT, fg="#fff",
             font=("Segoe UI", 15, "bold")).pack(side="left", padx=16, pady=10)
    tk.Label(hdr, text="Final Project  •  All Phases Integrated",
             bg=ACCENT, fg="#e0e0ff",
             font=FONTS["normal"]).pack(side="right", padx=20)

    # ── Main notebook ────────────────────────────────────────
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=8, pady=8)

    build_home_tab(nb)
    build_eda_tab(nb)
    build_ml_tab(nb)
    build_nlp_tab(nb)
    build_rag_tab(nb)

    root.mainloop()
