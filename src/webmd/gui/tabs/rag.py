# ============================================================
# gui/tabs/rag.py — RAG + LLM tab (Phase 4+5)
# Also exposes launch_rag_window() for the standalone webmd-rag entry point.
# ============================================================

from __future__ import annotations

import re
import threading
import tkinter as tk
from tkinter import ttk

from webmd.config import RAG_TOP_K
from webmd.gui.theme import (
    ACCENT, ACCENT2, BLUE, BORDER, CARD_BG, DARK_BG,
    FONT_BOLD, FONT_NORMAL, FONT_SMALL, GREEN, ORANGE,
    PANEL_BG, RED, SUBTEXT, TEXT, YELLOW,
    COLORS, FONTS,
)
from webmd.gui.widgets import btn, make_entry, scrolled_text
from webmd.rag.llm import build_client, fallback_response, llm_generate
from webmd.rag.retriever import Hit, retrieve

# Lazy-loaded RAG cache for the unified dashboard
_rag_cache: dict = {}


def _get_rag():
    """Load ChromaDB collection and SentenceTransformer on first call."""
    if "col" not in _rag_cache:
        from sentence_transformers import SentenceTransformer
        from webmd.config import EMBED_MODEL
        from webmd.rag.indexer import load_index
        _rag_cache["col"] = load_index()
        _rag_cache["emb"] = SentenceTransformer(EMBED_MODEL)
    return _rag_cache["col"], _rag_cache["emb"]


def build_rag_tab(nb: ttk.Notebook) -> None:
    """Add the RAG + LLM tab to *nb* (unified dashboard)."""
    tab = ttk.Frame(nb)
    nb.add(tab, text="  🔍  RAG + LLM  ")

    # Lazy-load LLM client
    llm_client = build_client()

    _build_rag_ui(tab, llm_client, get_rag_fn=_get_rag)


def launch_rag_window(col, embedder, llm_client=None) -> None:
    """Open a standalone Tk window for the RAG system (webmd-rag entry point)."""
    from webmd.gui.theme import apply_style

    root = tk.Tk()
    root.title("WebMD RAG — Drug Review Retrieval")
    root.geometry("1280x820")
    root.minsize(900, 600)
    root.configure(bg=DARK_BG)

    style = ttk.Style()
    apply_style(style)

    hdr = tk.Frame(root, bg=ACCENT, height=56)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)
    tk.Label(hdr, text="  🔍  WebMD RAG + LLM  |  Drug Review Retrieval & Generation",
             bg=ACCENT, fg="#fff", font=FONTS["title"]).pack(side="left", padx=16)
    llm_status = "🟢 LLM متصل" if llm_client else "🔴 LLM غير متصل"
    llm_color  = COLORS["green"] if llm_client else COLORS["red"]
    tk.Label(hdr, text=llm_status, bg=ACCENT, fg=llm_color,
             font=("Segoe UI", 10, "bold")).pack(side="right", padx=8)
    tk.Label(hdr, text=f"📦 {col.count():,} reviews  •  Phase 5 / 5",
             bg=ACCENT, fg="#e0e0ff", font=FONTS["normal"]).pack(side="right", padx=16)

    def _get_preloaded():
        return col, embedder

    _build_rag_ui(root, llm_client, get_rag_fn=_get_preloaded)
    root.mainloop()


def _build_rag_ui(parent: tk.Widget, llm_client, get_rag_fn) -> None:
    """Build the full RAG search UI inside *parent*.

    *get_rag_fn* is a callable that returns (col, embedder) — allows both
    the lazy-loading dashboard path and the pre-loaded standalone path.
    """
    # ── Search panel ─────────────────────────────────────────
    sf = tk.Frame(parent, bg=PANEL_BG, pady=10, padx=14)
    sf.pack(fill="x")

    # Need a root widget for clipboard access in make_entry
    root = parent.winfo_toplevel()

    tk.Label(sf, text="Query:", bg=PANEL_BG, fg=SUBTEXT,
             font=FONT_BOLD).grid(row=0, column=0, sticky="w", padx=(0, 8))
    q_var = tk.StringVar(value="What are people's experiences with ibuprofen for headache?")
    make_entry(sf, q_var, root, font=("Segoe UI", 11),
               row=0, column=1, columnspan=3, sticky="ew", ipady=7)

    tk.Label(sf, text="Drug (opt):",      bg=PANEL_BG, fg=SUBTEXT, font=FONT_SMALL).grid(row=1, column=0, sticky="w", pady=(8, 0))
    drug_var = tk.StringVar()
    make_entry(sf, drug_var, root, width=20, row=1, column=1, sticky="w", pady=(8, 0), ipady=4)

    tk.Label(sf, text="Condition (opt):", bg=PANEL_BG, fg=SUBTEXT, font=FONT_SMALL).grid(row=1, column=2, sticky="w", padx=(16, 6), pady=(8, 0))
    cond_var = tk.StringVar()
    make_entry(sf, cond_var, root, width=20, row=1, column=3, sticky="w", pady=(8, 0), ipady=4)

    tk.Label(sf, text="Top-K:", bg=PANEL_BG, fg=SUBTEXT, font=FONT_SMALL).grid(row=1, column=4, sticky="w", padx=(16, 4), pady=(8, 0))
    topk_var = tk.IntVar(value=RAG_TOP_K)
    tk.Spinbox(sf, from_=1, to=20, textvariable=topk_var, width=4,
               bg=CARD_BG, fg=TEXT, buttonbackground=PANEL_BG,
               relief="flat", font=FONT_NORMAL).grid(row=1, column=5, sticky="w", pady=(8, 0))
    sf.columnconfigure(1, weight=1)
    sf.columnconfigure(3, weight=1)

    # ── Action bar ───────────────────────────────────────────
    ab = tk.Frame(parent, bg=DARK_BG, pady=6, padx=8)
    ab.pack(fill="x")
    status_var  = tk.StringVar(value="Ready")
    search_btn  = btn(ab, "  🔍  Search  ", lambda: None)
    search_btn.pack(side="left")
    copy_btn    = btn(ab, "  📋  Copy  ", lambda: None, bg=CARD_BG, fg=TEXT)
    copy_btn.pack(side="left", padx=8)
    tk.Label(ab, textvariable=status_var, bg=DARK_BG, fg=SUBTEXT,
             font=FONT_SMALL).pack(side="left", padx=10)

    # ── Results notebook ─────────────────────────────────────
    res_nb = ttk.Notebook(parent)
    res_nb.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # Summary sub-tab
    sum_tab = ttk.Frame(res_nb)
    res_nb.add(sum_tab, text="  📊 Summary  ")
    sum_frame, sum_txt = scrolled_text(sum_tab)
    for tag, fg, font in [
        ("header",  ACCENT2,          FONT_BOLD),
        ("label",   BLUE,             FONT_BOLD),
        ("value",   TEXT,             FONT_NORMAL),
        ("pos",     GREEN,            FONT_BOLD),
        ("neg",     RED,              FONT_BOLD),
        ("neutral", YELLOW,           FONT_BOLD),
        ("review",  COLORS["pink"],   FONT_NORMAL),
        ("meta",    COLORS["teal"],   FONT_SMALL),
        ("side",    ORANGE,           FONT_NORMAL),
        ("divider", BORDER,           FONT_SMALL),
    ]:
        sum_txt.tag_configure(tag, foreground=fg, font=font)
    sum_frame.pack(fill="both", expand=True)

    # LLM sub-tab
    llm_tab = ttk.Frame(res_nb)
    res_nb.add(llm_tab, text="  🤖 LLM Answer  ")
    llm_frame, llm_txt = scrolled_text(llm_tab)
    llm_txt.tag_configure("llm_head", foreground=ACCENT2,  font=("Segoe UI", 11, "bold"))
    llm_txt.tag_configure("llm_body", foreground=TEXT,     font=("Segoe UI", 11))
    llm_txt.tag_configure("llm_warn", foreground=YELLOW,   font=FONT_SMALL)
    llm_txt.tag_configure("llm_key",  foreground=BLUE,     font=FONT_BOLD)
    llm_txt.tag_configure("divider",  foreground=BORDER,   font=FONT_SMALL)
    llm_frame.pack(fill="both", expand=True)

    # Table sub-tab
    tbl_tab = ttk.Frame(res_nb)
    res_nb.add(tbl_tab, text="  Retrieved Reviews  ")
    tf = tk.Frame(tbl_tab, bg=DARK_BG)
    tf.pack(fill="both", expand=True)
    cols   = ["#", "Sim", "Drug", "Condition", "Sat", "Eff", "Age", "Sex", "Review"]
    widths = [30,   65,    110,    130,          45,    45,    80,    50,    0]
    tree   = ttk.Treeview(tf, columns=cols, show="headings", selectmode="browse")
    for c, w in zip(cols, widths):
        tree.heading(c, text=c)
        tree.column(c, width=w, stretch=(w == 0),
                    anchor="center" if w < 100 else "w",
                    minwidth=w if w > 0 else 250)
    sbx = ttk.Scrollbar(tf, orient="horizontal", command=tree.xview)
    sby = ttk.Scrollbar(tf, command=tree.yview)
    tree.configure(yscrollcommand=sby.set, xscrollcommand=sbx.set)
    sby.pack(side="right", fill="y")
    sbx.pack(side="bottom", fill="x")
    tree.pack(fill="both", expand=True)

    det_frame = tk.Frame(tbl_tab, bg=PANEL_BG, height=110)
    det_frame.pack(fill="x")
    det_frame.pack_propagate(False)
    tk.Label(det_frame, text="Full Review:", bg=PANEL_BG, fg=SUBTEXT,
             font=FONT_SMALL).pack(anchor="w", padx=10, pady=(4, 2))
    det_txt = tk.Text(det_frame, bg=CARD_BG, fg=TEXT, font=FONT_NORMAL,
                      relief="flat", wrap="word", padx=10, pady=6, height=3)
    det_txt.pack(fill="both", expand=True, padx=8, pady=(0, 6))

    def _on_select(e: tk.Event) -> None:
        sel = tree.selection()
        if not sel:
            return
        vals = tree.item(sel[0])["values"]
        if vals:
            det_txt.configure(state="normal")
            det_txt.delete("1.0", "end")
            det_txt.insert("end", str(vals[-1]))
            det_txt.configure(state="disabled")

    tree.bind("<<TreeviewSelect>>", _on_select)

    _last: list[str] = [""]

    # ── Render helpers ────────────────────────────────────────

    def _render_summary(query: str, hits: list[Hit]) -> None:
        sum_txt.configure(state="normal")
        sum_txt.delete("1.0", "end")

        if not hits:
            sum_txt.insert("end", "No results found.")
            sum_txt.configure(state="disabled")
            return

        import numpy as np
        drugs = list({h.drug      for h in hits if h.drug})
        conds = list({h.condition for h in hits if h.condition})
        sats  = [h.satisfaction_float for h in hits if h.satisfaction_float]
        effs  = [float(h.effectiveness) for h in hits
                 if h.effectiveness.replace(".", "").isdigit()]
        pos   = sum(1 for h in hits if "sentiment: positive" in h.document.lower())
        neg   = sum(1 for h in hits if "sentiment: negative" in h.document.lower())
        neu   = len(hits) - pos - neg

        lines: list[tuple[str, str]] = [
            ("header",  f"Based on {len(hits)} retrieved reviews:\n"),
            ("label",   "Drugs: "),
            ("value",   ", ".join(drugs) + "\n"),
            ("label",   "Conditions: "),
            ("value",   ", ".join(conds) + "\n"),
        ]
        if sats:
            lines += [("label", "Avg Satisfaction: "),
                      ("value", f"{np.mean(sats):.1f}/5\n")]
        if effs:
            lines += [("label", "Avg Effectiveness: "),
                      ("value", f"{np.mean(effs):.1f}/5\n")]
        lines += [
            ("pos",     f"✅ Positive: {pos}  "),
            ("neutral", f"⚠️ Neutral: {neu}  "),
            ("neg",     f"❌ Negative: {neg}\n"),
            ("divider", "─" * 70 + "\n"),
        ]

        for h in hits:
            sat_f = h.satisfaction_float
            icon  = "✅" if sat_f >= 4 else ("❌" if sat_f <= 2 else "⚠️")
            tag   = "pos" if sat_f >= 4 else ("neg" if sat_f <= 2 else "neutral")
            lines += [
                (tag,      f"\n[{hits.index(h)+1}] {icon}  Sim: {h.similarity:.0%}  |  "
                           f"Sat: {h.satisfaction}/5  |  Eff: {h.effectiveness}/5\n"),
                ("meta",   f"Drug: {h.drug}  |  Condition: {h.condition}  |  "
                           f"Age: {h.age}  |  Sex: {h.sex}\n"),
                ("review", f'"{h.review_text}"\n'),
            ]

        _last[0] = "".join(v for _, v in lines)

        sum_txt.insert("end", "Query: ", "label")
        sum_txt.insert("end", query + "\n", "value")
        sum_txt.insert("end", "═" * 70 + "\n\n", "divider")
        for tag, val in lines:
            sum_txt.insert("end", val, tag)
        sum_txt.configure(state="disabled")

    def _render_table(hits: list[Hit]) -> None:
        tree.delete(*tree.get_children())
        for i, h in enumerate(hits, 1):
            sat_f = h.satisfaction_float
            tag   = "pos_row" if sat_f >= 4 else ("neg_row" if sat_f <= 2 else "")
            tree.insert("", "end", tags=(tag,), values=(
                i, f"{h.similarity:.0%}",
                h.drug[:20], h.condition[:25],
                h.satisfaction, h.effectiveness,
                h.age, h.sex, h.review_text,
            ))
        tree.tag_configure("pos_row", foreground=GREEN)
        tree.tag_configure("neg_row", foreground=RED)

    def _render_llm_loading() -> None:
        llm_txt.configure(state="normal")
        llm_txt.delete("1.0", "end")
        llm_txt.insert("end", "⏳  Generating LLM response...\n", "llm_warn")
        llm_txt.configure(state="disabled")

    def _render_llm(query: str, answer: str | None) -> None:
        llm_txt.configure(state="normal")
        llm_txt.delete("1.0", "end")
        llm_txt.insert("end", "🤖  LLM Answer\n", "llm_head")
        llm_txt.insert("end", "═" * 70 + "\n", "divider")
        llm_txt.insert("end", "Query: ", "llm_key")
        llm_txt.insert("end", query + "\n\n", "llm_body")

        if answer is None:
            llm_txt.insert("end",
                "⚠️  No OPENROUTER_API_KEY found in environment.\n\n"
                "To enable LLM answers, set the variable before running:\n\n"
                "    Windows CMD:        set OPENROUTER_API_KEY=your_key\n"
                "    Windows PowerShell: $env:OPENROUTER_API_KEY='your_key'\n\n"
                "Get a free key at: https://openrouter.ai\n",
                "llm_warn")
        else:
            for line in answer.split("\n"):
                if line.strip().startswith(("**", "##", "#", "---", "===")):
                    llm_txt.insert("end", line + "\n", "llm_key")
                elif line.strip() == "":
                    llm_txt.insert("end", "\n")
                else:
                    llm_txt.insert("end", line + "\n", "llm_body")

        llm_txt.configure(state="disabled")
        res_nb.select(1)   # auto-switch to LLM tab

    # ── Search logic ──────────────────────────────────────────

    def _do_search() -> None:
        query = q_var.get().strip()
        if not query:
            return
        status_var.set("⏳ Searching...")
        search_btn.configure(state="disabled")
        parent.update()

        def _run() -> None:
            try:
                col, emb = get_rag_fn()
                hits = retrieve(
                    query, col, emb,
                    top_k=topk_var.get(),
                    drug_filter=drug_var.get().strip() or None,
                    condition_filter=cond_var.get().strip() or None,
                )
                parent.after(0, lambda: _render_summary(query, hits))
                parent.after(0, lambda: _render_table(hits))
                parent.after(0, lambda: status_var.set(
                    f"✅ {len(hits)} reviews  |  Asking LLM..."))
                parent.after(0, lambda: search_btn.configure(state="normal"))

                threading.Thread(
                    target=_run_llm, args=(query, hits), daemon=True
                ).start()
            except Exception as exc:
                parent.after(0, lambda: status_var.set(f"❌ {exc}"))
                parent.after(0, lambda: search_btn.configure(state="normal"))

        def _run_llm(query: str, hits: list[Hit]) -> None:
            parent.after(0, _render_llm_loading)
            try:
                answer = llm_generate(query, hits, llm_client)
            except Exception as exc:
                answer = f"⚠️ LLM error: {exc}"
            parent.after(0, lambda a=answer: _render_llm(query, a))
            parent.after(0, lambda: status_var.set(f"✅ Done — {len(hits)} reviews"))

        threading.Thread(target=_run, daemon=True).start()

    def _do_copy() -> None:
        if _last[0]:
            parent.clipboard_clear()
            parent.clipboard_append(_last[0])
            status_var.set("✅ Copied!")
            parent.after(2000, lambda: status_var.set("Ready"))

    search_btn.configure(command=_do_search)
    copy_btn.configure(command=_do_copy)
    parent.bind("<Return>", lambda e: _do_search())
    parent.after(600, _do_search)
