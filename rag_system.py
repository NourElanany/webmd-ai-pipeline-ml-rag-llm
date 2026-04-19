# ============================================================
# Phase 4+5: RAG + LLM System — Drug Review Retrieval & Generation
# ChromaDB + SentenceTransformers + OpenRouter LLM
# WebMD Drug Reviews Dataset
# ============================================================

import os
import re
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Dependency check ──────────────────────────────────────
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Run: pip install chromadb sentence-transformers openai")

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — values read from os.environ directly

CHROMA_DIR  = "chroma_db"
COLLECTION  = "webmd_reviews"
EMBED_MODEL = "all-MiniLM-L6-v2"
SAMPLE_SIZE = 50000

# ── LLM config — read from .env, never hardcoded ─────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL          = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

# ============================================================
# 1. Load & Prepare
# ============================================================
def load_data(path="webmd_cleaned.csv", sample_size=SAMPLE_SIZE):
    df = pd.read_csv(path)
    df = df[df["Reviews"].str.strip().str.len() > 20].copy()
    df = df.dropna(subset=["Drug", "Condition", "Reviews"])
    df["Drug"]      = df["Drug"].str.strip().str.lower()
    df["Condition"] = df["Condition"].str.strip().str.lower()
    df["Sides"]     = df["Sides"].fillna("not reported")

    # Sample for speed — stratify by Drug to keep coverage
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    print(f"Loaded {len(df):,} reviews | {df['Drug'].nunique():,} drugs | {df['Condition'].nunique():,} conditions")
    return df


def build_document(row):
    """Combine fields into a rich text document for embedding."""
    sentiment = "positive" if row["Satisfaction"] >= 4 else (
                "neutral"  if row["Satisfaction"] == 3 else "negative")
    return (
        f"Drug: {row['Drug']}. "
        f"Condition: {row['Condition']}. "
        f"Sentiment: {sentiment}. "
        f"Effectiveness: {row['Effectiveness']}/5. "
        f"Side effects: {row['Sides']}. "
        f"Review: {row['Reviews']}"
    )


# ============================================================
# 2. Build / Load Vector Index
# ============================================================
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def build_index(df, force_rebuild=False):
    client = get_chroma_client()

    # Check if index already exists
    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing and not force_rebuild:
        col = client.get_collection(COLLECTION)
        print(f"Loaded existing index: {col.count():,} documents")
        return col

    # Delete old collection if rebuilding
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)

    print(f"Building index with {EMBED_MODEL}...")
    embedder = SentenceTransformer(EMBED_MODEL)

    col = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )

    docs      = df.apply(build_document, axis=1).tolist()
    ids       = [str(i) for i in df.index]
    metadatas = df[["Drug", "Condition", "Satisfaction", "Effectiveness", "Sides", "Sex", "Age"]].to_dict("records")
    # Chroma needs string values in metadata
    metadatas = [{k: str(v) for k, v in m.items()} for m in metadatas]

    # Batch embed (Chroma accepts embeddings directly)
    BATCH = 512
    for start in range(0, len(docs), BATCH):
        end   = min(start + BATCH, len(docs))
        batch_docs  = docs[start:end]
        batch_ids   = ids[start:end]
        batch_meta  = metadatas[start:end]
        batch_emb   = embedder.encode(batch_docs, show_progress_bar=False).tolist()
        col.add(documents=batch_docs, embeddings=batch_emb,
                ids=batch_ids, metadatas=batch_meta)
        print(f"  Indexed {end:,}/{len(docs):,}", end="\r")

    print(f"\nIndex built: {col.count():,} documents saved to '{CHROMA_DIR}/'")
    return col


# ============================================================
# 3. Retrieval
# ============================================================
def retrieve(query, col, embedder, top_k=5, drug_filter=None, condition_filter=None):
    """
    Semantic search with optional metadata filters.
    Returns list of result dicts.
    """
    where = {}
    if drug_filter:
        where["Drug"] = {"$eq": drug_filter.strip().lower()}
    if condition_filter:
        where["Condition"] = {"$eq": condition_filter.strip().lower()}

    query_emb = embedder.encode([query]).tolist()

    kwargs = dict(
        query_embeddings=query_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = col.query(**kwargs)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "document": doc,
            "drug":        meta.get("Drug", ""),
            "condition":   meta.get("Condition", ""),
            "satisfaction": meta.get("Satisfaction", ""),
            "effectiveness": meta.get("Effectiveness", ""),
            "sides":       meta.get("Sides", ""),
            "age":         meta.get("Age", ""),
            "sex":         meta.get("Sex", ""),
            "similarity":  round(1 - dist, 3),   # cosine distance → similarity
        })
    return hits


def generate_response(query, hits):
    if not hits:
        return "لم يتم العثور على مراجعات مطابقة لطلبك."

    drugs      = list({h["drug"] for h in hits if h["drug"]})
    conditions = list({h["condition"] for h in hits if h["condition"]})
    avg_sat    = np.mean([float(h["satisfaction"]) for h in hits if h["satisfaction"].replace(".","").isdigit()])
    avg_eff    = np.mean([float(h["effectiveness"]) for h in hits if h["effectiveness"].replace(".","").isdigit()])

    all_sides = []
    for h in hits:
        sides = h["sides"]
        if sides and sides.lower() not in ("not reported", "nan", ""):
            all_sides.append(sides)

    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for h in hits:
        doc = h["document"].lower()
        if "sentiment: positive" in doc:   sentiment_counts["positive"] += 1
        elif "sentiment: negative" in doc: sentiment_counts["negative"] += 1
        else:                              sentiment_counts["neutral"]  += 1

    lines = [
        f"بناءً على {len(hits)} مراجعة مشابهة لسؤالك:\n",
        f"الدواء/الأدوية: {', '.join(drugs) or 'متنوعة'}",
        f"الحالة المرضية: {', '.join(conditions) or 'متنوعة'}",
        f"متوسط الرضا: {avg_sat:.1f}/5  |  متوسط الفعالية: {avg_eff:.1f}/5",
        f"التقييمات: ✅ إيجابية: {sentiment_counts['positive']}  "
        f"⚠️ محايدة: {sentiment_counts['neutral']}  "
        f"❌ سلبية: {sentiment_counts['negative']}",
    ]

    if all_sides:
        lines.append(f"\nالآثار الجانبية المذكورة:\n  • " + "\n  • ".join(all_sides[:5]))

    # Show ALL retrieved reviews, not just top 3
    lines.append(f"\n── جميع المراجعات المسترجعة ({len(hits)}) ──")
    for i, h in enumerate(hits, 1):
        review_part = re.search(r"Review: (.+)", h["document"])
        review_text = review_part.group(1) if review_part else h["document"]
        sat = h["satisfaction"]
        sentiment_icon = "✅" if float(sat) >= 4 else ("❌" if float(sat) <= 2 else "⚠️")
        lines.append(
            f"\n[{i}] {sentiment_icon}  تشابه: {h['similarity']:.0%}  |  "
            f"رضا: {sat}/5  |  فعالية: {h['effectiveness']}/5  |  "
            f"عمر: {h['age']}  |  جنس: {h['sex']}\n"
            f"الدواء: {h['drug']}  |  الحالة: {h['condition']}\n"
            f"\"{review_text}\""
        )

    return "\n".join(lines)


# ============================================================
# LLM Generation via OpenRouter
# ============================================================
def build_llm_client(api_key: str):
    if not _openai_available:
        return None
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def llm_generate(query: str, hits: list, client, model: str = LLM_MODEL) -> str:
    """
    Send retrieved reviews as context to the LLM and get a medical summary.
    """
    if not client or not hits:
        return generate_response(query, hits)

    # Build context from retrieved reviews
    context_parts = []
    for i, h in enumerate(hits, 1):
        review_part = re.search(r"Review: (.+)", h["document"])
        review_text = review_part.group(1) if review_part else h["document"]
        sat = h["satisfaction"]
        sentiment = "إيجابية" if float(sat) >= 4 else ("سلبية" if float(sat) <= 2 else "محايدة")
        context_parts.append(
            f"[مراجعة {i}] الدواء: {h['drug']} | الحالة: {h['condition']} | "
            f"الرضا: {sat}/5 | التقييم: {sentiment} | "
            f"الآثار الجانبية: {h['sides']}\n"
            f"النص: {review_text}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """أنت مساعد طبي متخصص في تحليل تجارب المرضى مع الأدوية.
مهمتك: تلخيص تجارب المرضى الحقيقية بناءً على المراجعات المقدمة لك.

قواعد مهمة:
- استند فقط على المراجعات المقدمة، لا تخترع معلومات
- اذكر الآثار الجانبية الموجودة في المراجعات بوضوح
- وضح نسبة الرضا العامة
- استخدم لغة طبية بسيطة ومفهومة
- نبّه دائماً أن هذه تجارب شخصية وليست نصيحة طبية"""

    user_prompt = f"""سؤال المريض: {query}

المراجعات المسترجعة ({len(hits)} مراجعة):
{context}

المطلوب: اكتب ملخصاً طبياً دقيقاً يجيب على سؤال المريض بناءً على هذه التجارب الحقيقية."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        llm_text = response.choices[0].message.content.strip()
        # Append the structured summary below the LLM response
        structured = generate_response(query, hits)
        return f"{llm_text}\n\n{'═'*60}\n📊 البيانات الإحصائية:\n{'═'*60}\n{structured}"
    except Exception as e:
        # Fallback to template if LLM fails
        return f"⚠️ LLM غير متاح ({e})\n\n{generate_response(query, hits)}"


# ============================================================
# 4. GUI
# ============================================================
def launch_gui(col, embedder, df, llm_client=None):
    import tkinter as tk
    from tkinter import ttk

    DARK_BG  = "#1e1e2e"; PANEL_BG = "#2a2a3e"; ACCENT  = "#7c6af7"
    TEXT     = "#cdd6f4"; SUBTEXT  = "#a6adc8";  CARD_BG = "#313244"
    GREEN    = "#a6e3a1"; RED      = "#f38ba8";  BORDER  = "#45475a"

    root = tk.Tk()
    root.title("WebMD RAG — Drug Review Retrieval")
    root.geometry("1280x820")
    root.configure(bg=DARK_BG)
    root.minsize(900, 600)

    # ── Styles ───────────────────────────────────────────────
    style = ttk.Style(); style.theme_use("clam")
    style.configure("TNotebook",        background=DARK_BG,  borderwidth=0)
    style.configure("TNotebook.Tab",    background=PANEL_BG, foreground=TEXT,
                    padding=[16, 7],    font=("Segoe UI", 10))
    style.map("TNotebook.Tab",
              background=[("selected", ACCENT)], foreground=[("selected", "#fff")])
    style.configure("TFrame",           background=DARK_BG)
    style.configure("Treeview",         background=CARD_BG,  foreground=TEXT,
                    fieldbackground=CARD_BG, rowheight=30,   font=("Segoe UI", 9))
    style.configure("Treeview.Heading", background=PANEL_BG, foreground=ACCENT,
                    font=("Segoe UI", 9, "bold"))
    style.map("Treeview",               background=[("selected", ACCENT)],
                                        foreground=[("selected", "#fff")])
    style.configure("Vertical.TScrollbar", background=PANEL_BG, troughcolor=DARK_BG,
                    arrowcolor=SUBTEXT, borderwidth=0)

    # ── Header ───────────────────────────────────────────────
    hdr = tk.Frame(root, bg=ACCENT, height=56); hdr.pack(fill="x")
    hdr.pack_propagate(False)
    tk.Label(hdr, text="  🔍  WebMD RAG + LLM  |  Drug Review Retrieval & Generation",
             bg=ACCENT, fg="#fff", font=("Segoe UI", 14, "bold")).pack(side="left", padx=16)
    llm_status = "� LLM متصل" if llm_client else "🔴 LLM غير متصل"
    llm_color  = "#a6e3a1" if llm_client else "#f38ba8"
    tk.Label(hdr, text=llm_status, bg=ACCENT, fg=llm_color,
             font=("Segoe UI", 10, "bold")).pack(side="right", padx=8)
    tk.Label(hdr, text=f"📦 {col.count():,} reviews  •  🤖 {EMBED_MODEL}  •  Phase 5 / 5",
             bg=ACCENT, fg="#e0e0ff", font=("Segoe UI", 10)).pack(side="right", padx=16)

    # ── Search Panel ─────────────────────────────────────────
    search_frame = tk.Frame(root, bg=PANEL_BG, pady=12, padx=16)
    search_frame.pack(fill="x")

    # Row 0: query
    tk.Label(search_frame, text="السؤال:", bg=PANEL_BG, fg=SUBTEXT,
             font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0,10))
    query_var = tk.StringVar()
    query_entry = tk.Entry(search_frame, textvariable=query_var, bg=CARD_BG, fg=TEXT,
                           insertbackground=TEXT, font=("Segoe UI", 11),
                           relief="flat", bd=0, highlightthickness=1,
                           highlightbackground=BORDER, highlightcolor=ACCENT)
    query_entry.grid(row=0, column=1, columnspan=4, sticky="ew", ipady=8)
    query_var.set("What are people's experiences with ibuprofen for headache? Does it cause insomnia?")

    # Explicit paste binding — fixes Ctrl+V on Windows
    def _paste(event):
        try:
            text = root.clipboard_get()
            widget = event.widget
            try:
                sel_start = widget.index(tk.SEL_FIRST)
                sel_end   = widget.index(tk.SEL_LAST)
                widget.delete(sel_start, sel_end)
            except tk.TclError:
                pass
            widget.insert(tk.INSERT, text)
        except tk.TclError:
            pass
        return "break"

    query_entry.bind("<Control-v>", _paste)
    query_entry.bind("<Control-V>", _paste)

    # Row 1: filters
    def lbl(text, r, c): tk.Label(search_frame, text=text, bg=PANEL_BG, fg=SUBTEXT,
                                   font=("Segoe UI", 9)).grid(row=r, column=c, sticky="w",
                                                               padx=(0,6), pady=(8,0))
    def ent(var, r, c, w=22):
        e = tk.Entry(search_frame, textvariable=var, bg=CARD_BG, fg=TEXT,
                     insertbackground=TEXT, font=("Segoe UI", 10), relief="flat",
                     bd=0, highlightthickness=1, highlightbackground=BORDER,
                     highlightcolor=ACCENT, width=w)
        e.grid(row=r, column=c, sticky="w", pady=(8,0), ipady=4)
        return e

    drug_var = tk.StringVar(); cond_var = tk.StringVar(); topk_var = tk.IntVar(value=7)
    lbl("دواء (اختياري):", 1, 0);      drug_ent = ent(drug_var, 1, 1)
    lbl("حالة مرضية (اختياري):", 1, 2); cond_ent = ent(cond_var, 1, 3)
    drug_ent.bind("<Control-v>", _paste)
    drug_ent.bind("<Control-V>", _paste)
    cond_ent.bind("<Control-v>", _paste)
    cond_ent.bind("<Control-V>", _paste)
    lbl("عدد النتائج:", 1, 4)
    tk.Spinbox(search_frame, from_=1, to=20, textvariable=topk_var, width=4,
               bg=CARD_BG, fg=TEXT, buttonbackground=PANEL_BG, relief="flat",
               font=("Segoe UI", 10)).grid(row=1, column=5, sticky="w", pady=(8,0), padx=(4,0))

    search_frame.columnconfigure(1, weight=1)
    search_frame.columnconfigure(3, weight=1)

    # ── Action bar ───────────────────────────────────────────
    action_bar = tk.Frame(root, bg=DARK_BG, pady=6, padx=8)
    action_bar.pack(fill="x")

    status_var = tk.StringVar(value="جاهز للبحث")

    search_btn = tk.Button(action_bar, text="  🔍  بحث  ", bg=ACCENT, fg="#fff",
                           font=("Segoe UI", 11, "bold"), relief="flat",
                           padx=18, pady=5, cursor="hand2", activebackground="#6355d4",
                           activeforeground="#fff")
    search_btn.pack(side="left")

    copy_btn = tk.Button(action_bar, text="  📋  نسخ الملخص  ", bg=CARD_BG, fg=TEXT,
                         font=("Segoe UI", 10), relief="flat",
                         padx=14, pady=5, cursor="hand2", activebackground=ACCENT,
                         activeforeground="#fff")
    copy_btn.pack(side="left", padx=8)

    clear_btn = tk.Button(action_bar, text="  ✕  مسح  ", bg=CARD_BG, fg=SUBTEXT,
                          font=("Segoe UI", 10), relief="flat",
                          padx=14, pady=5, cursor="hand2")
    clear_btn.pack(side="left")

    tk.Label(action_bar, textvariable=status_var, bg=DARK_BG, fg=SUBTEXT,
             font=("Segoe UI", 9)).pack(side="left", padx=14)

    # ── Notebook ─────────────────────────────────────────────
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # ── Tab 1: Summary ───────────────────────────────────────
    tab_sum = ttk.Frame(nb); nb.add(tab_sum, text="  📊  الملخص  ")

    sum_text = tk.Text(tab_sum, bg=CARD_BG, fg=TEXT, font=("Segoe UI", 11),
                       relief="flat", wrap="word", padx=20, pady=14,
                       insertbackground=TEXT, selectbackground=ACCENT,
                       selectforeground="#fff", spacing1=2, spacing3=2)
    # Tags for coloured sections
    sum_text.tag_configure("header",  foreground="#cba6f7", font=("Segoe UI", 12, "bold"))
    sum_text.tag_configure("label",   foreground="#89b4fa", font=("Segoe UI", 10, "bold"))
    sum_text.tag_configure("value",   foreground="#cdd6f4", font=("Segoe UI", 10))
    sum_text.tag_configure("pos",     foreground="#a6e3a1", font=("Segoe UI", 10, "bold"))
    sum_text.tag_configure("neg",     foreground="#f38ba8", font=("Segoe UI", 10, "bold"))
    sum_text.tag_configure("neutral", foreground="#f9e2af", font=("Segoe UI", 10, "bold"))
    sum_text.tag_configure("review",  foreground="#f5c2e7", font=("Segoe UI", 10))
    sum_text.tag_configure("meta",    foreground="#94e2d5", font=("Segoe UI", 9))
    sum_text.tag_configure("divider", foreground="#45475a", font=("Segoe UI", 8))
    sum_text.tag_configure("side",    foreground="#fab387", font=("Segoe UI", 10))

    sb1 = ttk.Scrollbar(tab_sum, command=sum_text.yview)
    sum_text.configure(yscrollcommand=sb1.set)
    sb1.pack(side="right", fill="y"); sum_text.pack(fill="both", expand=True)

    # ── Tab 2: Reviews Table ─────────────────────────────────
    tab_raw = ttk.Frame(nb); nb.add(tab_raw, text="  📋  المراجعات المسترجعة  ")

    tree_frame = tk.Frame(tab_raw, bg=DARK_BG); tree_frame.pack(fill="both", expand=True)
    cols = ["#", "تشابه", "دواء", "حالة مرضية", "رضا", "فعالية", "عمر", "جنس", "المراجعة"]
    tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")
    widths = [35, 75, 110, 140, 55, 70, 85, 55, 0]
    for c, w in zip(cols, widths):
        tree.heading(c, text=c)
        if w == 0:
            tree.column(c, stretch=True, minwidth=300, anchor="w")
        else:
            tree.column(c, width=w, stretch=False,
                        anchor="center" if w < 120 else "w")

    sb2x = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
    sb2y = ttk.Scrollbar(tree_frame, command=tree.yview)
    tree.configure(yscrollcommand=sb2y.set, xscrollcommand=sb2x.set)
    sb2y.pack(side="right", fill="y")
    sb2x.pack(side="bottom", fill="x")
    tree.pack(fill="both", expand=True)

    # Detail panel below table
    detail_frame = tk.Frame(tab_raw, bg=PANEL_BG, height=120)
    detail_frame.pack(fill="x"); detail_frame.pack_propagate(False)
    tk.Label(detail_frame, text="المراجعة الكاملة:", bg=PANEL_BG, fg=SUBTEXT,
             font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=12, pady=(6,2))
    detail_text = tk.Text(detail_frame, bg=CARD_BG, fg=TEXT, font=("Segoe UI", 10),
                          relief="flat", wrap="word", padx=10, pady=6, height=4,
                          selectbackground=ACCENT, selectforeground="#fff")
    detail_text.pack(fill="both", expand=True, padx=8, pady=(0,8))

    def on_tree_select(event):
        sel = tree.selection()
        if not sel: return
        item = tree.item(sel[0])["values"]
        if item:
            detail_text.configure(state="normal")
            detail_text.delete("1.0", "end")
            detail_text.insert("end", str(item[-1]))
            detail_text.configure(state="disabled")

    tree.bind("<<TreeviewSelect>>", on_tree_select)

    # ── Search logic ─────────────────────────────────────────
    _last_response = [""]

    def do_search():
        query = query_var.get().strip()
        if not query: return
        status_var.set("⏳ جاري البحث...")
        search_btn.configure(state="disabled")
        root.update()

        try:
            hits = retrieve(
                query, col, embedder,
                top_k=topk_var.get(),
                drug_filter=drug_var.get().strip() or None,
                condition_filter=cond_var.get().strip() or None,
            )
            _render_summary(query, hits, use_llm=llm_client is not None)
            _render_table(hits)
            status_var.set(f"✅ تم استرجاع {len(hits)} مراجعة")
        except Exception as ex:
            status_var.set(f"❌ خطأ: {ex}")
        finally:
            search_btn.configure(state="normal")

    def _render_summary(query, hits, use_llm=False):
        if use_llm:
            status_var.set("⏳ الـ LLM بيولّد الإجابة...")
            root.update()
            response = llm_generate(query, hits, llm_client)
        else:
            response = generate_response(query, hits)
        _last_response[0] = response

        sum_text.configure(state="normal")
        sum_text.delete("1.0", "end")

        sum_text.insert("end", "السؤال: ", "label")
        sum_text.insert("end", f"{query}\n", "value")
        sum_text.insert("end", "═" * 72 + "\n\n", "divider")

        for line in response.split("\n"):
            if line.startswith("بناءً على") or line.startswith("──"):
                sum_text.insert("end", line + "\n", "header")
            elif line.startswith("[") and "]" in line:
                # Review header line — colour by sentiment icon
                if "✅" in line:
                    sum_text.insert("end", line + "\n", "pos")
                elif "❌" in line:
                    sum_text.insert("end", line + "\n", "neg")
                else:
                    sum_text.insert("end", line + "\n", "neutral")
            elif line.startswith("الدواء:"):
                sum_text.insert("end", line + "\n", "meta")
            elif line.startswith('"') or line.endswith('"'):
                sum_text.insert("end", line + "\n", "review")
            elif "✅" in line or "إيجابية" in line:
                sum_text.insert("end", line + "\n", "pos")
            elif "❌" in line or "سلبية" in line:
                sum_text.insert("end", line + "\n", "neg")
            elif line.startswith("  •"):
                sum_text.insert("end", line + "\n", "side")
            elif ":" in line:
                parts = line.split(":", 1)
                sum_text.insert("end", parts[0] + ":", "label")
                sum_text.insert("end", parts[1] + "\n", "value")
            else:
                sum_text.insert("end", line + "\n")

        sum_text.configure(state="disabled")

    def _render_table(hits):
        tree.delete(*tree.get_children())
        for i, h in enumerate(hits, 1):
            review_part = re.search(r"Review: (.+)", h["document"])
            review_text = review_part.group(1) if review_part else h["document"]
            sat = float(h["satisfaction"]) if h["satisfaction"].replace(".","").isdigit() else 0
            tag = "pos_row" if sat >= 4 else ("neg_row" if sat <= 2 else "")
            tree.insert("", "end", tags=(tag,), values=(
                i,
                f"{h['similarity']:.0%}",
                h["drug"][:20],
                h["condition"][:25],
                h["satisfaction"],
                h["effectiveness"],
                h["age"],
                h["sex"],
                review_text,
            ))
        tree.tag_configure("pos_row", foreground=GREEN)
        tree.tag_configure("neg_row", foreground=RED)

    def do_copy():
        text = _last_response[0]
        if text:
            root.clipboard_clear()
            root.clipboard_append(text)
            status_var.set("✅ تم النسخ!")
            root.after(2000, lambda: status_var.set("جاهز"))

    def do_clear():
        query_var.set(""); drug_var.set(""); cond_var.set("")
        sum_text.configure(state="normal")
        sum_text.delete("1.0", "end")
        sum_text.configure(state="disabled")
        tree.delete(*tree.get_children())
        detail_text.configure(state="normal")
        detail_text.delete("1.0", "end")
        detail_text.configure(state="disabled")
        status_var.set("جاهز للبحث")

    search_btn.configure(command=do_search)
    copy_btn.configure(command=do_copy)
    clear_btn.configure(command=do_clear)
    root.bind("<Return>", lambda e: do_search())

    root.after(400, do_search)
    root.mainloop()



# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    df       = load_data()
    col      = build_index(df)
    embedder = SentenceTransformer(EMBED_MODEL)

    # Load API key from environment variable (never hardcode it)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    llm_client = build_llm_client(api_key) if api_key else None
    if llm_client:
        print(f"LLM ready: {LLM_MODEL}")
    else:
        print("No OPENROUTER_API_KEY found — running without LLM (template mode)")

    launch_gui(col, embedder, df, llm_client)
