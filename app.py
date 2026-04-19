# ============================================================
# Final Project: ML + DL + LLM + RAG — Unified Dashboard
# WebMD Drug Reviews — All Phases Integration
# ============================================================

import os, re, warnings, threading
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — fall back to system env vars

# ── Theme ────────────────────────────────────────────────────
DARK_BG  = "#1e1e2e"
PANEL_BG = "#2a2a3e"
ACCENT   = "#7c6af7"
ACCENT2  = "#cba6f7"
TEXT     = "#cdd6f4"
SUBTEXT  = "#a6adc8"
CARD_BG  = "#313244"
GREEN    = "#a6e3a1"
RED      = "#f38ba8"
YELLOW   = "#f9e2af"
BORDER   = "#45475a"
BLUE     = "#89b4fa"
ORANGE   = "#fab387"

FONT_TITLE  = ("Segoe UI", 14, "bold")
FONT_BOLD   = ("Segoe UI", 10, "bold")
FONT_NORMAL = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)

# ── Plot paths ───────────────────────────────────────────────
EDA_PLOTS = {
    "Ratings Distribution":  "plots/plot1_ratings_distribution.png",
    "Gender Distribution":   "plots/plot2_gender_distribution.png",
    "Age Distribution":      "plots/plot3_age_distribution.png",
    "Top Conditions":        "plots/plot4_top_conditions.png",
    "Top Drugs Ratings":     "plots/plot5_top_drugs_ratings.png",
    "Reviews Over Years":    "plots/plot6_reviews_over_years.png",
    "Correlation Heatmap":   "plots/plot7_correlation_heatmap.png",
    "Review Length Dist.":   "plots/plot8_review_length_dist.png",
}
ML_PLOTS = {
    "Model Comparison":      "ml_plots/plot1_model_comparison.png",
    "Confusion Matrix":      "ml_plots/plot2_confusion_matrix.png",
    "Feature Importance":    "ml_plots/plot3_feature_importance.png",
    "Per-Class Metrics":     "ml_plots/plot4_per_class_metrics.png",
    "Actual vs Predicted":   "ml_plots/plot5_actual_vs_predicted.png",
}
NLP_PLOTS = {
    "Training History":           "nlp_plots/plot1_training_history.png",
    "Model Comparison":           "nlp_plots/plot2_model_comparison.png",
    "Confusion Matrix":           "nlp_plots/plot3_confusion_matrix.png",
    "Top Side Effects":           "nlp_plots/plot4_top_side_effects.png",
    "Side Effects by Sentiment":  "nlp_plots/plot5_side_effects_by_sentiment.png",
    "Confidence Distribution":    "nlp_plots/plot6_confidence_distribution.png",
}

# ============================================================
# Lazy model loaders (load only when tab is first opened)
# ============================================================
_cache = {}

def get_df():
    if "df" not in _cache:
        _cache["df"] = pd.read_csv("webmd_cleaned.csv")
    return _cache["df"]

def get_ml_model():
    if "ml" not in _cache:
        import joblib
        from sklearn.preprocessing import LabelEncoder
        df = get_df()
        top_conditions = df["Condition"].value_counts().head(50).index
        df2 = df.copy()
        df2["Condition_Clean"] = df2["Condition"].where(df2["Condition"].isin(top_conditions), other="Other")
        le = LabelEncoder()
        le.fit(df2["Condition_Clean"])
        _cache["le"] = le
        _cache["ml"] = joblib.load("rf_effectiveness_model.pkl")
    return _cache["ml"], _cache["le"]

def get_nlp_models():
    if "lstm" not in _cache:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from scipy.sparse import hstack
        import re as _re

        df = get_df()
        df = df[df["Reviews"].str.strip().str.len() > 10].copy()
        df = df[df["Satisfaction"] != 3].copy()
        df["Sentiment"] = (df["Satisfaction"] >= 4).astype(int)
        df = df.groupby("Sentiment", group_keys=False).apply(
            lambda x: x.sample(min(len(x), 40000), random_state=42)
        ).reset_index(drop=True)

        def _clean(text):
            text = str(text).lower()
            text = _re.sub(r"<[^>]+>", " ", text)
            text = _re.sub(r"n't", " not", text)
            text = _re.sub(r"[^a-z\s]", " ", text)
            return " ".join(w for w in text.split() if len(w) > 1)

        df["Clean"] = df["Reviews"].apply(_clean)
        _cache["clean_fn"] = _clean

        VOCAB, MAXLEN = 30000, 150
        tok = Tokenizer(num_words=VOCAB, oov_token="<OOV>", lower=True)
        tok.fit_on_texts(df["Clean"])
        _cache["tokenizer"] = tok
        _cache["maxlen"]    = MAXLEN

        wv = TfidfVectorizer(ngram_range=(1,3), max_features=100000, sublinear_tf=True, min_df=2)
        cv = TfidfVectorizer(ngram_range=(2,4), max_features=100000, sublinear_tf=True, min_df=3, analyzer="char_wb")
        X = hstack([wv.fit_transform(df["Clean"]), cv.fit_transform(df["Clean"])])
        y = df["Sentiment"].values

        svc = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000), cv=3)
        lr  = LogisticRegression(C=5.0, max_iter=500, solver="saga", n_jobs=-1)
        svc.fit(X, y); lr.fit(X, y)
        _cache["wv"] = wv; _cache["cv_"] = cv
        _cache["svc"] = svc; _cache["lr"] = lr
        _cache["hstack"] = hstack
        _cache["lstm"] = load_model("lstm_sentiment_model.keras", compile=False)

    return (_cache["lstm"], _cache["tokenizer"], _cache["maxlen"],
            _cache["svc"], _cache["lr"], _cache["wv"], _cache["cv_"],
            _cache["hstack"], _cache["clean_fn"])

def get_rag():
    if "rag_col" not in _cache:
        import chromadb
        from sentence_transformers import SentenceTransformer
        client = chromadb.PersistentClient(path="chroma_db")
        col    = client.get_collection("webmd_reviews")
        emb    = SentenceTransformer("all-MiniLM-L6-v2")
        _cache["rag_col"] = col
        _cache["rag_emb"] = emb
    return _cache["rag_col"], _cache["rag_emb"]

# ============================================================
# Helper widgets
# ============================================================
def apply_style(style):
    style.theme_use("clam")
    style.configure("TNotebook",         background=DARK_BG,  borderwidth=0)
    style.configure("TNotebook.Tab",     background=PANEL_BG, foreground=TEXT,
                    padding=[16, 7],     font=FONT_BOLD)
    style.map("TNotebook.Tab",
              background=[("selected", ACCENT)], foreground=[("selected", "#fff")])
    style.configure("TFrame",            background=DARK_BG)
    style.configure("Treeview",          background=CARD_BG,  foreground=TEXT,
                    fieldbackground=CARD_BG, rowheight=28,    font=FONT_MONO)
    style.configure("Treeview.Heading",  background=PANEL_BG, foreground=ACCENT,
                    font=FONT_BOLD)
    style.map("Treeview",                background=[("selected", ACCENT)],
                                         foreground=[("selected", "#fff")])
    style.configure("Vertical.TScrollbar",   background=PANEL_BG, troughcolor=DARK_BG,
                    arrowcolor=SUBTEXT, borderwidth=0)
    style.configure("Horizontal.TScrollbar", background=PANEL_BG, troughcolor=DARK_BG,
                    arrowcolor=SUBTEXT, borderwidth=0)

def scrolled_text(parent, **kw):
    frame = tk.Frame(parent, bg=DARK_BG)
    txt = tk.Text(frame, bg=CARD_BG, fg=TEXT, font=FONT_NORMAL,
                  relief="flat", wrap="word", padx=14, pady=10,
                  insertbackground=TEXT, selectbackground=ACCENT,
                  selectforeground="#fff", **kw)
    sb = ttk.Scrollbar(frame, command=txt.yview)
    txt.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    txt.pack(fill="both", expand=True)
    return frame, txt

def card_label(parent, title, value, color=ACCENT, col=0, row=0, colspan=1):
    f = tk.Frame(parent, bg=CARD_BG, padx=16, pady=12)
    f.grid(row=row, column=col, columnspan=colspan, padx=8, pady=6, sticky="nsew")
    tk.Label(f, text=title, bg=CARD_BG, fg=SUBTEXT, font=FONT_SMALL).pack(anchor="w")
    tk.Label(f, text=value, bg=CARD_BG, fg=color,   font=("Segoe UI", 15, "bold")).pack(anchor="w")

def section_label(parent, text):
    tk.Label(parent, text=text, bg=DARK_BG, fg=ACCENT2,
             font=FONT_BOLD).pack(anchor="w", padx=16, pady=(14, 4))

def btn(parent, text, cmd, bg=ACCENT, fg="#fff", **kw):
    return tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                     font=FONT_BOLD, relief="flat", cursor="hand2",
                     activebackground="#6355d4", activeforeground="#fff",
                     padx=16, pady=6, **kw)

def plot_viewer(parent, plots_dict):
    """Reusable sidebar + image viewer."""
    left = tk.Frame(parent, bg=PANEL_BG, width=200)
    left.pack(side="left", fill="y"); left.pack_propagate(False)
    tk.Label(left, text="Charts", bg=PANEL_BG, fg=ACCENT,
             font=FONT_BOLD).pack(pady=(14, 6), padx=10)

    right = tk.Frame(parent, bg=DARK_BG)
    right.pack(side="right", fill="both", expand=True)
    img_lbl = tk.Label(right, bg=DARK_BG)
    img_lbl.pack(fill="both", expand=True, padx=8, pady=8)

    _ref = [None]; _cur = [None]

    def show(path):
        _cur[0] = path
        try:
            img = Image.open(path)
            w = max(right.winfo_width() - 16, 800)
            h = max(right.winfo_height() - 16, 500)
            img.thumbnail((w, h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            _ref[0] = photo
            img_lbl.configure(image=photo, text="")
        except Exception as ex:
            img_lbl.configure(text=str(ex), fg=RED, image="")

    right.bind("<Configure>", lambda e: _cur[0] and show(_cur[0]))

    bkw = dict(bg=CARD_BG, fg=TEXT, activebackground=ACCENT, activeforeground="#fff",
               relief="flat", font=FONT_SMALL, cursor="hand2", anchor="w", padx=10, pady=5)
    first = [None]
    for name, path in plots_dict.items():
        if os.path.exists(path):
            tk.Button(left, text=f"  {name}", **bkw,
                      command=lambda p=path: show(p)).pack(fill="x", padx=6, pady=2)
            if first[0] is None: first[0] = path

    if first[0]:
        left.after(300, lambda: show(first[0]))

# ============================================================
# TAB 1 — EDA (Phase 1)
# ============================================================
def build_eda_tab(nb):
    tab = ttk.Frame(nb); nb.add(tab, text="  📊  EDA  ")

    # Stats cards
    top = tk.Frame(tab, bg=DARK_BG); top.pack(fill="x", padx=12, pady=10)
    for i in range(5): top.columnconfigure(i, weight=1)

    def load_stats():
        try:
            df = get_df()
            card_label(top, "Total Reviews",    f"{len(df):,}",                       ACCENT,  0, 0)
            card_label(top, "Unique Drugs",      f"{df['Drug'].nunique():,}",          BLUE,    1, 0)
            card_label(top, "Unique Conditions", f"{df['Condition'].nunique():,}",     YELLOW,  2, 0)
            card_label(top, "Avg Satisfaction",  f"{df['Satisfaction'].mean():.2f}/5", GREEN,   3, 0)
            card_label(top, "Avg Effectiveness", f"{df['Effectiveness'].mean():.2f}/5",ORANGE,  4, 0)
        except Exception as e:
            tk.Label(top, text=f"Load error: {e}", bg=DARK_BG, fg=RED).grid(row=0, column=0)

    threading.Thread(target=load_stats, daemon=True).start()

    # Plot viewer
    viz = tk.Frame(tab, bg=DARK_BG); viz.pack(fill="both", expand=True, padx=8)
    plot_viewer(viz, EDA_PLOTS)


# ============================================================
# TAB 2 — ML (Phase 2)
# ============================================================
AGE_MAP = {
    "0-2":1,"3-6":4,"7-12":9,"13-18":15,"19-24":21,
    "25-34":29,"35-44":39,"45-54":49,"55-64":59,"65-74":69,"75 or over":80
}

def build_ml_tab(nb):
    tab = ttk.Frame(nb); nb.add(tab, text="  🤖  ML  ")
    inner_nb = ttk.Notebook(tab); inner_nb.pack(fill="both", expand=True, padx=6, pady=6)

    # ── Sub-tab: Plots ───────────────────────────────────────
    plots_tab = ttk.Frame(inner_nb); inner_nb.add(plots_tab, text="  Plots  ")
    plot_viewer(plots_tab, ML_PLOTS)

    # ── Sub-tab: Live Predictor ──────────────────────────────
    pred_tab = ttk.Frame(inner_nb); inner_nb.add(pred_tab, text="  Live Predictor  ")

    outer = tk.Frame(pred_tab, bg=DARK_BG); outer.pack(expand=True, pady=20)
    tk.Label(outer, text="Predict Drug Effectiveness", bg=DARK_BG, fg=TEXT,
             font=FONT_TITLE).grid(row=0, column=0, columnspan=2, pady=(0,16))

    fields = {}
    defs = [
        ("Age Group",    "combo", list(AGE_MAP.keys())),
        ("Sex",          "combo", ["Male","Female"]),
        ("Ease of Use",  "combo", ["1","2","3","4","5"]),
        ("Satisfaction", "combo", ["1","2","3","4","5"]),
        ("Useful Count", "entry", "5"),
        ("Year",         "entry", "2020"),
        ("Review Length","entry", "200"),
    ]
    for i, (lbl_txt, ftype, opts) in enumerate(defs):
        tk.Label(outer, text=lbl_txt, bg=DARK_BG, fg=SUBTEXT,
                 font=FONT_NORMAL, width=14, anchor="e").grid(row=i+1, column=0, padx=(0,10), pady=5, sticky="e")
        var = tk.StringVar(value=opts[0] if ftype=="combo" else opts)
        if ftype == "combo":
            ttk.Combobox(outer, textvariable=var, values=opts,
                         state="readonly", width=22, font=FONT_NORMAL).grid(row=i+1, column=1, pady=5, sticky="w")
        else:
            tk.Entry(outer, textvariable=var, width=24, bg=CARD_BG, fg=TEXT,
                     insertbackground=TEXT, relief="flat", font=FONT_NORMAL).grid(row=i+1, column=1, pady=5, sticky="w")
        fields[lbl_txt] = var

    res_var  = tk.StringVar(value="—")
    conf_var = tk.StringVar(value="")
    res_lbl  = tk.Label(outer, textvariable=res_var, bg=CARD_BG, fg=ACCENT,
                        font=("Segoe UI", 20, "bold"), width=28, pady=10)
    res_lbl.grid(row=len(defs)+2, column=0, columnspan=2, pady=(14,4))
    tk.Label(outer, textvariable=conf_var, bg=DARK_BG, fg=SUBTEXT,
             font=FONT_SMALL).grid(row=len(defs)+3, column=0, columnspan=2)

    def predict():
        res_var.set("⏳ Loading model..."); conf_var.set(""); outer.update()
        try:
            model, le = get_ml_model()
            df = get_df()
            top_cond = df["Condition"].value_counts().head(50).index
            age_num  = AGE_MAP[fields["Age Group"].get()]
            sex_enc  = 1 if fields["Sex"].get() == "Female" else 0
            ease     = int(fields["Ease of Use"].get())
            sat      = int(fields["Satisfaction"].get())
            useful   = int(fields["Useful Count"].get())
            year     = int(fields["Year"].get())
            rev_len  = int(fields["Review Length"].get())
            # condition not in predictor — use median encoded value
            cond_enc = int(len(le.classes_) // 2)
            X = np.array([[age_num, sex_enc, cond_enc, ease, sat, useful, year, rev_len]])
            pred  = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            conf  = proba.max() * 100
            stars = "★" * pred + "☆" * (5 - pred)
            res_var.set(f"Effectiveness: {pred}/5  {stars}")
            conf_var.set(f"Confidence: {conf:.1f}%")
            res_lbl.configure(fg=GREEN if pred >= 4 else (YELLOW if pred == 3 else RED))
        except Exception as ex:
            res_var.set("Error"); conf_var.set(str(ex)); res_lbl.configure(fg=RED)

    btn(outer, "  Predict  ", predict).grid(row=len(defs)+1, column=0, columnspan=2, pady=12)

# ============================================================
# TAB 3 — NLP / DL (Phase 3)
# ============================================================
SIDE_EFFECT_KEYWORDS = [
    "drowsiness","dizziness","nausea","headache","fatigue","vomiting",
    "diarrhea","constipation","rash","insomnia","anxiety","depression",
    "weight gain","weight loss","dry mouth","blurred vision","sweating",
    "tremor","palpitations","shortness of breath","chest pain","itching",
    "swelling","fever","chills","muscle pain","joint pain","hair loss",
    "stomach pain","upset stomach","bloating","gas","heartburn","cramps",
    "confusion","memory loss","mood swings","irritability","numbness",
    "tingling","weakness","loss of appetite","increased appetite",
    "dry skin","acne","bruising","bleeding","infection","back pain",
]

def build_nlp_tab(nb):
    tab = ttk.Frame(nb); nb.add(tab, text="  🧠  NLP / DL  ")
    inner_nb = ttk.Notebook(tab); inner_nb.pack(fill="both", expand=True, padx=6, pady=6)

    # ── Sub-tab: Plots ───────────────────────────────────────
    plots_tab = ttk.Frame(inner_nb); inner_nb.add(plots_tab, text="  Plots  ")
    plot_viewer(plots_tab, NLP_PLOTS)

    # ── Sub-tab: Live Analyzer ───────────────────────────────
    live_tab = ttk.Frame(inner_nb); inner_nb.add(live_tab, text="  Live Analyzer  ")

    outer = tk.Frame(live_tab, bg=DARK_BG); outer.pack(fill="both", expand=True, padx=30, pady=16)
    tk.Label(outer, text="Sentiment & Side-Effect Analyzer",
             bg=DARK_BG, fg=TEXT, font=FONT_TITLE).pack(anchor="w", pady=(0,10))

    tk.Label(outer, text="Enter a drug review:", bg=DARK_BG, fg=SUBTEXT,
             font=FONT_NORMAL).pack(anchor="w")
    review_box = tk.Text(outer, height=5, bg=CARD_BG, fg=TEXT,
                         insertbackground=TEXT, font=("Segoe UI", 11),
                         relief="flat", wrap="word")
    review_box.pack(fill="x", pady=(4, 10))
    review_box.insert("end", "This medication worked great for my condition. "
                              "I experienced some drowsiness and dry mouth but overall very satisfied.")

    res_card = tk.Frame(outer, bg=CARD_BG, padx=20, pady=14)
    res_card.pack(fill="x", pady=(0, 10))

    sent_var  = tk.StringVar(value="—")
    conf_var  = tk.StringVar(value="")
    sides_var = tk.StringVar(value="")
    status_var = tk.StringVar(value="")

    for row_i, (lbl_txt, var, color) in enumerate([
        ("Sentiment:",       sent_var,  ACCENT),
        ("Confidence:",      conf_var,  TEXT),
        ("Side Effects:",    sides_var, ORANGE),
    ]):
        tk.Label(res_card, text=lbl_txt, bg=CARD_BG, fg=SUBTEXT,
                 font=FONT_NORMAL, width=14, anchor="e").grid(row=row_i, column=0, sticky="e", pady=4)
        tk.Label(res_card, textvariable=var, bg=CARD_BG, fg=color,
                 font=FONT_BOLD, wraplength=650, justify="left").grid(row=row_i, column=1, sticky="w", padx=12)

    status_lbl = tk.Label(outer, textvariable=status_var, bg=DARK_BG, fg=SUBTEXT, font=FONT_SMALL)
    status_lbl.pack(anchor="w")

    def _detect_sides(text):
        NEGATIONS = {"no","not","never","without","didn't","don't","doesn't","wasn't","free","absence"}
        found = []
        tl = text.lower()
        for kw in SIDE_EFFECT_KEYWORDS:
            idx = tl.find(kw)
            if idx == -1: continue
            before = tl[:idx].split()[-5:]
            if not any(n in before for n in NEGATIONS):
                found.append(kw)
        return found

    def analyze():
        text = review_box.get("1.0", "end").strip()
        if not text: return
        sent_var.set("⏳ Loading models..."); status_var.set(""); outer.update()
        def _run():
            try:
                lstm, tok, maxlen, svc, lr, wv, cv_, hstack, clean_fn = get_nlp_models()
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                cleaned = clean_fn(text)
                X_te = hstack([wv.transform([cleaned]), cv_.transform([cleaned])])
                tfidf_p = float(0.5*svc.predict_proba(X_te)[0][1] + 0.5*lr.predict_proba(X_te)[0][1])
                seq    = tok.texts_to_sequences([cleaned])
                padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
                lstm_p = float(lstm.predict(padded, verbose=0)[0][0])
                prob   = 0.20*lstm_p + 0.80*tfidf_p
                label  = "Positive 😊" if prob >= 0.5 else "Negative 😞"
                conf   = prob if prob >= 0.5 else 1 - prob
                found  = _detect_sides(text)
                sent_var.set(label)
                conf_var.set(f"{conf*100:.1f}%  (TF-IDF: {tfidf_p:.2f} | LSTM: {lstm_p:.2f})")
                sides_var.set(", ".join(found) if found else "None detected")
                status_var.set("")
            except Exception as ex:
                sent_var.set("Error"); status_var.set(str(ex))
        threading.Thread(target=_run, daemon=True).start()

    btn(outer, "  Analyze  ", analyze).pack(anchor="w", pady=6)

# ============================================================
# TAB 4 — RAG + LLM (Phase 4)
# ============================================================
def build_rag_tab(nb):
    tab = ttk.Frame(nb); nb.add(tab, text="  🔍  RAG + LLM  ")

    # ── Search bar ───────────────────────────────────────────
    sf = tk.Frame(tab, bg=PANEL_BG, pady=10, padx=14)
    sf.pack(fill="x")

    # Universal paste handler — fixes Ctrl+V on Windows dark-themed entries
    def _paste(event):
        try:
            text = tab.clipboard_get()
            w = event.widget
            try: w.delete(tk.SEL_FIRST, tk.SEL_LAST)
            except tk.TclError: pass
            w.insert(tk.INSERT, text)
        except tk.TclError: pass
        return "break"

    def make_entry(parent, var, font=FONT_NORMAL, width=None, **grid_kw):
        kw = dict(textvariable=var, bg=CARD_BG, fg=TEXT, insertbackground=TEXT,
                  font=font, relief="flat", bd=0,
                  highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT)
        if width: kw["width"] = width
        e = tk.Entry(parent, **kw)
        e.grid(**grid_kw)
        e.bind("<Control-v>", _paste); e.bind("<Control-V>", _paste)
        e.bind("<Button-3>", lambda ev: _show_ctx(ev, e))
        return e

    def _show_ctx(event, widget):
        m = tk.Menu(tab, tearoff=0, bg=CARD_BG, fg=TEXT,
                    activebackground=ACCENT, activeforeground="#fff", relief="flat")
        m.add_command(label="Paste", command=lambda: widget.event_generate("<<Paste>>"))
        m.add_command(label="Copy",  command=lambda: widget.event_generate("<<Copy>>"))
        m.add_command(label="Cut",   command=lambda: widget.event_generate("<<Cut>>"))
        m.add_command(label="Select All", command=lambda: widget.select_range(0, "end"))
        try: m.tk_popup(event.x_root, event.y_root)
        finally: m.grab_release()

    tk.Label(sf, text="Query:", bg=PANEL_BG, fg=SUBTEXT, font=FONT_BOLD).grid(row=0, column=0, sticky="w", padx=(0,8))
    q_var = tk.StringVar(value="What are people's experiences with ibuprofen for headache?")
    make_entry(sf, q_var, font=("Segoe UI", 11),
               row=0, column=1, columnspan=3, sticky="ew", ipady=7)

    tk.Label(sf, text="Drug (opt):", bg=PANEL_BG, fg=SUBTEXT, font=FONT_SMALL).grid(row=1, column=0, sticky="w", pady=(8,0))
    drug_var = tk.StringVar()
    make_entry(sf, drug_var, width=20, row=1, column=1, sticky="w", pady=(8,0), ipady=4)

    tk.Label(sf, text="Condition (opt):", bg=PANEL_BG, fg=SUBTEXT, font=FONT_SMALL).grid(row=1, column=2, sticky="w", padx=(16,6), pady=(8,0))
    cond_var = tk.StringVar()
    make_entry(sf, cond_var, width=20, row=1, column=3, sticky="w", pady=(8,0), ipady=4)

    tk.Label(sf, text="Top-K:", bg=PANEL_BG, fg=SUBTEXT, font=FONT_SMALL).grid(row=1, column=4, sticky="w", padx=(16,4), pady=(8,0))
    topk_var = tk.IntVar(value=7)
    tk.Spinbox(sf, from_=1, to=20, textvariable=topk_var, width=4,
               bg=CARD_BG, fg=TEXT, buttonbackground=PANEL_BG,
               relief="flat", font=FONT_NORMAL).grid(row=1, column=5, sticky="w", pady=(8,0))
    sf.columnconfigure(1, weight=1); sf.columnconfigure(3, weight=1)

    # ── Action bar ───────────────────────────────────────────
    ab = tk.Frame(tab, bg=DARK_BG, pady=6, padx=8); ab.pack(fill="x")
    status_var = tk.StringVar(value="Ready")
    search_btn = btn(ab, "  🔍  Search  ", lambda: None)
    search_btn.pack(side="left")
    copy_btn = btn(ab, "  📋  Copy  ", lambda: None, bg=CARD_BG, fg=TEXT)
    copy_btn.pack(side="left", padx=8)
    tk.Label(ab, textvariable=status_var, bg=DARK_BG, fg=SUBTEXT, font=FONT_SMALL).pack(side="left", padx=10)

    # ── Results notebook ─────────────────────────────────────
    res_nb = ttk.Notebook(tab); res_nb.pack(fill="both", expand=True, padx=8, pady=(0,8))

    # Summary tab
    sum_tab = ttk.Frame(res_nb); res_nb.add(sum_tab, text="  📊 Summary  ")
    sum_frame, sum_txt = scrolled_text(sum_tab)
    sum_txt.tag_configure("header",  foreground=ACCENT2, font=FONT_BOLD)
    sum_txt.tag_configure("label",   foreground=BLUE,    font=FONT_BOLD)
    sum_txt.tag_configure("value",   foreground=TEXT,    font=FONT_NORMAL)
    sum_txt.tag_configure("pos",     foreground=GREEN,   font=FONT_BOLD)
    sum_txt.tag_configure("neg",     foreground=RED,     font=FONT_BOLD)
    sum_txt.tag_configure("neutral", foreground=YELLOW,  font=FONT_BOLD)
    sum_txt.tag_configure("review",  foreground="#f5c2e7",font=FONT_NORMAL)
    sum_txt.tag_configure("meta",    foreground="#94e2d5",font=FONT_SMALL)
    sum_txt.tag_configure("side",    foreground=ORANGE,  font=FONT_NORMAL)
    sum_txt.tag_configure("divider", foreground=BORDER,  font=FONT_SMALL)
    sum_frame.pack(fill="both", expand=True)

    # LLM tab
    llm_tab = ttk.Frame(res_nb); res_nb.add(llm_tab, text="  🤖 LLM Answer  ")
    llm_frame, llm_txt = scrolled_text(llm_tab)
    llm_txt.tag_configure("llm_head",  foreground=ACCENT2,   font=("Segoe UI", 11, "bold"))
    llm_txt.tag_configure("llm_body",  foreground=TEXT,      font=("Segoe UI", 11))
    llm_txt.tag_configure("llm_warn",  foreground=YELLOW,    font=FONT_SMALL)
    llm_txt.tag_configure("llm_key",   foreground=BLUE,      font=FONT_BOLD)
    llm_txt.tag_configure("divider",   foreground=BORDER,    font=FONT_SMALL)
    llm_frame.pack(fill="both", expand=True)

    # Table tab
    tbl_tab = ttk.Frame(res_nb); res_nb.add(tbl_tab, text="  Retrieved Reviews  ")
    tf = tk.Frame(tbl_tab, bg=DARK_BG); tf.pack(fill="both", expand=True)
    cols = ["#","Sim","Drug","Condition","Sat","Eff","Age","Sex","Review"]
    tree = ttk.Treeview(tf, columns=cols, show="headings", selectmode="browse")
    widths = [30,65,110,130,45,45,80,50,0]
    for c, w in zip(cols, widths):
        tree.heading(c, text=c)
        tree.column(c, width=w, stretch=(w==0), anchor="center" if w<100 else "w",
                    minwidth=w if w>0 else 250)
    sbx = ttk.Scrollbar(tf, orient="horizontal", command=tree.xview)
    sby = ttk.Scrollbar(tf, command=tree.yview)
    tree.configure(yscrollcommand=sby.set, xscrollcommand=sbx.set)
    sby.pack(side="right", fill="y"); sbx.pack(side="bottom", fill="x")
    tree.pack(fill="both", expand=True)

    det_frame = tk.Frame(tbl_tab, bg=PANEL_BG, height=110); det_frame.pack(fill="x"); det_frame.pack_propagate(False)
    tk.Label(det_frame, text="Full Review:", bg=PANEL_BG, fg=SUBTEXT, font=FONT_SMALL).pack(anchor="w", padx=10, pady=(4,2))
    det_txt = tk.Text(det_frame, bg=CARD_BG, fg=TEXT, font=FONT_NORMAL, relief="flat",
                      wrap="word", padx=10, pady=6, height=3)
    det_txt.pack(fill="both", expand=True, padx=8, pady=(0,6))

    def on_select(e):
        sel = tree.selection()
        if not sel: return
        vals = tree.item(sel[0])["values"]
        if vals:
            det_txt.configure(state="normal")
            det_txt.delete("1.0","end"); det_txt.insert("end", str(vals[-1]))
            det_txt.configure(state="disabled")
    tree.bind("<<TreeviewSelect>>", on_select)

    _last = [""]

    def _render_summary(query, hits):
        import numpy as _np
        if not hits:
            sum_txt.configure(state="normal"); sum_txt.delete("1.0","end")
            sum_txt.insert("end","No results found."); sum_txt.configure(state="disabled"); return

        drugs = list({h["drug"] for h in hits if h["drug"]})
        conds = list({h["condition"] for h in hits if h["condition"]})
        sats  = [float(h["satisfaction"]) for h in hits if str(h["satisfaction"]).replace(".","").isdigit()]
        effs  = [float(h["effectiveness"]) for h in hits if str(h["effectiveness"]).replace(".","").isdigit()]
        pos = sum(1 for h in hits if "sentiment: positive" in h["document"].lower())
        neg = sum(1 for h in hits if "sentiment: negative" in h["document"].lower())
        neu = len(hits) - pos - neg

        lines = []
        lines.append(("header", f"Based on {len(hits)} retrieved reviews:\n"))
        lines.append(("label",  "Drugs: ")); lines.append(("value", ", ".join(drugs) + "\n"))
        lines.append(("label",  "Conditions: ")); lines.append(("value", ", ".join(conds) + "\n"))
        if sats: lines.append(("label","Avg Satisfaction: ")); lines.append(("value",f"{_np.mean(sats):.1f}/5\n"))
        if effs: lines.append(("label","Avg Effectiveness: ")); lines.append(("value",f"{_np.mean(effs):.1f}/5\n"))
        lines.append(("pos",  f"✅ Positive: {pos}  ")); lines.append(("neutral",f"⚠️ Neutral: {neu}  ")); lines.append(("neg",f"❌ Negative: {neg}\n"))
        lines.append(("divider","─"*70+"\n"))

        for i, h in enumerate(hits, 1):
            rv = re.search(r"Review: (.+)", h["document"])
            rv_txt = rv.group(1) if rv else h["document"]
            sat = float(h["satisfaction"]) if str(h["satisfaction"]).replace(".","").isdigit() else 0
            icon = "✅" if sat >= 4 else ("❌" if sat <= 2 else "⚠️")
            tag = "pos" if sat >= 4 else ("neg" if sat <= 2 else "neutral")
            lines.append((tag, f"\n[{i}] {icon}  Sim: {h['similarity']:.0%}  |  Sat: {h['satisfaction']}/5  |  Eff: {h['effectiveness']}/5\n"))
            lines.append(("meta", f"Drug: {h['drug']}  |  Condition: {h['condition']}  |  Age: {h['age']}  |  Sex: {h['sex']}\n"))
            lines.append(("review", f'"{rv_txt}"\n'))

        full = "".join(v for _, v in lines)
        _last[0] = full

        sum_txt.configure(state="normal"); sum_txt.delete("1.0","end")
        sum_txt.insert("end","Query: ","label"); sum_txt.insert("end",query+"\n","value")
        sum_txt.insert("end","═"*70+"\n\n","divider")
        for tag, val in lines:
            sum_txt.insert("end", val, tag)
        sum_txt.configure(state="disabled")

    def _render_table(hits):
        tree.delete(*tree.get_children())
        for i, h in enumerate(hits, 1):
            rv = re.search(r"Review: (.+)", h["document"])
            rv_txt = rv.group(1) if rv else h["document"]
            sat = float(h["satisfaction"]) if str(h["satisfaction"]).replace(".","").isdigit() else 0
            tag = "pos_row" if sat >= 4 else ("neg_row" if sat <= 2 else "")
            tree.insert("","end", tags=(tag,), values=(
                i, f"{h['similarity']:.0%}", h["drug"][:20], h["condition"][:25],
                h["satisfaction"], h["effectiveness"], h["age"], h["sex"], rv_txt
            ))
        tree.tag_configure("pos_row", foreground=GREEN)
        tree.tag_configure("neg_row", foreground=RED)

    def do_search():
        query = q_var.get().strip()
        if not query: return
        status_var.set("⏳ Searching..."); search_btn.configure(state="disabled"); tab.update()

        def _run():
            try:
                col, emb = get_rag()
                q_emb = emb.encode([query]).tolist()
                where = {}
                if drug_var.get().strip(): where["Drug"] = {"$eq": drug_var.get().strip().lower()}
                if cond_var.get().strip(): where["Condition"] = {"$eq": cond_var.get().strip().lower()}
                kw = dict(query_embeddings=q_emb, n_results=topk_var.get(),
                          include=["documents","metadatas","distances"])
                if where: kw["where"] = where
                res = col.query(**kw)
                hits = []
                for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                    hits.append({"document":doc, "drug":meta.get("Drug",""),
                                 "condition":meta.get("Condition",""),
                                 "satisfaction":meta.get("Satisfaction",""),
                                 "effectiveness":meta.get("Effectiveness",""),
                                 "sides":meta.get("Sides",""),
                                 "age":meta.get("Age",""), "sex":meta.get("Sex",""),
                                 "similarity":round(1-dist,3)})

                # RAG results shown immediately — independent of LLM
                tab.after(0, lambda: _render_summary(query, hits))
                tab.after(0, lambda: _render_table(hits))
                tab.after(0, lambda: status_var.set(f"✅ {len(hits)} reviews  |  Asking LLM..."))
                tab.after(0, lambda: search_btn.configure(state="normal"))

                # LLM runs in its own thread — won't block anything
                threading.Thread(target=_run_llm, args=(query, hits), daemon=True).start()

            except Exception as ex:
                tab.after(0, lambda: status_var.set(f"❌ {ex}"))
                tab.after(0, lambda: search_btn.configure(state="normal"))

        def _run_llm(query, hits):
            tab.after(0, _render_llm_loading)
            try:
                llm_answer = _call_llm(query, hits)
            except Exception as ex:
                llm_answer = f"⚠️ LLM error: {ex}"
            tab.after(0, lambda a=llm_answer: _render_llm(query, a))
            tab.after(0, lambda: status_var.set(f"✅ Done — {len(hits)} reviews"))

        threading.Thread(target=_run, daemon=True).start()

    def _render_llm_loading():
        llm_txt.configure(state="normal"); llm_txt.delete("1.0","end")
        llm_txt.insert("end", "⏳  Generating LLM response...\n", "llm_warn")
        llm_txt.configure(state="disabled")

    def _call_llm(query, hits):
        """Try OpenRouter LLM with timeout; returns None if no key, str on error."""
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return None

        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                timeout=30.0,   # 30s timeout — won't hang forever
            )
            context_parts = []
            for i, h in enumerate(hits, 1):
                rv = re.search(r"Review: (.+)", h["document"])
                rv_txt = rv.group(1) if rv else h["document"]
                sat = h["satisfaction"]
                sentiment = "Positive" if float(sat) >= 4 else ("Negative" if float(sat) <= 2 else "Neutral")
                context_parts.append(
                    f"[Review {i}] Drug: {h['drug']} | Condition: {h['condition']} | "
                    f"Satisfaction: {sat}/5 | Sentiment: {sentiment} | "
                    f"Side effects: {h['sides']}\nText: {rv_txt}"
                )
            context = "\n\n".join(context_parts)
            system_prompt = (
                "You are a medical assistant specialized in analyzing patient drug experiences. "
                "Summarize the provided reviews to answer the user's question. "
                "Be concise, mention side effects, satisfaction rates, and always note "
                "these are personal experiences, not medical advice."
            )
            user_prompt = f"Question: {query}\n\nReviews ({len(hits)} total):\n{context}\n\nProvide a clear medical summary."
            llm_model = os.environ.get("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
            resp = client.chat.completions.create(
                model=llm_model,
                messages=[{"role":"system","content":system_prompt},
                          {"role":"user","content":user_prompt}],
                max_tokens=1024, temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as ex:
            return f"⚠️ LLM error: {ex}"

    def _render_llm(query, answer):
        llm_txt.configure(state="normal"); llm_txt.delete("1.0","end")
        llm_txt.insert("end", "🤖  LLM Answer\n", "llm_head")
        llm_txt.insert("end", "═"*70+"\n", "divider")
        llm_txt.insert("end", "Query: ", "llm_key")
        llm_txt.insert("end", query+"\n\n", "llm_body")

        if answer is None:
            llm_txt.insert("end",
                "⚠️  No OPENROUTER_API_KEY found in environment.\n\n"
                "To enable LLM answers, set the variable before running:\n\n"
                "    Windows CMD:        set OPENROUTER_API_KEY=your_key\n"
                "    Windows PowerShell: $env:OPENROUTER_API_KEY='your_key'\n\n"
                "Get a free key at: https://openrouter.ai\n",
                "llm_warn")
        else:
            # Colour key lines
            for line in answer.split("\n"):
                if line.strip().startswith(("**","##","#","---","===")):
                    llm_txt.insert("end", line+"\n", "llm_key")
                elif line.strip() == "":
                    llm_txt.insert("end", "\n")
                else:
                    llm_txt.insert("end", line+"\n", "llm_body")

        llm_txt.configure(state="disabled")
        # Auto-switch to LLM tab
        res_nb.select(1)

    def do_copy():
        if _last[0]:
            tab.clipboard_clear(); tab.clipboard_append(_last[0])
            status_var.set("✅ Copied!"); tab.after(2000, lambda: status_var.set("Ready"))

    search_btn.configure(command=do_search)
    copy_btn.configure(command=do_copy)
    tab.bind("<Return>", lambda e: do_search())
    tab.after(600, do_search)

# ============================================================
# TAB 5 — Overview / Home
# ============================================================
def build_home_tab(nb):
    tab = ttk.Frame(nb); nb.add(tab, text="  🏠  Overview  ")

    canvas = tk.Canvas(tab, bg=DARK_BG, highlightthickness=0)
    sb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y"); canvas.pack(fill="both", expand=True)
    inner = tk.Frame(canvas, bg=DARK_BG)
    canvas.create_window((0,0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    # Title
    tk.Label(inner, text="WebMD Drug Reviews — Final Project",
             bg=DARK_BG, fg=ACCENT, font=("Segoe UI", 18, "bold")).pack(pady=(24,4))
    tk.Label(inner, text="ML + DL + LLM + RAG Integration Pipeline",
             bg=DARK_BG, fg=SUBTEXT, font=("Segoe UI", 12)).pack(pady=(0,24))

    # Phase cards
    phases = [
        ("📊", "Phase 1 — EDA",
         "Python · NumPy · Pandas · Matplotlib · Seaborn\n"
         "Data cleaning, descriptive stats, 8 visualizations",
         BLUE),
        ("🤖", "Phase 2 — Machine Learning",
         "Random Forest · Gradient Boosting · Logistic Regression · XGBoost\n"
         "Effectiveness prediction (1-5) · 70/15/15 split · Feature importance",
         GREEN),
        ("🧠", "Phase 3 — Deep Learning & NLP",
         "BiLSTM · TF-IDF + LinearSVC Ensemble · Sentiment Analysis\n"
         "Side-effect extraction · Negation-aware detection · 6 plots",
         YELLOW),
        ("🔍", "Phase 4 — RAG + LLM",
         "ChromaDB · SentenceTransformers (all-MiniLM-L6-v2)\n"
         "Semantic retrieval · OpenRouter LLM · Arabic medical summaries",
         ORANGE),
    ]

    grid = tk.Frame(inner, bg=DARK_BG); grid.pack(padx=30, pady=10, fill="x")
    grid.columnconfigure(0, weight=1); grid.columnconfigure(1, weight=1)

    for i, (icon, title, desc, color) in enumerate(phases):
        f = tk.Frame(grid, bg=CARD_BG, padx=20, pady=16)
        f.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="nsew")
        tk.Label(f, text=f"{icon}  {title}", bg=CARD_BG, fg=color,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(f, text=desc, bg=CARD_BG, fg=SUBTEXT,
                 font=FONT_SMALL, justify="left").pack(anchor="w", pady=(6,0))

    # Dataset stats
    section_label(inner, "Dataset")
    stats_frame = tk.Frame(inner, bg=DARK_BG); stats_frame.pack(padx=30, fill="x")
    for i in range(4): stats_frame.columnconfigure(i, weight=1)

    def load_home_stats():
        try:
            df = get_df()
            card_label(stats_frame, "Total Reviews",    f"{len(df):,}",                        ACCENT, 0, 0)
            card_label(stats_frame, "Unique Drugs",      f"{df['Drug'].nunique():,}",           BLUE,   1, 0)
            card_label(stats_frame, "Unique Conditions", f"{df['Condition'].nunique():,}",      YELLOW, 2, 0)
            card_label(stats_frame, "Date Range",
                       f"{int(df['Year'].min())} – {int(df['Year'].max())}",                   ORANGE, 3, 0)
        except Exception as e:
            tk.Label(stats_frame, text=f"Stats unavailable: {e}",
                     bg=DARK_BG, fg=RED, font=FONT_SMALL).grid(row=0, column=0, columnspan=4)

    threading.Thread(target=load_home_stats, daemon=True).start()

    tk.Label(inner, text="", bg=DARK_BG).pack(pady=20)


# ============================================================
# Main
# ============================================================
def main():
    root = tk.Tk()
    root.title("WebMD Drug Reviews — Final Project Dashboard")
    root.geometry("1350x820")
    root.minsize(1000, 650)
    root.configure(bg=DARK_BG)

    style = ttk.Style(); apply_style(style)

    # ── Header ───────────────────────────────────────────────
    hdr = tk.Frame(root, bg=ACCENT, height=58); hdr.pack(fill="x"); hdr.pack_propagate(False)
    tk.Label(hdr, text="  🧬  WebMD Drug Reviews  |  ML + DL + LLM + RAG",
             bg=ACCENT, fg="#fff", font=("Segoe UI", 15, "bold")).pack(side="left", padx=16, pady=10)
    tk.Label(hdr, text="Final Project  •  All Phases Integrated",
             bg=ACCENT, fg="#e0e0ff", font=FONT_NORMAL).pack(side="right", padx=20)

    # ── Main notebook ────────────────────────────────────────
    nb = ttk.Notebook(root); nb.pack(fill="both", expand=True, padx=8, pady=8)

    build_home_tab(nb)
    build_eda_tab(nb)
    build_ml_tab(nb)
    build_nlp_tab(nb)
    build_rag_tab(nb)

    root.mainloop()


if __name__ == "__main__":
    main()
