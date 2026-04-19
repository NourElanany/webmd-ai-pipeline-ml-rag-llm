# ============================================================
# Phase 3: Deep Learning & NLP
# Sentiment Analysis (TF-IDF + LSTM Ensemble) + Side-Effect Extraction
# WebMD Drug Reviews Dataset
# ============================================================

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.sparse import hstack

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# ── GPU Setup ──────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"GPU detected: {[g.name for g in gpus]}  |  Mixed precision: ON")
else:
    print("No GPU detected — running on CPU")
# ───────────────────────────────────────────────────────────

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, LSTM, Dense, Dropout,
                                     Bidirectional, GlobalMaxPooling1D,
                                     SpatialDropout1D)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

sns.set_theme(style="whitegrid")
matplotlib.rcParams["figure.dpi"] = 120

PLOTS_DIR  = "nlp_plots"
MODEL_PATH = "lstm_sentiment_model.keras"
os.makedirs(PLOTS_DIR, exist_ok=True)

VOCAB_SIZE = 30000
MAX_LEN    = 150
EMBED_DIM  = 128
BATCH_SIZE = 1024 if gpus else 256
EPOCHS     = 20

# Known medical side-effect keywords
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

# ============================================================
# 1. Text Cleaning
# ============================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Keep ALL words — stopwords like "not", "very", "but" carry sentiment signal
    return " ".join(w for w in text.split() if len(w) > 1)

# ============================================================
# 2. Load Data
# ============================================================
def load_nlp_data(path="webmd_cleaned.csv", sample_size=80000):
    df = pd.read_csv(path)
    df = df[df["Reviews"].str.strip().str.len() > 10].copy()

    # Drop ambiguous middle rating (3) — it's noise
    df = df[df["Satisfaction"] != 3].copy()
    df["Sentiment"] = (df["Satisfaction"] >= 4).astype(int)

    df = df.groupby("Sentiment", group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size // 2), random_state=42)
    ).reset_index(drop=True)

    df["Clean_Review"] = df["Reviews"].apply(clean_text)
    df = df[df["Clean_Review"].str.len() > 5]

    pos = (df["Sentiment"] == 1).sum()
    neg = (df["Sentiment"] == 0).sum()
    print(f"NLP dataset: {len(df):,} samples  |  Positive: {pos:,}  Negative: {neg:,}")
    return df

# ============================================================
# 3. Unified Split — same indices for LSTM sequences AND raw texts
# ============================================================
def prepare_data(df):
    """
    Single split so TF-IDF and LSTM always see the exact same train/val/test rows.
    Returns both padded sequences (for LSTM) and raw texts (for TF-IDF).
    """
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>", lower=True)
    tokenizer.fit_on_texts(df["Clean_Review"])
    sequences = tokenizer.texts_to_sequences(df["Clean_Review"])
    X_seq = pad_sequences(sequences, maxlen=MAX_LEN, truncating="pre", padding="pre")
    X_txt = df["Clean_Review"].values
    y     = df["Sentiment"].values

    # One split, shared indices
    idx = np.arange(len(y))
    idx_train, idx_temp = train_test_split(idx, test_size=0.30, random_state=42, stratify=y)
    idx_val,   idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=y[idx_temp])

    print(f"Split — Train: {len(idx_train):,} | Val: {len(idx_val):,} | Test: {len(idx_test):,}")
    return (tokenizer,
            X_seq[idx_train], X_seq[idx_val], X_seq[idx_test],
            X_txt[idx_train], X_txt[idx_val], X_txt[idx_test],
            y[idx_train],     y[idx_val],     y[idx_test])

# ============================================================
# 4. TF-IDF + LinearSVC  (PRIMARY — targets 90%+)
# ============================================================
def build_tfidf_model(train_texts, y_train, val_texts, y_val):
    """
    Two TF-IDF classifiers ensembled:
    1. LinearSVC  — fast, high precision on sparse features
    2. LogisticRegression — better calibrated probabilities
    Combined word (1-3) + char (2-4) n-grams.
    """
    print("\n-- Training TF-IDF models --")

    word_vec = TfidfVectorizer(
        ngram_range=(1, 3), max_features=150000,
        sublinear_tf=True, min_df=2, analyzer="word",
        strip_accents="unicode", token_pattern=r"\b\w+\b",
    )
    char_vec = TfidfVectorizer(
        ngram_range=(2, 4), max_features=150000,
        sublinear_tf=True, min_df=3, analyzer="char_wb",
    )

    X_tr = hstack([word_vec.fit_transform(train_texts), char_vec.fit_transform(train_texts)])
    X_vl = hstack([word_vec.transform(val_texts),       char_vec.transform(val_texts)])

    # Model 1: LinearSVC with higher C
    svc = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=3000), cv=3)
    svc.fit(X_tr, y_train)

    # Model 2: Logistic Regression
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=5.0, max_iter=1000, solver="saga", n_jobs=-1)
    lr.fit(X_tr, y_train)

    # Ensemble of the two TF-IDF models
    val_prob = 0.5 * svc.predict_proba(X_vl)[:, 1] + 0.5 * lr.predict_proba(X_vl)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)
    print(f"TF-IDF Ensemble Val  Acc: {(val_pred==y_val).mean():.4f}  "
          f"F1: {f1_score(y_val, val_pred, average='weighted'):.4f}")
    return svc, lr, word_vec, char_vec


def evaluate_tfidf(svc, lr, word_vec, char_vec, test_texts, y_test):
    X_te   = hstack([word_vec.transform(test_texts), char_vec.transform(test_texts)])
    y_prob = 0.5 * svc.predict_proba(X_te)[:, 1] + 0.5 * lr.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = (y_pred == y_test).mean()
    f1  = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred,
                                   target_names=["Negative", "Positive"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n-- TF-IDF Test Results --")
    print(f"Test Accuracy: {acc:.4f}  |  F1 (weighted): {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    return {"accuracy": acc, "f1": f1, "report": report, "cm": cm,
            "y_pred": y_pred, "y_prob": y_prob}

# ============================================================
# 5. BiLSTM  (SECONDARY — used in ensemble)
# ============================================================
def build_model():
    """Kept for the live analyzer only — minimal model, not trained from scratch."""
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN, name="embedding"),
        SpatialDropout1D(0.5, name="spatial_dropout"),
        Bidirectional(LSTM(32, dropout=0.4, recurrent_dropout=0.4), name="bilstm"),
        Dense(16, activation="relu", kernel_regularizer=l2(1e-3), name="dense_1"),
        Dropout(0.6),
        Dense(1, activation="sigmoid", name="output", dtype="float32"),
    ])
    model.compile(optimizer=Adam(5e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(X_train, X_val, y_train, y_val):
    """
    Heavily constrained LSTM — small enough to not overfit.
    EarlyStopping with patience=2 stops it the moment val_loss stops improving.
    """
    print("\n-- Training BiLSTM (constrained) --")
    model = build_model()
    model.summary()
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True,
                      verbose=1, min_delta=1e-3),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1),
    ]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=10, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)
    model.save(MODEL_PATH)
    print(f"Model saved to '{MODEL_PATH}'")
    return model, history


def evaluate_model(model, X_test, y_test):
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    acc    = (y_pred == y_test).mean()
    f1     = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred,
                                   target_names=["Negative", "Positive"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n-- BiLSTM Test Results --")
    print(f"Test Accuracy: {acc:.4f}  |  F1 (weighted): {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    return {"accuracy": acc, "f1": f1, "report": report, "cm": cm,
            "y_pred": y_pred, "y_prob": y_prob}

# ============================================================
# 6. Weighted Ensemble (TF-IDF 65% + LSTM 35%)
# ============================================================
def evaluate_ensemble(lstm_probs, tfidf_probs, y_test):
    # TF-IDF is clearly stronger — give it 80% weight
    y_prob = 0.20 * lstm_probs + 0.80 * tfidf_probs
    y_pred = (y_prob >= 0.5).astype(int)
    acc    = (y_pred == y_test).mean()
    f1     = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred,
                                   target_names=["Negative", "Positive"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n-- Ensemble (LSTM 35% + TF-IDF 65%) --")
    print(f"Test Accuracy: {acc:.4f}  |  F1 (weighted): {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    return {"accuracy": acc, "f1": f1, "report": report, "cm": cm,
            "y_pred": y_pred, "y_prob": y_prob}

# ============================================================
# 7. Side-Effect Extraction
# ============================================================
def extract_side_effects(df, top_n=30):
    freq = Counter()
    for text in df["Reviews"].str.lower():
        for kw in SIDE_EFFECT_KEYWORDS:
            if kw in text:
                freq[kw] += 1
    for sides_text in df["Sides"].dropna():
        parts = str(sides_text).lower().split(",")
        for part in parts:
            part = re.sub(r"\s+", " ", part.strip().strip("."))
            if 3 < len(part) < 40 and part not in ("", " ", "may occur"):
                for kw in SIDE_EFFECT_KEYWORDS:
                    if kw in part:
                        freq[kw] += 1
    return freq.most_common(top_n)


def side_effects_by_sentiment(df, top_n=15):
    neg_df = df[df["Sentiment"] == 0]
    pos_df = df[df["Sentiment"] == 1]
    neg_freq, pos_freq = Counter(), Counter()
    for text in neg_df["Reviews"].str.lower():
        for kw in SIDE_EFFECT_KEYWORDS:
            if kw in text: neg_freq[kw] += 1
    for text in pos_df["Reviews"].str.lower():
        for kw in SIDE_EFFECT_KEYWORDS:
            if kw in text: pos_freq[kw] += 1
    neg_norm = {k: v / len(neg_df) for k, v in neg_freq.items()}
    pos_norm = {k: v / len(pos_df) for k, v in pos_freq.items()}
    all_kws  = sorted(set(list(neg_norm)[:top_n] + list(pos_norm)[:top_n]),
                      key=lambda k: neg_norm.get(k, 0) + pos_norm.get(k, 0), reverse=True)[:top_n]
    return all_kws, neg_norm, pos_norm

# ============================================================
# 8. Generate NLP Plots
# ============================================================
def generate_nlp_plots(history, lstm_results, tfidf_results, ensemble_results, df, tokenizer):
    plots = {}

    # Plot 1: Training History
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("BiLSTM Training History", fontsize=15, fontweight="bold")
    for ax, metric, title in zip(axes, ["loss", "accuracy"], ["Loss", "Accuracy"]):
        ax.plot(history.history[metric],          label="Train", color="#7c6af7", marker="o")
        ax.plot(history.history[f"val_{metric}"], label="Val",   color="#DD8452", marker="s")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot1_training_history.png")
    plt.savefig(p); plt.close(); plots["Training History"] = p

    # Plot 2: Model Comparison Bar Chart
    models  = ["BiLSTM", "TF-IDF+SVC", "Ensemble"]
    accs    = [lstm_results["accuracy"], tfidf_results["accuracy"], ensemble_results["accuracy"]]
    f1s     = [lstm_results["f1"],       tfidf_results["f1"],       ensemble_results["f1"]]
    x = np.arange(len(models)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, accs, w, label="Accuracy", color="#7c6af7", edgecolor="white")
    ax.bar(x + w/2, f1s,  w, label="F1",       color="#a6e3a1", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylim(0.7, 1.0)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w/2, a + 0.003, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, f + 0.003, f"{f:.3f}", ha="center", fontsize=9)
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(); plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot2_model_comparison.png")
    plt.savefig(p); plt.close(); plots["Model Comparison"] = p

    # Plot 3: Confusion Matrix (best model = ensemble)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(ensemble_results["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    ax.set_title("Confusion Matrix — Ensemble", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot3_confusion_matrix.png")
    plt.savefig(p); plt.close(); plots["Confusion Matrix"] = p

    # Plot 4: Top Side Effects
    side_effects = extract_side_effects(df, top_n=20)
    if side_effects:
        labels, counts = zip(*side_effects)
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(labels[::-1], counts[::-1], color="#7c6af7", edgecolor="white")
        ax.set_title("Top 20 Side Effects Mentioned in Reviews", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mention Count")
        for bar, val in zip(bars, counts[::-1]):
            ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                    f"{val:,}", va="center", fontsize=8)
        plt.tight_layout()
        p = os.path.join(PLOTS_DIR, "plot4_top_side_effects.png")
        plt.savefig(p); plt.close(); plots["Top Side Effects"] = p

    # Plot 5: Side Effects by Sentiment
    kws, neg_norm, pos_norm = side_effects_by_sentiment(df, top_n=15)
    x = np.arange(len(kws)); width = 0.38
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, [neg_norm.get(k, 0) for k in kws], width,
           label="Negative Reviews", color="#f38ba8", edgecolor="white")
    ax.bar(x + width/2, [pos_norm.get(k, 0) for k in kws], width,
           label="Positive Reviews", color="#a6e3a1", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(kws, rotation=35, ha="right")
    ax.set_title("Side-Effect Frequency: Negative vs Positive (Normalized)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Mentions per Review"); ax.legend()
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot5_side_effects_by_sentiment.png")
    plt.savefig(p); plt.close(); plots["Side Effects by Sentiment"] = p

    # Plot 6: Confidence Distribution (ensemble)
    fig, ax = plt.subplots(figsize=(10, 5))
    pos_probs = ensemble_results["y_prob"][ensemble_results["y_pred"] == 1]
    neg_probs = ensemble_results["y_prob"][ensemble_results["y_pred"] == 0]
    ax.hist(pos_probs, bins=40, alpha=0.7, color="#a6e3a1", label="Predicted Positive", edgecolor="white")
    ax.hist(neg_probs, bins=40, alpha=0.7, color="#f38ba8", label="Predicted Negative", edgecolor="white")
    ax.axvline(0.5, color="white", linestyle="--", linewidth=1.5, label="Decision Boundary")
    ax.set_title("Prediction Confidence Distribution (Ensemble)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Probability (Positive)"); ax.set_ylabel("Count"); ax.legend()
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot6_confidence_distribution.png")
    plt.savefig(p); plt.close(); plots["Confidence Distribution"] = p

    print(f"All {len(plots)} NLP plots saved to '{PLOTS_DIR}/'")
    return plots

# ============================================================
# 9. GUI Dashboard
# ============================================================
def launch_gui(ensemble_results, tfidf_results, lstm_results, plots,
               tokenizer, model, svc, lr, word_vec, char_vec, df):
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk

    DARK_BG  = "#1e1e2e"; PANEL_BG = "#2a2a3e"; ACCENT = "#7c6af7"
    TEXT     = "#cdd6f4"; SUBTEXT  = "#a6adc8";  CARD_BG = "#313244"
    GREEN    = "#a6e3a1"; RED      = "#f38ba8"

    root = tk.Tk()
    root.title("WebMD NLP Dashboard — Sentiment & Side-Effect Analysis")
    root.geometry("1280x780")
    root.configure(bg=DARK_BG)

    style = ttk.Style(); style.theme_use("clam")
    style.configure("TNotebook",        background=DARK_BG,  borderwidth=0)
    style.configure("TNotebook.Tab",    background=PANEL_BG, foreground=TEXT,
                    padding=[14, 6],    font=("Segoe UI", 10))
    style.map("TNotebook.Tab",
              background=[("selected", ACCENT)], foreground=[("selected", "#ffffff")])
    style.configure("TFrame",           background=DARK_BG)
    style.configure("Treeview",         background=CARD_BG,  foreground=TEXT,
                    fieldbackground=CARD_BG, rowheight=26, font=("Consolas", 9))
    style.configure("Treeview.Heading", background=PANEL_BG, foreground=ACCENT,
                    font=("Segoe UI", 9, "bold"))
    style.map("Treeview", background=[("selected", ACCENT)])

    hdr = tk.Frame(root, bg=ACCENT, height=52); hdr.pack(fill="x")
    tk.Label(hdr, text="  WebMD NLP Dashboard  |  Sentiment & Side-Effect Analysis",
             bg=ACCENT, fg="#fff", font=("Segoe UI", 14, "bold")).pack(side="left", pady=10, padx=12)
    tk.Label(hdr, text=f"Ensemble  •  Acc: {ensemble_results['accuracy']:.4f}  "
                       f"•  F1: {ensemble_results['f1']:.4f}  •  Phase 3 of 3",
             bg=ACCENT, fg="#e0e0ff", font=("Segoe UI", 10)).pack(side="right", padx=16)

    nb = ttk.Notebook(root); nb.pack(fill="both", expand=True, padx=8, pady=8)

    # ── Tab 1: Results ───────────────────────────────────────
    tab_res = ttk.Frame(nb); nb.add(tab_res, text="  Results  ")
    cv = tk.Canvas(tab_res, bg=DARK_BG, highlightthickness=0)
    sb = ttk.Scrollbar(tab_res, orient="vertical", command=cv.yview)
    cv.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y"); cv.pack(fill="both", expand=True)
    inner = tk.Frame(cv, bg=DARK_BG)
    cv.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))

    def card(parent, title, value, color=ACCENT, col=0, row=0):
        f = tk.Frame(parent, bg=CARD_BG, padx=18, pady=14)
        f.grid(row=row, column=col, padx=10, pady=8, sticky="nsew")
        tk.Label(f, text=title, bg=CARD_BG, fg=SUBTEXT, font=("Segoe UI", 9)).pack(anchor="w")
        tk.Label(f, text=value, bg=CARD_BG, fg=color,   font=("Segoe UI", 16, "bold")).pack(anchor="w")

    g = tk.Frame(inner, bg=DARK_BG); g.pack(padx=20, pady=20, fill="x")
    for i in range(4): g.columnconfigure(i, weight=1)

    rep = ensemble_results["report"]
    card(g, "Best Model",       "Ensemble (TF-IDF + BiLSTM)",                ACCENT, 0, 0)
    card(g, "Ensemble Acc",     f"{ensemble_results['accuracy']:.4f}",       GREEN,  1, 0)
    card(g, "Ensemble F1",      f"{ensemble_results['f1']:.4f}",             GREEN,  2, 0)
    card(g, "TF-IDF Acc",       f"{tfidf_results['accuracy']:.4f}",          TEXT,   3, 0)
    card(g, "BiLSTM Acc",       f"{lstm_results['accuracy']:.4f}",           TEXT,   0, 1)
    card(g, "Positive F1",      f"{rep['Positive']['f1-score']:.4f}",        GREEN,  1, 1)
    card(g, "Negative F1",      f"{rep['Negative']['f1-score']:.4f}",        RED,    2, 1)
    card(g, "Vocab Size",       f"{VOCAB_SIZE:,}  |  MaxLen: {MAX_LEN}",     SUBTEXT,3, 1)

    tk.Label(inner, text="Classification Report — Ensemble",
             bg=DARK_BG, fg=TEXT, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=20, pady=(16, 4))
    rf = tk.Frame(inner, bg=DARK_BG); rf.pack(fill="x", padx=20, pady=(0, 20))
    cols = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rt = ttk.Treeview(rf, columns=cols, show="headings", height=5)
    for c in cols:
        rt.heading(c, text=c); rt.column(c, width=140, anchor="center")
    for cls in ["Negative", "Positive", "macro avg", "weighted avg"]:
        if cls in rep:
            r = rep[cls]
            rt.insert("", "end", values=(
                cls, f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1-score']:.4f}",
                f"{int(r['support']):,}" if isinstance(r.get("support"), (int, float)) else "—"
            ))
    rt.pack(fill="x")

    # ── Tab 2: Visualizations ────────────────────────────────
    tab_viz = ttk.Frame(nb); nb.add(tab_viz, text="  Visualizations  ")
    left = tk.Frame(tab_viz, bg=PANEL_BG, width=220)
    left.pack(side="left", fill="y"); left.pack_propagate(False)
    tk.Label(left, text="NLP Charts", bg=PANEL_BG, fg=ACCENT,
             font=("Segoe UI", 11, "bold")).pack(pady=(16, 8), padx=10)
    right = tk.Frame(tab_viz, bg=DARK_BG)
    right.pack(side="right", fill="both", expand=True)
    img_lbl = tk.Label(right, bg=DARK_BG); img_lbl.pack(fill="both", expand=True, padx=10, pady=10)
    _ref = [None]; _cur = [None]

    def show_plot(path):
        _cur[0] = path
        try:
            img = Image.open(path)
            img.thumbnail((right.winfo_width()-20 or 950, right.winfo_height()-20 or 580), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            _ref[0] = photo; img_lbl.configure(image=photo, text="")
        except Exception as ex:
            img_lbl.configure(text=str(ex), fg="red")

    right.bind("<Configure>", lambda e: _cur[0] and show_plot(_cur[0]))
    btn_kw = dict(bg=CARD_BG, fg=TEXT, activebackground=ACCENT, activeforeground="#fff",
                  relief="flat", font=("Segoe UI", 9), cursor="hand2", anchor="w", padx=12, pady=6)
    first = [None]
    for name, path in plots.items():
        tk.Button(left, text=f"  {name}", **btn_kw,
                  command=lambda p=path: show_plot(p)).pack(fill="x", padx=8, pady=2)
        if first[0] is None: first[0] = path
    root.after(200, lambda: show_plot(first[0]))

    # ── Tab 3: Live Analyzer ─────────────────────────────────
    tab_live = ttk.Frame(nb); nb.add(tab_live, text="  Live Analyzer  ")
    outer = tk.Frame(tab_live, bg=DARK_BG); outer.pack(expand=True, fill="both", padx=40, pady=20)
    tk.Label(outer, text="Live Sentiment & Side-Effect Analyzer",
             bg=DARK_BG, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 12))

    txt_frame = tk.Frame(outer, bg=DARK_BG); txt_frame.pack(fill="x")
    tk.Label(txt_frame, text="Enter a drug review:", bg=DARK_BG, fg=SUBTEXT,
             font=("Segoe UI", 10)).pack(anchor="w")
    review_box = tk.Text(txt_frame, height=5, bg=CARD_BG, fg=TEXT,
                         insertbackground=TEXT, font=("Segoe UI", 11), relief="flat", wrap="word")
    review_box.pack(fill="x", pady=(4, 0))
    review_box.insert("end", "This medication worked great for my condition. "
                              "I experienced some drowsiness and dry mouth but overall very satisfied.")

    res_frame = tk.Frame(outer, bg=CARD_BG, pady=16, padx=20); res_frame.pack(fill="x", pady=16)
    sent_var  = tk.StringVar(value="—")
    conf_var  = tk.StringVar(value="")
    sides_var = tk.StringVar(value="")

    tk.Label(res_frame, text="Sentiment:",      bg=CARD_BG, fg=SUBTEXT, font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w", pady=4)
    sent_lbl = tk.Label(res_frame, textvariable=sent_var, bg=CARD_BG, fg=ACCENT, font=("Segoe UI", 18, "bold"))
    sent_lbl.grid(row=0, column=1, sticky="w", padx=12)
    tk.Label(res_frame, text="Confidence:",     bg=CARD_BG, fg=SUBTEXT, font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w", pady=4)
    tk.Label(res_frame, textvariable=conf_var,  bg=CARD_BG, fg=TEXT,    font=("Segoe UI", 11)).grid(row=1, column=1, sticky="w", padx=12)
    tk.Label(res_frame, text="Side Effects\nDetected:", bg=CARD_BG, fg=SUBTEXT, font=("Segoe UI", 10)).grid(row=2, column=0, sticky="nw", pady=4)
    tk.Label(res_frame, textvariable=sides_var, bg=CARD_BG, fg="#fab387",
             font=("Segoe UI", 10), wraplength=700, justify="left").grid(row=2, column=1, sticky="w", padx=12)

    def detect_side_effects(text):
        """
        Negation-aware side effect detection.
        Checks a window of 5 words before each keyword for negation terms.
        e.g. "did not experience headache" → headache is negated → skip it.
        """
        NEGATIONS = {"no", "not", "never", "without", "neither", "nor",
                     "hardly", "barely", "scarcely", "didn't", "don't",
                     "doesn't", "wasn't", "weren't", "haven't", "hasn't",
                     "free", "absence", "absent", "lack", "lacking"}
        text_lower = text.lower()
        found = []
        for kw in SIDE_EFFECT_KEYWORDS:
            idx = text_lower.find(kw)
            if idx == -1:
                continue
            # grab up to 5 words immediately before the keyword
            before = text_lower[:idx].split()[-5:]
            if not any(neg in before for neg in NEGATIONS):
                found.append(kw)
        return found

    def analyze():
        text = review_box.get("1.0", "end").strip()
        if not text: return
        cleaned = clean_text(text)

        # TF-IDF prediction (ensemble of SVC + LR)
        X_te = hstack([word_vec.transform([cleaned]), char_vec.transform([cleaned])])
        tfidf_prob = float(0.5 * svc.predict_proba(X_te)[0][1] + 0.5 * lr.predict_proba(X_te)[0][1])

        # LSTM prediction
        seq    = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        lstm_prob = float(model.predict(padded, verbose=0)[0][0])

        # Ensemble
        prob  = 0.20 * lstm_prob + 0.80 * tfidf_prob
        label = "Positive" if prob >= 0.5 else "Negative"
        conf  = prob if prob >= 0.5 else 1 - prob

        found = detect_side_effects(text)
        sent_var.set(f"{label} {'😊' if prob >= 0.5 else '😞'}")
        conf_var.set(f"{conf*100:.1f}%  (TF-IDF: {tfidf_prob:.2f} | LSTM: {lstm_prob:.2f})")
        sides_var.set(", ".join(found) if found else "None detected")
        sent_lbl.configure(fg=GREEN if prob >= 0.5 else RED)

    tk.Button(outer, text="  Analyze  ", command=analyze,
              bg=ACCENT, fg="#fff", font=("Segoe UI", 11, "bold"),
              relief="flat", padx=24, pady=8, cursor="hand2").pack(anchor="w")

    root.mainloop()

# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    df = load_nlp_data()

    # Single unified split — same rows for both models
    (tokenizer,
     X_seq_train, X_seq_val, X_seq_test,
     t_train,     t_val,     t_test,
     y_train,     y_val,     y_test) = prepare_data(df)

    # Train both models
    svc, lr, word_vec, char_vec = build_tfidf_model(t_train, y_train, t_val, y_val)
    model, history              = train_model(X_seq_train, X_seq_val, y_train, y_val)

    # Evaluate individually
    tfidf_results = evaluate_tfidf(svc, lr, word_vec, char_vec, t_test, y_test)
    lstm_results  = evaluate_model(model, X_seq_test, y_test)

    # Ensemble
    ensemble_results = evaluate_ensemble(lstm_results["y_prob"], tfidf_results["y_prob"], y_test)

    plots = generate_nlp_plots(history, lstm_results, tfidf_results, ensemble_results, df, tokenizer)
    launch_gui(ensemble_results, tfidf_results, lstm_results, plots,
               tokenizer, model, svc, lr, word_vec, char_vec, df)
