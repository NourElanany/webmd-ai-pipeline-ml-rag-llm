# ============================================================
# Phase 2: Machine Learning — Effectiveness Rating Prediction
# WebMD Drug Reviews Dataset
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
matplotlib.rcParams["figure.dpi"] = 120

PLOTS_DIR = "ml_plots"
MODEL_PATH = "rf_effectiveness_model.pkl"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
# 1. Load & Prepare Features
# ============================================================
AGE_MAP = {
    "0-2": 1, "3-6": 4, "7-12": 9, "13-18": 15,
    "19-24": 21, "25-34": 29, "35-44": 39,
    "45-54": 49, "55-64": 59, "65-74": 69, "75 or over": 80
}

def load_and_prepare(path="webmd_cleaned.csv"):
    df = pd.read_csv(path)

    # Map age groups to numeric midpoints
    df["Age_Num"] = df["Age"].map(AGE_MAP)

    # Drop rows with unknown/missing key features
    df = df[df["Age_Num"].notna()]
    df = df[df["Sex"].str.strip().isin(["Male", "Female"])]
    df = df[df["Condition"].str.strip() != "Unknown"]

    # Encode Sex
    df["Sex_Enc"] = (df["Sex"].str.strip() == "Female").astype(int)

    # Encode top-N conditions (keep top 50, rest → "Other")
    top_conditions = df["Condition"].value_counts().head(50).index
    df["Condition_Clean"] = df["Condition"].where(
        df["Condition"].isin(top_conditions), other="Other"
    )
    le = LabelEncoder()
    df["Condition_Enc"] = le.fit_transform(df["Condition_Clean"])

    # Target: Effectiveness (1-5) — keep as-is for classification
    df = df[df["Effectiveness"].between(1, 5)]

    features = ["Age_Num", "Sex_Enc", "Condition_Enc",
                "EaseofUse", "Satisfaction", "UsefulCount",
                "Year", "Review_Length"]
    target = "Effectiveness"

    X = df[features].copy()
    y = df[target].copy()

    print(f"Prepared dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"Target distribution:\n{y.value_counts().sort_index().to_string()}")
    return X, y, le, df


# ============================================================
# 2. Train Models & Evaluate  (70% train | 15% val | 15% test)
# ============================================================
def train_and_evaluate(X, y):
    # --- Split: 70 / 15 / 15 ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"\nSplit sizes — Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # XGBoost expects labels 0-indexed
    y_train_xgb = y_train - 1
    y_val_xgb   = y_val   - 1
    y_test_xgb  = y_test  - 1

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=10,
            n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42, n_jobs=-1))
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            n_jobs=-1, random_state=42, verbosity=0
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")

        # XGBoost uses 0-indexed labels + validation set for early stopping
        if name == "XGBoost":
            model.fit(
                X_train, y_train_xgb,
                eval_set=[(X_val, y_val_xgb)],
                verbose=False
            )
            y_pred_val  = model.predict(X_val)  + 1   # shift back to 1-5
            y_pred_test = model.predict(X_test) + 1
        else:
            model.fit(X_train, y_train)
            y_pred_val  = model.predict(X_val)
            y_pred_test = model.predict(X_test)

        val_f1  = f1_score(y_val,  y_pred_val,  average="weighted")
        acc     = accuracy_score(y_test, y_pred_test)
        f1      = f1_score(y_test, y_pred_test, average="weighted")
        report  = classification_report(y_test, y_pred_test, output_dict=True)
        cm      = confusion_matrix(y_test, y_pred_test)

        results[name] = {
            "model": model, "y_pred": y_pred_test,
            "accuracy": acc, "f1": f1, "val_f1": val_f1,
            "report": report, "cm": cm
        }
        print(f"  Val F1: {val_f1:.4f}  |  Test Accuracy: {acc:.4f}  |  Test F1: {f1:.4f}")

    # Best model by test F1
    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\nBest model: {best_name} (Test F1={results[best_name]['f1']:.4f})")

    joblib.dump(results[best_name]["model"], MODEL_PATH)
    print(f"Model saved to '{MODEL_PATH}'")

    return results, best_name, X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# 3. Generate ML Plots
# ============================================================
FEATURE_NAMES = ["Age", "Sex", "Condition", "Ease of Use",
                 "Satisfaction", "Useful Count", "Year", "Review Length"]

def generate_ml_plots(results, best_name, X_train, X_val, X_test, y_test):
    plots = {}

    # --- Plot 1: Model Comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Comparison", fontsize=15, fontweight="bold")
    names  = list(results.keys())
    accs   = [results[n]["accuracy"] for n in names]
    f1s    = [results[n]["f1"]       for n in names]
    val_f1s= [results[n]["val_f1"]   for n in names]
    colors = ["#7c6af7" if n == best_name else "#4C72B0" for n in names]

    for ax, vals, title in zip(axes, [accs, val_f1s, f1s],
                                ["Test Accuracy", "Val F1 (Weighted)", "Test F1 (Weighted)"]):
        ax.barh(names, vals, color=colors, edgecolor="white")
        ax.set_title(title); ax.set_xlim(0, 1)
        for i, v in enumerate(vals):
            ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot1_model_comparison.png")
    plt.savefig(p); plt.close(); plots["Model Comparison"] = p

    # --- Plot 2: Confusion Matrix (best model) ---
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = results[best_name]["cm"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
    ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot2_confusion_matrix.png")
    plt.savefig(p); plt.close(); plots["Confusion Matrix"] = p

    # --- Plot 3: Feature Importance (RF or XGBoost) ---
    for fi_model_name in ["XGBoost", "Random Forest"]:
        if fi_model_name in results:
            fi_model = results[fi_model_name]["model"]
            importances = fi_model.feature_importances_
            break
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(importances)),
                  importances[idx], color="#7c6af7", edgecolor="white")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in idx], rotation=30, ha="right")
    ax.set_title(f"Feature Importance — {fi_model_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Importance Score")
    for bar, val in zip(bars, importances[idx]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot3_feature_importance.png")
    plt.savefig(p); plt.close(); plots["Feature Importance"] = p

    # --- Plot 4: Per-Class F1 (best model) ---
    report = results[best_name]["report"]
    classes = [str(i) for i in range(1, 6)]
    f1_per_class = [report[c]["f1-score"] for c in classes if c in report]
    prec = [report[c]["precision"] for c in classes if c in report]
    rec  = [report[c]["recall"]    for c in classes if c in report]

    x = np.arange(len(classes))
    width = 0.28
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, prec,      width, label="Precision", color="#4C72B0", edgecolor="white")
    ax.bar(x,         f1_per_class, width, label="F1",     color="#7c6af7", edgecolor="white")
    ax.bar(x + width, rec,       width, label="Recall",    color="#55A868", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([f"Class {c}" for c in classes])
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title(f"Per-Class Metrics — {best_name}", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot4_per_class_metrics.png")
    plt.savefig(p); plt.close(); plots["Per-Class Metrics"] = p

    # --- Plot 5: Actual vs Predicted Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Actual vs Predicted Distribution — {best_name}",
                 fontsize=13, fontweight="bold")
    y_pred = results[best_name]["y_pred"]
    for ax, data, title, color in zip(
        axes,
        [y_test, y_pred],
        ["Actual", "Predicted"],
        ["#4C72B0", "#7c6af7"]
    ):
        vals, cnts = np.unique(data, return_counts=True)
        ax.bar(vals, cnts, color=color, edgecolor="white")
        ax.set_title(title); ax.set_xlabel("Effectiveness Rating")
        ax.set_ylabel("Count"); ax.set_xticks([1,2,3,4,5])
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot5_actual_vs_predicted.png")
    plt.savefig(p); plt.close(); plots["Actual vs Predicted"] = p

    print(f"All {len(plots)} ML plots saved to '{PLOTS_DIR}/'")
    return plots


# ============================================================
# 4. GUI Dashboard
# ============================================================
def launch_gui(results, best_name, plots, le_condition):
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk

    # ── Theme ────────────────────────────────────────────────
    DARK_BG  = "#1e1e2e"
    PANEL_BG = "#2a2a3e"
    ACCENT   = "#7c6af7"
    TEXT     = "#cdd6f4"
    SUBTEXT  = "#a6adc8"
    CARD_BG  = "#313244"
    GREEN    = "#a6e3a1"
    RED      = "#f38ba8"

    root = tk.Tk()
    root.title("WebMD ML Dashboard — Effectiveness Prediction")
    root.geometry("1280x780")
    root.configure(bg=DARK_BG)

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TNotebook",       background=DARK_BG,  borderwidth=0)
    style.configure("TNotebook.Tab",   background=PANEL_BG, foreground=TEXT,
                    padding=[14, 6],   font=("Segoe UI", 10))
    style.map("TNotebook.Tab",
              background=[("selected", ACCENT)],
              foreground=[("selected", "#ffffff")])
    style.configure("TFrame",          background=DARK_BG)
    style.configure("Treeview",        background=CARD_BG,  foreground=TEXT,
                    fieldbackground=CARD_BG, rowheight=26,
                    font=("Consolas", 9))
    style.configure("Treeview.Heading", background=PANEL_BG, foreground=ACCENT,
                    font=("Segoe UI", 9, "bold"))
    style.map("Treeview", background=[("selected", ACCENT)])
    style.configure("TCombobox", fieldbackground=CARD_BG, background=CARD_BG,
                    foreground=TEXT, selectbackground=ACCENT)

    # ── Header ───────────────────────────────────────────────
    hdr = tk.Frame(root, bg=ACCENT, height=52)
    hdr.pack(fill="x")
    tk.Label(hdr, text="  WebMD ML Dashboard  |  Effectiveness Prediction",
             bg=ACCENT, fg="#fff", font=("Segoe UI", 14, "bold")).pack(side="left", pady=10, padx=12)
    tk.Label(hdr, text=f"Best Model: {best_name}  •  Phase 2 of 2",
             bg=ACCENT, fg="#e0e0ff", font=("Segoe UI", 10)).pack(side="right", padx=16)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)

    # ── Tab 1: Results Summary ───────────────────────────────
    tab_results = ttk.Frame(notebook)
    notebook.add(tab_results, text="  Results  ")

    canvas_r = tk.Canvas(tab_results, bg=DARK_BG, highlightthickness=0)
    scr_r = ttk.Scrollbar(tab_results, orient="vertical", command=canvas_r.yview)
    canvas_r.configure(yscrollcommand=scr_r.set)
    scr_r.pack(side="right", fill="y")
    canvas_r.pack(fill="both", expand=True)
    inner_r = tk.Frame(canvas_r, bg=DARK_BG)
    canvas_r.create_window((0, 0), window=inner_r, anchor="nw")
    inner_r.bind("<Configure>", lambda e: canvas_r.configure(scrollregion=canvas_r.bbox("all")))

    def metric_card(parent, title, value, color=ACCENT, col=0, row=0):
        f = tk.Frame(parent, bg=CARD_BG, padx=18, pady=14)
        f.grid(row=row, column=col, padx=10, pady=8, sticky="nsew")
        tk.Label(f, text=title, bg=CARD_BG, fg=SUBTEXT, font=("Segoe UI", 9)).pack(anchor="w")
        tk.Label(f, text=value, bg=CARD_BG, fg=color,   font=("Segoe UI", 16, "bold")).pack(anchor="w")

    grid = tk.Frame(inner_r, bg=DARK_BG)
    grid.pack(padx=20, pady=20, fill="x")
    for i in range(3): grid.columnconfigure(i, weight=1)

    best = results[best_name]
    metric_card(grid, "Best Model",          best_name,                          ACCENT, 0, 0)
    metric_card(grid, "Test Accuracy",       f"{best['accuracy']:.4f}",          GREEN,  1, 0)
    metric_card(grid, "Test F1 (Weighted)",  f"{best['f1']:.4f}",                GREEN,  2, 0)

    for col_i, (name, res) in enumerate(results.items()):
        color = GREEN if name == best_name else TEXT
        metric_card(grid, f"{name}", f"Val F1: {res['val_f1']:.4f}  |  Test F1: {res['f1']:.4f}", color, col_i % 4, 1 + col_i // 4)

    # Classification report table
    tk.Label(inner_r, text=f"Classification Report — {best_name}",
             bg=DARK_BG, fg=TEXT, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=20, pady=(16, 4))

    rep_frame = tk.Frame(inner_r, bg=DARK_BG)
    rep_frame.pack(fill="x", padx=20, pady=(0, 20))

    rep_cols = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rtree = ttk.Treeview(rep_frame, columns=rep_cols, show="headings", height=7)
    for c in rep_cols:
        rtree.heading(c, text=c)
        rtree.column(c, width=130, anchor="center")

    report = best["report"]
    for cls in ["1", "2", "3", "4", "5"]:
        if cls in report:
            r = report[cls]
            rtree.insert("", "end", values=(
                f"Effectiveness {cls}",
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1-score']:.4f}",
                f"{int(r['support']):,}"
            ))
    for key in ["macro avg", "weighted avg"]:
        if key in report:
            r = report[key]
            rtree.insert("", "end", values=(
                key.title(),
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1-score']:.4f}",
                "—"
            ))
    rtree.pack(fill="x")

    # ── Tab 2: Visualizations ────────────────────────────────
    tab_viz = ttk.Frame(notebook)
    notebook.add(tab_viz, text="  Visualizations  ")

    left = tk.Frame(tab_viz, bg=PANEL_BG, width=210)
    left.pack(side="left", fill="y"); left.pack_propagate(False)
    tk.Label(left, text="ML Charts", bg=PANEL_BG, fg=ACCENT,
             font=("Segoe UI", 11, "bold")).pack(pady=(16, 8), padx=10)

    right = tk.Frame(tab_viz, bg=DARK_BG)
    right.pack(side="right", fill="both", expand=True)
    img_lbl = tk.Label(right, bg=DARK_BG)
    img_lbl.pack(fill="both", expand=True, padx=10, pady=10)

    _ref = [None]
    _cur = [None]

    def show_plot(path):
        _cur[0] = path
        try:
            img = Image.open(path)
            w = right.winfo_width() - 20 or 950
            h = right.winfo_height() - 20 or 580
            img.thumbnail((w, h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            _ref[0] = photo
            img_lbl.configure(image=photo, text="")
        except Exception as ex:
            img_lbl.configure(text=str(ex), fg="red")

    right.bind("<Configure>", lambda e: _cur[0] and show_plot(_cur[0]))

    btn_kw = dict(bg=CARD_BG, fg=TEXT, activebackground=ACCENT,
                  activeforeground="#fff", relief="flat",
                  font=("Segoe UI", 9), cursor="hand2",
                  anchor="w", padx=12, pady=6)
    first = [None]
    for name, path in plots.items():
        tk.Button(left, text=f"  {name}", **btn_kw,
                  command=lambda p=path: show_plot(p)).pack(fill="x", padx=8, pady=2)
        if first[0] is None: first[0] = path
    root.after(200, lambda: show_plot(first[0]))

    # ── Tab 3: Live Predictor ────────────────────────────────
    tab_pred = ttk.Frame(notebook)
    notebook.add(tab_pred, text="  Live Predictor  ")

    pred_outer = tk.Frame(tab_pred, bg=DARK_BG)
    pred_outer.pack(expand=True)

    tk.Label(pred_outer, text="Predict Drug Effectiveness",
             bg=DARK_BG, fg=TEXT, font=("Segoe UI", 14, "bold")).grid(
             row=0, column=0, columnspan=2, pady=(20, 16))

    fields = {}
    field_defs = [
        ("Age Group",    "combobox", list(AGE_MAP.keys())),
        ("Sex",          "combobox", ["Male", "Female"]),
        ("Condition",    "combobox", sorted(list(le_condition.classes_))),
        ("Ease of Use",  "combobox", ["1", "2", "3", "4", "5"]),
        ("Satisfaction", "combobox", ["1", "2", "3", "4", "5"]),
        ("Useful Count", "entry",    None),
        ("Year",         "entry",    None),
        ("Review Length","entry",    None),
    ]

    for i, (label, ftype, options) in enumerate(field_defs):
        tk.Label(pred_outer, text=label, bg=DARK_BG, fg=SUBTEXT,
                 font=("Segoe UI", 10), width=14, anchor="e").grid(
                 row=i+1, column=0, padx=(0, 10), pady=6, sticky="e")
        if ftype == "combobox":
            var = tk.StringVar(value=options[0])
            cb = ttk.Combobox(pred_outer, textvariable=var, values=options,
                              state="readonly", width=22,
                              font=("Segoe UI", 10))
            cb.grid(row=i+1, column=1, pady=6, sticky="w")
            fields[label] = var
        else:
            defaults = {"Useful Count": "5", "Year": "2020", "Review Length": "200"}
            var = tk.StringVar(value=defaults.get(label, ""))
            ent = tk.Entry(pred_outer, textvariable=var, width=24,
                           bg=CARD_BG, fg=TEXT, insertbackground=TEXT,
                           relief="flat", font=("Segoe UI", 10))
            ent.grid(row=i+1, column=1, pady=6, sticky="w")
            fields[label] = var

    result_var = tk.StringVar(value="—")
    result_lbl = tk.Label(pred_outer, textvariable=result_var,
                          bg=CARD_BG, fg=ACCENT,
                          font=("Segoe UI", 22, "bold"),
                          width=20, pady=12, relief="flat")
    result_lbl.grid(row=len(field_defs)+2, column=0, columnspan=2, pady=(16, 4))

    conf_var = tk.StringVar(value="")
    tk.Label(pred_outer, textvariable=conf_var, bg=DARK_BG, fg=SUBTEXT,
             font=("Segoe UI", 9)).grid(row=len(field_defs)+3, column=0, columnspan=2)

    def predict():
        try:
            model = results[best_name]["model"]
            age_num  = AGE_MAP[fields["Age Group"].get()]
            sex_enc  = 1 if fields["Sex"].get() == "Female" else 0
            cond_raw = fields["Condition"].get()
            cond_enc = le_condition.transform([cond_raw])[0]
            ease     = int(fields["Ease of Use"].get())
            sat      = int(fields["Satisfaction"].get())
            useful   = int(fields["Useful Count"].get())
            year     = int(fields["Year"].get())
            rev_len  = int(fields["Review Length"].get())

            X_input = np.array([[age_num, sex_enc, cond_enc,
                                  ease, sat, useful, year, rev_len]])
            pred  = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0]
            conf  = proba.max() * 100

            stars = "★" * pred + "☆" * (5 - pred)
            result_var.set(f"Effectiveness: {pred}/5  {stars}")
            conf_var.set(f"Confidence: {conf:.1f}%")
            result_lbl.configure(fg=GREEN if pred >= 4 else (ACCENT if pred == 3 else RED))
        except Exception as ex:
            result_var.set("Error")
            conf_var.set(str(ex))
            result_lbl.configure(fg=RED)

    tk.Button(pred_outer, text="  Predict  ", command=predict,
              bg=ACCENT, fg="#fff", font=("Segoe UI", 11, "bold"),
              relief="flat", padx=20, pady=8, cursor="hand2").grid(
              row=len(field_defs)+1, column=0, columnspan=2, pady=12)

    root.mainloop()


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    X, y, le_condition, df_clean = load_and_prepare()
    results, best_name, X_train, X_val, X_test, y_train, y_val, y_test = train_and_evaluate(X, y)
    plots = generate_ml_plots(results, best_name, X_train, X_val, X_test, y_test)
    launch_gui(results, best_name, plots, le_condition)
