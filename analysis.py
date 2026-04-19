# ============================================================
# Phase 1: Data Analysis & Preparation - WebMD Drug Reviews
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
matplotlib.rcParams["figure.dpi"] = 120

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
# 1. Load Data
# ============================================================
def load_data(path="webmd.csv"):
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df

# ============================================================
# 2. Data Cleaning
# ============================================================
def clean_data(df):
    report = {}

    # --- Missing values before ---
    missing_before = df.isnull().sum()
    report["missing_before"] = missing_before[missing_before > 0].to_dict()

    # Fill missing text columns
    for col in ["Age", "Condition", "Sex", "Sides", "Reviews"]:
        df[col] = df[col].fillna("Unknown")

    # Fill missing numeric columns with median
    for col in ["EaseofUse", "Effectiveness", "Satisfaction"]:
        df[col] = df[col].fillna(df[col].median())

    report["missing_after"] = df.isnull().sum().sum()

    # Parse dates and extract year
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year

    # Clean reviews text and compute length
    df["Reviews"] = df["Reviews"].astype(str).str.strip()
    df["Review_Length"] = df["Reviews"].apply(len)

    # Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    report["duplicates_removed"] = before - len(df)

    # Remove out-of-range ratings
    for col in ["EaseofUse", "Effectiveness", "Satisfaction"]:
        df = df[df[col].between(1, 5)]

    report["final_shape"] = df.shape

    # Descriptive stats
    report["stats"] = df[
        ["EaseofUse", "Effectiveness", "Satisfaction", "UsefulCount", "Review_Length"]
    ].describe().round(2)

    print(f"Cleaning done. Final shape: {df.shape[0]:,} rows")
    return df, report


# ============================================================
# 3. Generate & Save All Plots
# ============================================================
def generate_plots(df):
    plots = {}  # name -> filepath

    # --- Plot 1: Ratings Distribution ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Ratings Distribution", fontsize=16, fontweight="bold")
    for ax, col, color, label in zip(
        axes,
        ["Satisfaction", "Effectiveness", "EaseofUse"],
        ["#4C72B0", "#DD8452", "#55A868"],
        ["Satisfaction", "Effectiveness", "Ease of Use"],
    ):
        counts = df[col].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=color, edgecolor="white")
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Rating (1-5)")
        ax.set_ylabel("Number of Reviews")
        ax.set_xticks([1, 2, 3, 4, 5])
        for i, v in zip(counts.index, counts.values):
            ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=8)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot1_ratings_distribution.png")
    plt.savefig(p); plt.close(); plots["Ratings Distribution"] = p

    # --- Plot 2: Gender Distribution ---
    fig, ax = plt.subplots(figsize=(6, 6))
    sex_counts = df["Sex"].value_counts()
    sex_counts = sex_counts[sex_counts.index != "Unknown"]
    ax.pie(sex_counts.values, labels=sex_counts.index, autopct="%1.1f%%",
           colors=["#4C72B0", "#DD8452", "#55A868"], startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title("Gender Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot2_gender_distribution.png")
    plt.savefig(p); plt.close(); plots["Gender Distribution"] = p

    # --- Plot 3: Age Group Distribution ---
    fig, ax = plt.subplots(figsize=(10, 5))
    age_order = ["7-12", "13-18", "19-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75 or over"]
    age_counts = df["Age"].value_counts()
    age_counts = age_counts.reindex([a for a in age_order if a in age_counts.index])
    sns.barplot(x=age_counts.index, y=age_counts.values, palette="Blues_d", ax=ax)
    ax.set_title("Age Group Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age Group"); ax.set_ylabel("Number of Reviews")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(age_counts.values):
        ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=8)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot3_age_distribution.png")
    plt.savefig(p); plt.close(); plots["Age Distribution"] = p

    # --- Plot 4: Top 15 Conditions ---
    fig, ax = plt.subplots(figsize=(12, 6))
    top_cond = df["Condition"].value_counts().head(15)
    top_cond = top_cond[top_cond.index != "Unknown"]
    sns.barplot(x=top_cond.values, y=top_cond.index, palette="viridis", ax=ax)
    ax.set_title("Top 15 Medical Conditions", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Reviews"); ax.set_ylabel("Condition")
    for i, v in enumerate(top_cond.values):
        ax.text(v + 100, i, f"{v:,}", va="center", fontsize=8)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot4_top_conditions.png")
    plt.savefig(p); plt.close(); plots["Top Conditions"] = p

    # --- Plot 5: Top 10 Drugs Average Ratings ---
    fig, ax = plt.subplots(figsize=(12, 6))
    top_drugs = df["Drug"].value_counts().head(10).index
    drug_ratings = (
        df[df["Drug"].isin(top_drugs)]
        .groupby("Drug")[["Satisfaction", "Effectiveness", "EaseofUse"]]
        .mean().round(2)
    )
    drug_ratings.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
    ax.set_title("Avg Ratings for Top 10 Most-Reviewed Drugs", fontsize=14, fontweight="bold")
    ax.set_xlabel("Drug"); ax.set_ylabel("Average Rating")
    ax.set_ylim(0, 5.5)
    ax.legend(["Satisfaction", "Effectiveness", "Ease of Use"])
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot5_top_drugs_ratings.png")
    plt.savefig(p); plt.close(); plots["Top Drugs Ratings"] = p

    # --- Plot 6: Reviews Over Years ---
    fig, ax = plt.subplots(figsize=(12, 5))
    yearly = df.groupby("Year").size().reset_index(name="Count").dropna()
    ax.plot(yearly["Year"], yearly["Count"], marker="o", linewidth=2,
            color="#4C72B0", markersize=6)
    ax.fill_between(yearly["Year"], yearly["Count"], alpha=0.15, color="#4C72B0")
    ax.set_title("Number of Reviews Over the Years", fontsize=14, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("Number of Reviews")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot6_reviews_over_years.png")
    plt.savefig(p); plt.close(); plots["Reviews Over Years"] = p

    # --- Plot 7: Correlation Heatmap ---
    fig, ax = plt.subplots(figsize=(7, 5))
    corr_cols = ["EaseofUse", "Effectiveness", "Satisfaction", "UsefulCount", "Review_Length"]
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, square=True)
    ax.set_title("Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot7_correlation_heatmap.png")
    plt.savefig(p); plt.close(); plots["Correlation Heatmap"] = p

    # --- Plot 8: Review Length Distribution ---
    fig, ax = plt.subplots(figsize=(10, 5))
    lengths = df["Review_Length"].clip(upper=2000)
    ax.hist(lengths, bins=50, color="#55A868", edgecolor="white", linewidth=0.5)
    ax.axvline(lengths.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: {lengths.mean():.0f} chars")
    ax.axvline(lengths.median(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Median: {lengths.median():.0f} chars")
    ax.set_title("Review Length Distribution (Unstructured Text)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Review Length (chars)"); ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "plot8_review_length_dist.png")
    plt.savefig(p); plt.close(); plots["Review Length Dist."] = p

    print(f"All {len(plots)} plots saved to '{PLOTS_DIR}/'")
    return plots


# ============================================================
# 4. GUI Dashboard
# ============================================================
def launch_gui(df, report, plots):
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("WebMD Drug Reviews — Analysis Dashboard")
    root.geometry("1200x750")
    root.configure(bg="#1e1e2e")

    DARK_BG   = "#1e1e2e"
    PANEL_BG  = "#2a2a3e"
    ACCENT    = "#7c6af7"
    TEXT      = "#cdd6f4"
    SUBTEXT   = "#a6adc8"
    CARD_BG   = "#313244"

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TNotebook",        background=DARK_BG,  borderwidth=0)
    style.configure("TNotebook.Tab",    background=PANEL_BG, foreground=TEXT,
                    padding=[14, 6],    font=("Segoe UI", 10))
    style.map("TNotebook.Tab",
              background=[("selected", ACCENT)],
              foreground=[("selected", "#ffffff")])
    style.configure("TFrame",           background=DARK_BG)
    style.configure("Treeview",         background=CARD_BG,  foreground=TEXT,
                    fieldbackground=CARD_BG, rowheight=24,
                    font=("Consolas", 9))
    style.configure("Treeview.Heading", background=PANEL_BG, foreground=ACCENT,
                    font=("Segoe UI", 9, "bold"))
    style.map("Treeview", background=[("selected", ACCENT)])
    style.configure("Vertical.TScrollbar", background=PANEL_BG, troughcolor=DARK_BG)

    # ── Header ──────────────────────────────────────────────
    header = tk.Frame(root, bg=ACCENT, height=52)
    header.pack(fill="x")
    tk.Label(header, text="  WebMD Drug Reviews  |  Analysis Dashboard",
             bg=ACCENT, fg="#ffffff", font=("Segoe UI", 14, "bold")).pack(side="left", pady=10, padx=12)
    tk.Label(header, text=f"Dataset: {report['final_shape'][0]:,} rows  •  Phase 1 of 2",
             bg=ACCENT, fg="#e0e0ff", font=("Segoe UI", 10)).pack(side="right", padx=16)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)

    # ── Tab 1: Summary ──────────────────────────────────────
    tab_summary = ttk.Frame(notebook)
    notebook.add(tab_summary, text="  Summary  ")

    canvas_s = tk.Canvas(tab_summary, bg=DARK_BG, highlightthickness=0)
    scroll_s = ttk.Scrollbar(tab_summary, orient="vertical", command=canvas_s.yview)
    canvas_s.configure(yscrollcommand=scroll_s.set)
    scroll_s.pack(side="right", fill="y")
    canvas_s.pack(fill="both", expand=True)
    inner_s = tk.Frame(canvas_s, bg=DARK_BG)
    canvas_s.create_window((0, 0), window=inner_s, anchor="nw")
    inner_s.bind("<Configure>", lambda e: canvas_s.configure(scrollregion=canvas_s.bbox("all")))

    def card(parent, title, value, col=0, row=0):
        f = tk.Frame(parent, bg=CARD_BG, padx=18, pady=14, relief="flat")
        f.grid(row=row, column=col, padx=10, pady=8, sticky="nsew")
        tk.Label(f, text=title, bg=CARD_BG, fg=SUBTEXT, font=("Segoe UI", 9)).pack(anchor="w")
        tk.Label(f, text=value,  bg=CARD_BG, fg=ACCENT,  font=("Segoe UI", 16, "bold")).pack(anchor="w")

    grid = tk.Frame(inner_s, bg=DARK_BG)
    grid.pack(padx=20, pady=20, fill="x")
    for i in range(4): grid.columnconfigure(i, weight=1)

    stats = report["stats"]
    card(grid, "Total Reviews",      f"{report['final_shape'][0]:,}",                    0, 0)
    card(grid, "Duplicates Removed", f"{report['duplicates_removed']:,}",                1, 0)
    card(grid, "Avg Satisfaction",   f"{stats.loc['mean','Satisfaction']:.2f} / 5",      2, 0)
    card(grid, "Avg Effectiveness",  f"{stats.loc['mean','Effectiveness']:.2f} / 5",     3, 0)
    card(grid, "Avg Ease of Use",    f"{stats.loc['mean','EaseofUse']:.2f} / 5",         0, 1)
    card(grid, "Avg Review Length",  f"{stats.loc['mean','Review_Length']:.0f} chars",   1, 1)
    card(grid, "Avg Useful Count",   f"{stats.loc['mean','UsefulCount']:.2f}",           2, 1)
    card(grid, "Missing (before)",   str(report["missing_before"]),                      3, 1)

    # Descriptive stats table
    tk.Label(inner_s, text="Descriptive Statistics", bg=DARK_BG, fg=TEXT,
             font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=20, pady=(16, 4))

    stat_frame = tk.Frame(inner_s, bg=DARK_BG)
    stat_frame.pack(fill="x", padx=20, pady=(0, 16))

    stat_df = report["stats"].reset_index()
    cols = list(stat_df.columns)
    tree = ttk.Treeview(stat_frame, columns=cols, show="headings", height=8)
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=130, anchor="center")
    for _, row_data in stat_df.iterrows():
        tree.insert("", "end", values=list(row_data))
    vsb = ttk.Scrollbar(stat_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    tree.pack(fill="x")

    # ── Tab 2: Visualizations ───────────────────────────────
    tab_viz = ttk.Frame(notebook)
    notebook.add(tab_viz, text="  Visualizations  ")

    left_panel = tk.Frame(tab_viz, bg=PANEL_BG, width=200)
    left_panel.pack(side="left", fill="y")
    left_panel.pack_propagate(False)

    tk.Label(left_panel, text="Charts", bg=PANEL_BG, fg=ACCENT,
             font=("Segoe UI", 11, "bold")).pack(pady=(16, 8), padx=10)

    right_panel = tk.Frame(tab_viz, bg=DARK_BG)
    right_panel.pack(side="right", fill="both", expand=True)

    img_label = tk.Label(right_panel, bg=DARK_BG)
    img_label.pack(fill="both", expand=True, padx=10, pady=10)

    _photo_ref = [None]

    def show_plot(path):
        try:
            img = Image.open(path)
            w, h = right_panel.winfo_width() - 20, right_panel.winfo_height() - 20
            if w < 100: w, h = 900, 550
            img.thumbnail((w, h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            _photo_ref[0] = photo
            img_label.configure(image=photo)
        except Exception as ex:
            img_label.configure(text=f"Could not load image:\n{ex}", fg="red")

    btn_style = {"bg": CARD_BG, "fg": TEXT, "activebackground": ACCENT,
                 "activeforeground": "#fff", "relief": "flat",
                 "font": ("Segoe UI", 9), "cursor": "hand2",
                 "anchor": "w", "padx": 12, "pady": 6}

    first_path = [None]
    for name, path in plots.items():
        b = tk.Button(left_panel, text=f"  {name}", **btn_style,
                      command=lambda p=path: show_plot(p))
        b.pack(fill="x", padx=8, pady=2)
        if first_path[0] is None:
            first_path[0] = path

    # show first plot after window renders
    root.after(200, lambda: show_plot(first_path[0]))
    right_panel.bind("<Configure>", lambda e: show_plot(_photo_ref[0] and first_path[0]))

    # ── Tab 3: Data Preview ─────────────────────────────────
    tab_data = ttk.Frame(notebook)
    notebook.add(tab_data, text="  Data Preview  ")

    tk.Label(tab_data, text="Cleaned Dataset — First 500 Rows",
             bg=DARK_BG, fg=TEXT, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=8)

    preview_frame = tk.Frame(tab_data, bg=DARK_BG)
    preview_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    preview_cols = ["Age", "Sex", "Condition", "Drug", "EaseofUse",
                    "Effectiveness", "Satisfaction", "UsefulCount", "Year", "Review_Length"]
    ptree = ttk.Treeview(preview_frame, columns=preview_cols, show="headings", height=25)
    col_widths = {"Age": 80, "Sex": 70, "Condition": 160, "Drug": 160,
                  "EaseofUse": 80, "Effectiveness": 90, "Satisfaction": 90,
                  "UsefulCount": 90, "Year": 60, "Review_Length": 100}
    for c in preview_cols:
        ptree.heading(c, text=c)
        ptree.column(c, width=col_widths.get(c, 100), anchor="center")
    for _, row_data in df[preview_cols].head(500).iterrows():
        ptree.insert("", "end", values=list(row_data))

    ph = ttk.Scrollbar(preview_frame, orient="horizontal", command=ptree.xview)
    pv = ttk.Scrollbar(preview_frame, orient="vertical",   command=ptree.yview)
    ptree.configure(xscrollcommand=ph.set, yscrollcommand=pv.set)
    ph.pack(side="bottom", fill="x")
    pv.pack(side="right",  fill="y")
    ptree.pack(fill="both", expand=True)

    root.mainloop()

# ============================================================
# 5. Save cleaned data for Phase 2
# ============================================================
def save_cleaned(df):
    out = "webmd_cleaned.csv"
    df.to_csv(out, index=False)
    print(f"Cleaned data saved to '{out}' — ready for Phase 2")

# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    df = load_data()
    df, report = clean_data(df)
    plots = generate_plots(df)
    save_cleaned(df)
    launch_gui(df, report, plots)
