# ============================================================
# cli.py — Entry points wired to pyproject.toml [project.scripts]
#
# Usage (after `uv sync`):
#   uv run webmd-eda    # Phase 1: clean data + generate EDA plots
#   uv run webmd-ml     # Phase 2: train ML models + generate ML plots
#   uv run webmd-nlp    # Phase 3: train NLP models + generate NLP plots
#   uv run webmd-rag    # Phase 4+5: build RAG index + launch RAG GUI
#   uv run webmd-app    # All phases: launch unified dashboard
# ============================================================

from __future__ import annotations


def run_eda() -> None:
    """Phase 1 — EDA: clean raw data, save cleaned CSV, generate 8 plots."""
    from webmd.config import RAW_CSV, CLEANED_CSV, EDA_PLOTS_DIR
    from webmd.data.loader import load_raw
    from webmd.data.cleaner import clean_data, save_cleaned
    from webmd.analysis.plots import generate_eda_plots

    df = load_raw(RAW_CSV)
    df, _report = clean_data(df)
    save_cleaned(df, CLEANED_CSV)
    generate_eda_plots(df, EDA_PLOTS_DIR)


def run_ml() -> None:
    """Phase 2 — ML: train effectiveness models, save best, generate 5 plots."""
    # Populated in Phase 3
    raise NotImplementedError("Phase 2 not yet implemented — run after Phase 3 of refactor")


def run_nlp() -> None:
    """Phase 3 — NLP/DL: train TF-IDF + BiLSTM ensemble, save all artifacts, generate 6 plots."""
    # Populated in Phase 4
    raise NotImplementedError("Phase 3 not yet implemented — run after Phase 4 of refactor")


def run_rag() -> None:
    """Phase 4+5 — RAG+LLM: build ChromaDB index, launch RAG GUI."""
    # Populated in Phase 5
    raise NotImplementedError("Phase 4+5 not yet implemented — run after Phase 5 of refactor")


def run_app() -> None:
    """Launch the unified Tkinter dashboard (all phases)."""
    # Populated in Phase 6
    raise NotImplementedError("GUI not yet implemented — run after Phase 6 of refactor")
