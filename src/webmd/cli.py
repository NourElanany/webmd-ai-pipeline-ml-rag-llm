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
    from webmd.config import CLEANED_CSV, ML_MODEL_PATH, ML_PLOTS_DIR
    from webmd.data.loader import load_cleaned
    from webmd.ml.features import build_features
    from webmd.ml.train import build_models, split_data, train_all, select_best, save_model
    from webmd.ml.plots import generate_ml_plots

    df      = load_cleaned(CLEANED_CSV)
    X, y, _ = build_features(df)
    splits  = split_data(X, y)
    results = train_all(build_models(), splits)
    best    = select_best(results)
    save_model(results[best].model, ML_MODEL_PATH)
    generate_ml_plots(results, best, splits, ML_PLOTS_DIR)


def run_nlp() -> None:
    """Phase 3 — NLP/DL: train TF-IDF + BiLSTM ensemble, save all artifacts, generate 6 plots."""
    import os
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

    from webmd.config import ARTIFACTS_DIR, CLEANED_CSV, NLP_PLOTS_DIR
    from webmd.nlp.ensemble import evaluate_ensemble
    from webmd.nlp.lstm import build_lstm_model, evaluate_lstm, save_lstm, train_lstm
    from webmd.nlp.plots import generate_nlp_plots
    from webmd.nlp.preprocess import load_nlp_data, prepare_data
    from webmd.nlp.tfidf import build_tfidf_model, evaluate_tfidf, save_tfidf

    df     = load_nlp_data(CLEANED_CSV)
    splits = prepare_data(df)

    # Train TF-IDF ensemble
    tfidf_artifacts = build_tfidf_model(
        splits.X_txt_train, splits.y_train,
        splits.X_txt_val,   splits.y_val,
    )
    save_tfidf(tfidf_artifacts, ARTIFACTS_DIR)
    tfidf_result = evaluate_tfidf(tfidf_artifacts, splits.X_txt_test, splits.y_test)

    # Train BiLSTM
    lstm_model, history = train_lstm(
        build_lstm_model(),
        splits.X_seq_train, splits.X_seq_val,
        splits.y_train,     splits.y_val,
    )
    save_lstm(lstm_model, splits.tokenizer)
    lstm_result = evaluate_lstm(lstm_model, splits.X_seq_test, splits.y_test)

    # Weighted ensemble evaluation
    ensemble_result = evaluate_ensemble(lstm_result.y_prob, tfidf_result.y_prob, splits.y_test)

    generate_nlp_plots(history, lstm_result, tfidf_result, ensemble_result, df, NLP_PLOTS_DIR)


def run_rag() -> None:
    """Phase 4+5 — RAG+LLM: build ChromaDB index, launch RAG GUI."""
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from sentence_transformers import SentenceTransformer  # heavy import

    from webmd.config import EMBED_MODEL, LLM_MODEL, OPENROUTER_API_KEY
    from webmd.rag.indexer import build_index, load_rag_data
    from webmd.rag.llm import build_client

    df         = load_rag_data()
    col        = build_index(df)
    embedder   = SentenceTransformer(EMBED_MODEL)
    llm_client = build_client(OPENROUTER_API_KEY)

    if llm_client:
        print(f"LLM ready: {LLM_MODEL}")
    else:
        print("No OPENROUTER_API_KEY found — running without LLM (template mode)")

    # GUI is imported here so heavy GUI deps don't load at CLI startup
    from webmd.gui.tabs.rag import launch_rag_window
    launch_rag_window(col, embedder, llm_client)


def run_app() -> None:
    """Launch the unified Tkinter dashboard (all phases)."""
    # Populated in Phase 6
    raise NotImplementedError("GUI not yet implemented — run after Phase 6 of refactor")
