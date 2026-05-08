# ============================================================
# rag/indexer.py — Build and load the ChromaDB vector index
# ============================================================

from __future__ import annotations

from pathlib import Path

import pandas as pd

from webmd.config import (
    CHROMA_DIR,
    CLEANED_CSV,
    COLLECTION_NAME,
    EMBED_MODEL,
    ML_RANDOM_STATE,
    RAG_SAMPLE_SIZE,
)

_EMBED_BATCH = 512
_META_COLS   = ["Drug", "Condition", "Satisfaction", "Effectiveness", "Sides", "Sex", "Age"]


def load_rag_data(path: Path = CLEANED_CSV, sample_size: int = RAG_SAMPLE_SIZE) -> pd.DataFrame:
    """Load and prepare the cleaned CSV for RAG indexing.

    - Drops reviews shorter than 20 characters.
    - Normalises Drug and Condition to lowercase.
    - Fills missing Sides with "not reported".
    - Samples *sample_size* rows for indexing speed.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {path}\n"
            "Run `uv run webmd-eda` first to generate it."
        )

    df = pd.read_csv(path)
    df = df[df["Reviews"].str.strip().str.len() > 20].copy()
    df = df.dropna(subset=["Drug", "Condition", "Reviews"])
    df["Drug"]      = df["Drug"].str.strip().str.lower()
    df["Condition"] = df["Condition"].str.strip().str.lower()
    df["Sides"]     = df["Sides"].fillna("not reported")

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=ML_RANDOM_STATE).reset_index(drop=True)

    print(
        f"Loaded {len(df):,} reviews | "
        f"{df['Drug'].nunique():,} drugs | "
        f"{df['Condition'].nunique():,} conditions"
    )
    return df


def build_document(row: pd.Series) -> str:
    """Combine row fields into a rich text document for embedding.

    Format is identical to the original rag_system.py so existing ChromaDB
    indexes remain compatible.
    """
    sentiment = (
        "positive" if row["Satisfaction"] >= 4
        else ("neutral" if row["Satisfaction"] == 3 else "negative")
    )
    return (
        f"Drug: {row['Drug']}. "
        f"Condition: {row['Condition']}. "
        f"Sentiment: {sentiment}. "
        f"Effectiveness: {row['Effectiveness']}/5. "
        f"Side effects: {row['Sides']}. "
        f"Review: {row['Reviews']}"
    )


def _get_client():
    """Return a ChromaDB PersistentClient pointed at CHROMA_DIR."""
    import chromadb  # heavy import — kept local
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build_index(df: pd.DataFrame, force_rebuild: bool = False):
    """Build (or load) the ChromaDB collection.

    If the collection already exists and *force_rebuild* is False, the
    existing index is returned immediately without re-embedding.

    Returns the ChromaDB Collection object.
    """
    from sentence_transformers import SentenceTransformer  # heavy import

    client   = _get_client()
    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME in existing and not force_rebuild:
        col = client.get_collection(COLLECTION_NAME)
        print(f"Loaded existing index: {col.count():,} documents")
        return col

    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    print(f"Building index with {EMBED_MODEL}...")
    embedder = SentenceTransformer(EMBED_MODEL)

    col = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    docs      = df.apply(build_document, axis=1).tolist()
    ids       = [str(i) for i in df.index]
    metadatas = [
        {k: str(v) for k, v in row.items()}
        for row in df[_META_COLS].to_dict("records")
    ]

    for start in range(0, len(docs), _EMBED_BATCH):
        end         = min(start + _EMBED_BATCH, len(docs))
        batch_emb   = embedder.encode(docs[start:end], show_progress_bar=False).tolist()
        col.add(
            documents=docs[start:end],
            embeddings=batch_emb,
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Indexed {end:,}/{len(docs):,}", end="\r")

    print(f"\nIndex built: {col.count():,} documents saved to '{CHROMA_DIR}/'")
    return col


def load_index():
    """Load an existing ChromaDB collection from disk.

    Raises FileNotFoundError if the index has not been built yet.
    """
    client   = _get_client()
    existing = [c.name for c in client.list_collections()]

    if COLLECTION_NAME not in existing:
        raise FileNotFoundError(
            f"ChromaDB collection '{COLLECTION_NAME}' not found in '{CHROMA_DIR}'.\n"
            "Run `uv run webmd-rag` first to build the index."
        )

    col = client.get_collection(COLLECTION_NAME)
    print(f"Loaded existing index: {col.count():,} documents")
    return col
