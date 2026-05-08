# ============================================================
# rag/retriever.py — Semantic search against the ChromaDB index
# ============================================================

from __future__ import annotations

import re
from dataclasses import dataclass

from webmd.config import RAG_TOP_K


@dataclass
class Hit:
    """A single retrieved review with its metadata and similarity score."""
    document:     str
    drug:         str
    condition:    str
    satisfaction: str
    effectiveness:str
    sides:        str
    age:          str
    sex:          str
    similarity:   float

    @property
    def review_text(self) -> str:
        """Extract the raw review text from the embedded document string."""
        match = re.search(r"Review: (.+)", self.document)
        return match.group(1) if match else self.document

    @property
    def satisfaction_float(self) -> float:
        """Return satisfaction as float, or 0.0 if unparseable."""
        try:
            return float(self.satisfaction)
        except ValueError:
            return 0.0


def retrieve(
    query: str,
    col,
    embedder,
    top_k: int = RAG_TOP_K,
    drug_filter: str | None = None,
    condition_filter: str | None = None,
) -> list[Hit]:
    """Semantic search with optional metadata filters.

    Args:
        query:            Natural language query string.
        col:              ChromaDB Collection object.
        embedder:         Fitted SentenceTransformer instance.
        top_k:            Number of results to return.
        drug_filter:      Exact drug name filter (lowercased).
        condition_filter: Exact condition filter (lowercased).

    Returns:
        List of Hit dataclasses sorted by descending similarity.
    """
    where: dict = {}
    if drug_filter:
        where["Drug"] = {"$eq": drug_filter.strip().lower()}
    if condition_filter:
        where["Condition"] = {"$eq": condition_filter.strip().lower()}

    query_emb = embedder.encode([query]).tolist()

    kwargs: dict = dict(
        query_embeddings=query_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = col.query(**kwargs)

    hits: list[Hit] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append(Hit(
            document=doc,
            drug=meta.get("Drug", ""),
            condition=meta.get("Condition", ""),
            satisfaction=meta.get("Satisfaction", ""),
            effectiveness=meta.get("Effectiveness", ""),
            sides=meta.get("Sides", ""),
            age=meta.get("Age", ""),
            sex=meta.get("Sex", ""),
            similarity=round(1 - dist, 3),  # cosine distance → similarity
        ))

    return hits
