"""
RAG retriever — embed a query, query Chroma, filter by cosine threshold.

Critical rule: only chunks with cosine >= RETRIEVAL_MIN_COSINE (0.6) are
returned. If nothing clears the threshold, an empty list is returned, which
the Evidence Agent treats as "insufficient_evidence" without calling the LLM.

Public API:
  retrieve(query, top_k=None, min_cosine=None) -> list[dict]

Each returned dict:
  {text, doi, pub_date, source_type, chunk_idx, cosine}
"""

from __future__ import annotations

import config
from src.rag.embedder    import embed_one
from src.rag.vector_store import get_collection


def retrieve(
    query: str,
    top_k: int | None = None,
    min_cosine: float | None = None,
) -> list[dict]:
    """Query the vector store and return chunks above the relevance threshold.

    Parameters
    ----------
    query:      natural-language query string
    top_k:      number of candidates to pull from Chroma (default: config.RETRIEVAL_TOP_K)
    min_cosine: cosine threshold (default: config.RETRIEVAL_MIN_COSINE = 0.6)

    Returns
    -------
    Filtered list of chunk dicts, sorted by cosine descending.
    Returns [] if nothing clears the threshold.
    """
    top_k     = top_k     or config.RETRIEVAL_TOP_K
    min_cosine = min_cosine if min_cosine is not None else config.RETRIEVAL_MIN_COSINE

    col = get_collection()
    if col.count() == 0:
        return []

    query_vec = embed_one(query)

    results = col.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB with cosine space returns distance = 1 - cosine
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = []
    for doc, meta, dist in zip(docs, metas, distances):
        cosine = 1.0 - float(dist)
        if cosine >= min_cosine:
            chunks.append({
                "text":        doc,
                "doi":         meta.get("doi", ""),
                "pub_date":    meta.get("pub_date", ""),
                "source_type": meta.get("source_type", "abstract"),
                "chunk_idx":   meta.get("chunk_idx", 0),
                "cosine":      round(cosine, 4),
            })

    # Sort best-first
    chunks.sort(key=lambda x: x["cosine"], reverse=True)
    return chunks
