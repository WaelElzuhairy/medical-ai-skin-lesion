"""
ChromaDB vector store wrapper.

Uses a persistent client stored at config.CHROMA_DIR.
All chunks are stored in a single collection (config.CHROMA_COLLECTION).

Public API:
  get_collection()         -> chromadb.Collection
  upsert_chunks(chunks, embeddings)
  count()                  -> int
  reset()                  -> deletes and recreates the collection
"""

from __future__ import annotations

import uuid

import numpy as np

import config

_client     = None
_collection = None


def _get_client():
    global _client
    if _client is None:
        import chromadb
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def get_collection():
    """Return (or create) the persistent ChromaDB collection."""
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def upsert_chunks(chunks: list[dict], embeddings: np.ndarray) -> None:
    """Insert or update chunks in the collection.

    Parameters
    ----------
    chunks:     list of dicts from chunker.chunk_text()
    embeddings: np.ndarray shape (N, dim) from embedder.embed()
    """
    if not chunks:
        return

    col  = get_collection()
    ids  = [str(uuid.uuid4()) for _ in chunks]
    docs = [c["text"] for c in chunks]
    metas = [
        {
            "doi":         c.get("doi", ""),
            "pub_date":    c.get("pub_date", ""),
            "source_type": c.get("source_type", "abstract"),
            "chunk_idx":   c.get("chunk_idx", 0),
        }
        for c in chunks
    ]

    col.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=docs,
        metadatas=metas,
    )


def count() -> int:
    """Return number of stored chunks."""
    return get_collection().count()


def reset() -> None:
    """Delete and recreate the collection (wipes all data)."""
    global _collection
    client = _get_client()
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:
        pass
    _collection = None
    get_collection()
