"""
Sentence embedder for the RAG corpus.

Uses pritamdeka/S-PubMedBert-MS-MARCO (domain-tuned for biomedical retrieval).
Model is loaded once and cached on first call.

Provides:
  embed(texts)  -> np.ndarray  shape (N, dim)
  embed_one(text) -> np.ndarray shape (dim,)
"""

from __future__ import annotations

import numpy as np

import config

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """Embed a list of strings.

    Returns
    -------
    np.ndarray, shape (len(texts), embedding_dim), dtype float32
    """
    if not texts:
        return np.empty((0,), dtype=np.float32)
    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # unit-norm → cosine = dot product
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def embed_one(text: str) -> np.ndarray:
    """Embed a single string. Returns 1-D array."""
    return embed([text])[0]
