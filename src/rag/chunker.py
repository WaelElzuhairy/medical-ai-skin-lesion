"""
Text chunker for the RAG corpus.

Splits a document string into overlapping token-counted chunks.
Uses tiktoken (cl100k_base) for counting — model-agnostic.

Config-driven:
  CHUNK_SIZE_TOKENS    = 500  (from config.py)
  CHUNK_OVERLAP_TOKENS = 50   (from config.py)
"""

from __future__ import annotations

import tiktoken

import config

_enc = tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str,
    doi: str = "",
    pub_date: str = "",
    source_type: str = "abstract",
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    """Split *text* into overlapping chunks.

    Parameters
    ----------
    text:        raw document text
    doi:         DOI string (carried into every chunk's metadata)
    pub_date:    publication date string e.g. "2023"
    source_type: "abstract" | "guideline" | "fulltext"
    chunk_size:  override config.CHUNK_SIZE_TOKENS
    overlap:     override config.CHUNK_OVERLAP_TOKENS

    Returns
    -------
    list of dicts, each: {text, doi, pub_date, source_type, chunk_idx}
    """
    chunk_size = chunk_size or config.CHUNK_SIZE_TOKENS
    overlap    = overlap    or config.CHUNK_OVERLAP_TOKENS

    tokens = _enc.encode(text)
    if not tokens:
        return []

    chunks   = []
    start    = 0
    chunk_idx = 0

    while start < len(tokens):
        end   = min(start + chunk_size, len(tokens))
        chunk = _enc.decode(tokens[start:end])
        chunks.append({
            "text":        chunk,
            "doi":         doi,
            "pub_date":    pub_date,
            "source_type": source_type,
            "chunk_idx":   chunk_idx,
        })
        chunk_idx += 1
        if end == len(tokens):
            break
        start = end - overlap   # slide back by overlap

    return chunks
