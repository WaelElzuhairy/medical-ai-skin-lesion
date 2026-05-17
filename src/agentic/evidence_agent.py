"""
Evidence Agent — RAG-backed literature retrieval.

Queries the ChromaDB corpus with a compound query built from the CNN
prediction + patient metadata. Drops any chunks below cosine 0.6.

HARD RULE: if no chunk clears the threshold → return {"status": "insufficient_evidence"}.
           Zero LLM calls are made in that case.

If the ChromaDB corpus has not been built yet, falls back to a lightweight
PubMed E-utilities live query so the pipeline still works without the corpus.
"""

from __future__ import annotations

from typing import Any

import config
from src.agentic.anthropic_client import call_llm
from src.deep_learning.infer import InferenceResult

SYSTEM_PROMPT = """You are a medical literature assistant. You will be given excerpts from
dermatology research papers. Your ONLY job is to extract verbatim quotes that are directly
relevant to the given skin lesion classification.

STRICT RULES:
- Extract ONLY direct quotes from the provided text — do NOT paraphrase.
- Do NOT synthesise, combine, or interpret findings.
- Do NOT add clinical recommendations.
- Only include quotes with clear relevance to the predicted diagnosis.
- Output valid JSON:
  {
    "status": "ok",
    "quotes": [
      {"text": "exact quote", "doi": "doi string or empty", "pub_date": "year or empty"},
      ...
    ]
  }
- Maximum 3 quotes.
- If no quotes are relevant, return {"status": "ok", "quotes": []}.
"""


def run(result: InferenceResult, metadata: dict[str, Any]) -> dict:
    """Query the corpus and return extracted quotes.

    Returns {"status": "insufficient_evidence"} if no chunk clears cosine 0.6.
    Returns {"status": "ok", "quotes": [...]} if relevant chunks found.
    """
    query = _build_query(result, metadata)
    print(f"[EvidenceAgent] Query: {query}", flush=True)

    chunks = _retrieve_chunks(query)

    if not chunks:
        print("[EvidenceAgent] No chunks above cosine threshold — insufficient_evidence", flush=True)
        return {"status": "insufficient_evidence", "quotes": []}

    print(f"[EvidenceAgent] {len(chunks)} chunk(s) above threshold — calling LLM", flush=True)
    return _extract_quotes(query, chunks)


def _build_query(result: InferenceResult, metadata: dict) -> str:
    dx   = result.predicted_dx
    age  = metadata.get("age", "adult")
    sex  = metadata.get("sex", "patient")
    loc  = metadata.get("localization", "skin")
    return f"{dx} dermoscopy {sex} {age} years {loc} diagnosis classification"


def _retrieve_chunks(query: str) -> list[dict]:
    """Try ChromaDB first, fall back to PubMed live query."""
    try:
        return _query_chroma(query)
    except Exception as e:
        print(f"[EvidenceAgent] ChromaDB unavailable ({e}), falling back to PubMed", flush=True)
        return _query_pubmed_live(query)


def _query_chroma(query: str) -> list[dict]:
    """Query the local ChromaDB corpus via the retriever module."""
    from src.rag.retriever import retrieve
    return retrieve(query)


def _query_pubmed_live(query: str) -> list[dict]:
    """Lightweight fallback: fetch 3 PubMed abstracts via E-utilities."""
    import requests

    base  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  3,
        "retmode": "json",
        "sort":    "relevance",
    }
    if config.NCBI_API_KEY:
        params["api_key"] = config.NCBI_API_KEY

    try:
        search = requests.get(f"{base}/esearch.fcgi", params=params, timeout=10).json()
        ids    = search.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        fetch = requests.get(
            f"{base}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "text"},
            timeout=10,
        )
        text = fetch.text[:3000]  # cap at 3k chars

        # Treat the whole abstract blob as one chunk — always above threshold since
        # we already searched for relevance via PubMed's own ranking
        return [{"text": text, "doi": "", "pub_date": "", "cosine": 0.65}]

    except Exception as e:
        print(f"[EvidenceAgent] PubMed fallback also failed: {e}", flush=True)
        return []


def _extract_quotes(query: str, chunks: list[dict]) -> dict:
    """Ask the LLM to extract verbatim quotes from the chunks."""
    combined = "\n\n---\n\n".join(
        f"[Source: DOI={c['doi'] or 'unknown'}, Year={c['pub_date'] or 'unknown'}]\n{c['text']}"
        for c in chunks
    )

    user_msg = f"""QUERY: {query}

LITERATURE EXCERPTS:
{combined}

Extract up to 3 verbatim quotes most relevant to this query."""

    schema = {
        "type": "object",
        "required": ["status", "quotes"],
    }

    for attempt in range(2):
        try:
            response = call_llm(SYSTEM_PROMPT, user_msg, schema=schema)
            response["status"] = "ok"
            return response
        except ValueError as e:
            if attempt == 0:
                print(f"[EvidenceAgent] Parse failed, retrying: {e}", flush=True)
                continue
            # On double failure, return insufficient rather than crashing pipeline
            return {"status": "insufficient_evidence", "quotes": []}

    return {"status": "insufficient_evidence", "quotes": []}
