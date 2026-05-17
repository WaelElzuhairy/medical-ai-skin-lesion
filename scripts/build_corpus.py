"""
One-shot RAG corpus builder.

Usage:
  python scripts/build_corpus.py [--reset] [--max-per-term N] [--dry-run]

Steps:
  1. Fetch PubMed abstracts for all configured search terms
  2. Chunk each abstract (500 tokens, 50 overlap)
  3. Embed all chunks with S-PubMedBERT
  4. Upsert into ChromaDB

Run this once to build the corpus, then again with --reset to rebuild.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.rag.ingest_pubmed import fetch_all
from src.rag.chunker       import chunk_text
from src.rag.embedder      import embed
from src.rag.vector_store  import upsert_chunks, count, reset as reset_store


def main():
    parser = argparse.ArgumentParser(description="Build the RAG corpus for the Evidence Agent.")
    parser.add_argument("--reset",        action="store_true",  help="Wipe existing collection before ingesting")
    parser.add_argument("--max-per-term", type=int, default=None, help="Override PUBMED_MAX_RESULTS_PER_TERM")
    parser.add_argument("--dry-run",      action="store_true",  help="Fetch and chunk but do NOT write to Chroma")
    args = parser.parse_args()

    print("=" * 60)
    print("Medical-AI RAG Corpus Builder")
    print("=" * 60)
    print(f"  Embedding model : {config.EMBEDDING_MODEL}")
    print(f"  Chroma dir      : {config.CHROMA_DIR}")
    print(f"  Collection      : {config.CHROMA_COLLECTION}")
    print(f"  Chunk size      : {config.CHUNK_SIZE_TOKENS} tokens ({config.CHUNK_OVERLAP_TOKENS} overlap)")
    print(f"  Cosine gate     : {config.RETRIEVAL_MIN_COSINE}")
    print()

    # ---- Step 0: optionally wipe ----
    if args.reset and not args.dry_run:
        print("[reset] Wiping existing collection …")
        reset_store()
        print(f"[reset] Done. Current count: {count()}")

    existing = count()
    if existing > 0 and not args.reset:
        print(f"[info] Collection already has {existing} chunks. "
              "Use --reset to rebuild from scratch, or this run will ADD new records.")

    # ---- Step 1: fetch PubMed abstracts ----
    t0 = time.time()
    records = fetch_all(max_per_term=args.max_per_term, verbose=True)
    print(f"\n[fetch] {len(records)} unique abstracts in {time.time()-t0:.1f}s")

    if not records:
        print("[warn] No records fetched. Check network or NCBI availability.")
        sys.exit(1)

    # Persist raw records as JSONL for reproducibility
    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = config.CORPUS_DIR / "pubmed_records.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[save] Raw records -> {jsonl_path}")

    # ---- Step 2: chunk ----
    t1 = time.time()
    all_chunks: list[dict] = []
    for rec in records:
        text = f"{rec['title']}\n\n{rec['abstract']}".strip()
        chunks = chunk_text(
            text=text,
            doi=rec["doi"],
            pub_date=rec["pub_date"],
            source_type="abstract",
        )
        all_chunks.extend(chunks)
    print(f"[chunk] {len(all_chunks)} chunks from {len(records)} abstracts ({time.time()-t1:.1f}s)")

    if args.dry_run:
        print("[dry-run] Skipping embed + upsert.")
        print(f"  First chunk preview:\n    {all_chunks[0]['text'][:200]!r}")
        return

    # ---- Step 3: embed (batched) ----
    t2    = time.time()
    texts = [c["text"] for c in all_chunks]
    print(f"[embed] Embedding {len(texts)} chunks with {config.EMBEDDING_MODEL} …")
    vecs  = embed(texts, batch_size=64, show_progress=True)
    print(f"[embed] Done — shape {vecs.shape} ({time.time()-t2:.1f}s)")

    # ---- Step 4: upsert to ChromaDB ----
    t3         = time.time()
    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i : i + batch_size]
        batch_vecs   = vecs[i : i + batch_size]
        upsert_chunks(batch_chunks, batch_vecs)
        print(f"  Upserted {min(i+batch_size, len(all_chunks))}/{len(all_chunks)}")
    print(f"[upsert] Done ({time.time()-t3:.1f}s)")

    # ---- Summary ----
    final_count = count()
    print()
    print("=" * 60)
    print(f"Corpus ready: {final_count} chunks in ChromaDB")
    print(f"Total time: {time.time()-t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
