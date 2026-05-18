"""
startup.py — Downloads model checkpoints and RAG corpus from HF Hub if not present.

Called automatically by the Streamlit app on first load.
Safe to call multiple times — skips files that already exist.
"""

from __future__ import annotations
from pathlib import Path

HF_REPO  = "Wael-Elzuhairy/medical-ai-models"
CKPT_DIR = Path("models/checkpoints")
CHROMA_DB = Path("data/corpus/chroma/chroma.sqlite3")

MODELS = ["vit_v4.pt", "vgg16.pt"]


def download_models() -> None:
    """Download model checkpoints if missing."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [m for m in MODELS if not (CKPT_DIR / m).exists()]
    if not missing:
        return

    print(f"[startup] Downloading {len(missing)} model(s): {missing}")
    try:
        from huggingface_hub import hf_hub_download
        for fname in missing:
            print(f"[startup]   Fetching {fname} …")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=fname,
                repo_type="model",
                local_dir=str(CKPT_DIR),
                local_dir_use_symlinks=False,
            )
            print(f"[startup]   {fname} ready.")
    except Exception as e:
        print(f"[startup] WARNING: model download failed — {e}")


def download_corpus() -> None:
    """Download ChromaDB sqlite file if missing."""
    if CHROMA_DB.exists():
        return

    CHROMA_DB.parent.mkdir(parents=True, exist_ok=True)
    print("[startup] Downloading RAG corpus (chroma.sqlite3) …")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id=HF_REPO,
            filename="chroma.sqlite3",
            repo_type="model",
            local_dir=str(CHROMA_DB.parent),
            local_dir_use_symlinks=False,
        )
        print("[startup] RAG corpus ready.")
    except Exception as e:
        print(f"[startup] WARNING: corpus download failed — {e}")
        print("[startup] Evidence Agent will return insufficient_evidence for all queries.")


def run() -> None:
    download_models()
    download_corpus()


if __name__ == "__main__":
    run()
