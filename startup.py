"""
startup.py — Downloads model checkpoints from HF Hub if not present locally.

Called automatically by the Streamlit app entry point before any page loads.
Safe to call multiple times (no-op if files already exist).
"""

from __future__ import annotations
import os
from pathlib import Path

HF_REPO   = "Wael-Elzuhairy/medical-ai-models"
CKPT_DIR  = Path("models/checkpoints")
MODELS    = ["vit_v4.pt", "vgg16.pt"]


def download_models() -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [m for m in MODELS if not (CKPT_DIR / m).exists()]
    if not missing:
        return

    print(f"Downloading {len(missing)} model(s) from HF Hub: {missing}")
    try:
        from huggingface_hub import hf_hub_download
        for fname in missing:
            print(f"  Fetching {fname} …")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=fname,
                repo_type="model",
                local_dir=str(CKPT_DIR),
                local_dir_use_symlinks=False,
            )
            print(f"  {fname} saved to {CKPT_DIR / fname}")
    except Exception as e:
        print(f"Warning: model download failed — {e}")
        print("Models must be present in models/checkpoints/ to run inference.")


if __name__ == "__main__":
    download_models()
