"""
CLI wrapper: fine-tune ViT-Base-16 on HAM10000 7-class task.

Run (from project root):
    python -u scripts/train_vit.py [version_tag]

Default version_tag = vit_v1.

After training completes, calibrate with:
    python scripts/calibrate_model.py vit_v1

And evaluate with:
    python scripts/evaluate.py vit_v1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deep_learning.vit_train import train_vit

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "vit_v1"
    print(f"Starting ViT training — version tag: {tag}", flush=True)
    best_ckpt = train_vit(version_tag=tag)
    print(f"\nDone.  Best checkpoint: {best_ckpt}", flush=True)
    print("\nNext steps:", flush=True)
    print(f"  python scripts/calibrate_model.py {tag}", flush=True)
    print(f"  python scripts/evaluate.py {tag}", flush=True)
