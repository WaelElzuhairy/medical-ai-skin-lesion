"""
CLI wrapper that kicks off CNN training.

Run:
    python scripts/train_cnn.py v2          # 7-class from ImageNet weights
    python scripts/train_cnn.py v3          # focal loss warm-start from v2
    python scripts/train_cnn.py v3 v2       # explicit: version_tag warm_start_tag
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deep_learning.train import train, train_v3  # noqa: E402


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v2"
    if tag == "v3" or (len(sys.argv) > 2):
        warm = sys.argv[2] if len(sys.argv) > 2 else "v2"
        train_v3(warm_start_tag=warm, version_tag=tag)
    else:
        train(version_tag=tag)
