"""
Fit temperature scaling on the saved validation logits and report ECE.

Run AFTER training:
    python scripts/calibrate_model.py [version_tag]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.deep_learning.calibration import calibrate_from_saved_logits  # noqa: E402


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v2"
    calibrate_from_saved_logits(version_tag=tag)
