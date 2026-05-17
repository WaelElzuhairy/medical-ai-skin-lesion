"""
Inference entry point. ALWAYS applies temperature-scaled softmax.

For 7-class models the result includes BOTH the full 7-class breakdown AND
a binary benign/malignant collapse used by the confidence router:

    malignant_prob = sum of P(akiec) + P(bcc) + P(mel)
    benign_prob    = sum of P(bkl)   + P(df)  + P(nv) + P(vasc)
    confidence     = malignant_prob  (what the router sees)

Per the project plan, raw softmax is forbidden downstream. The router, Guard
Agent, and reporting pipeline only ever receive calibrated probabilities. The
raw `logits` field is debug-only — never passed into the agentic layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

import config
from src.deep_learning.calibration import load_temperature
from src.deep_learning.model import load_checkpoint
from src.preprocessing.transforms import preprocess


@dataclass
class InferenceResult:
    # --- 7-class output ---
    dx_probs: np.ndarray            # shape (7,) calibrated, sums to 1
    dx_labels: list[str]            # config.HAM10000_DX_LABELS
    predicted_dx_idx: int           # argmax of dx_probs
    predicted_dx: str               # e.g. "mel"

    # --- Binary collapse (for router) ---
    binary_probs: np.ndarray        # shape (2,) [benign, malignant]
    predicted_class_idx: int        # 0=benign, 1=malignant
    predicted_label: str            # "benign" or "malignant"
    confidence: float               # max(P(benign), P(malignant)) — router uses this
    malignant_prob: float           # P(malignant) — shown in clinical display

    # --- Calibration ---
    temperature: float
    logits: np.ndarray              # raw 7-class logits, debug-only

    # --- Backwards-compat alias ---
    calibrated_probs: np.ndarray = field(init=False)
    class_labels: list[str] = field(init=False)

    def __post_init__(self):
        # Aliases so existing Streamlit / eval code keeps working
        self.calibrated_probs = self.dx_probs
        self.class_labels = self.dx_labels


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _binary_collapse(dx_probs: np.ndarray) -> tuple[np.ndarray, int, str, float, float]:
    """Collapse 7-class probs to binary [benign, malignant] via index sum.

    Returns:
        binary_probs, pred_idx, pred_label, confidence, malignant_prob

    confidence = max(P(benign), P(malignant)) — overall certainty, used by router.
    malignant_prob = P(malignant) — used for clinical display only.
    """
    mal_prob   = float(dx_probs[config.MALIGNANT_CLASS_INDICES].sum())
    ben_prob   = 1.0 - mal_prob
    binary     = np.array([ben_prob, mal_prob], dtype=np.float32)
    pred_idx   = int(np.argmax(binary))
    confidence = float(np.max(binary))   # max(P(benign), P(malignant))
    return binary, pred_idx, config.BINARY_LABELS[pred_idx], confidence, mal_prob


@torch.no_grad()
def infer(
    image: Image.Image,
    checkpoint_path: Path,
    version_tag: str = "v2",
    device: Optional[str] = None,
) -> InferenceResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(checkpoint_path, device=device)
    T = load_temperature(version_tag=version_tag)

    x = preprocess(image).to(device)
    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        logits = model(x).cpu().float().numpy()[0]

    dx_probs = _softmax(logits / T)
    pred_dx_idx = int(np.argmax(dx_probs))

    # Use labels stored in checkpoint if available, fall back to config
    dx_labels = meta.get("dx_labels", config.HAM10000_DX_LABELS)

    binary_probs, bin_idx, bin_label, confidence, mal_prob = _binary_collapse(dx_probs)

    return InferenceResult(
        dx_probs=dx_probs,
        dx_labels=dx_labels,
        predicted_dx_idx=pred_dx_idx,
        predicted_dx=dx_labels[pred_dx_idx],
        binary_probs=binary_probs,
        predicted_class_idx=bin_idx,
        predicted_label=bin_label,
        confidence=confidence,
        malignant_prob=mal_prob,
        temperature=T,
        logits=logits,
    )
