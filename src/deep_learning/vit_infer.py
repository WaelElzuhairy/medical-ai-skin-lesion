"""
Inference for the locally-trained ViT-Base-16 model.

Returns the same InferenceResult dataclass as src/deep_learning/infer.py so
the rest of the pipeline (confidence router, Streamlit UI, agents) needs no
changes — just call vit_infer() instead of infer() when a vit_*.pt checkpoint
is selected.

Temperature scaling is applied the same way as for EfficientNet:
    calibrated_logits = logits / T
    dx_probs = softmax(calibrated_logits)
    malignant_prob = sum of P(akiec) + P(bcc) + P(mel)   <- what the router sees

Raw logits are included in the result for debug/calibration purposes only and
MUST NOT be passed into the agentic layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

import config
from src.deep_learning.calibration import load_temperature
from src.deep_learning.infer import InferenceResult, _binary_collapse, _softmax
from src.deep_learning.vit_train import load_vit_checkpoint


@torch.no_grad()
def vit_infer(
    image: Image.Image,
    checkpoint_path: Path,
    version_tag: str = "vit_v1",
    device: Optional[str] = None,
) -> InferenceResult:
    """Run a locally-trained ViT checkpoint on a single PIL image.

    Parameters
    ----------
    image:           PIL.Image (RGB)
    checkpoint_path: Path to a vit_*.pt checkpoint saved by vit_train.train_vit()
    version_tag:     Matches the calibration JSON in models/calibration/<version_tag>.json
    device:          'cuda', 'cpu', or None (auto-detect)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, meta = load_vit_checkpoint(checkpoint_path, device=device)
    model.eval()

    T = load_temperature(version_tag=version_tag)

    # HF processor: resize + normalise with the model's own statistics
    processor = AutoImageProcessor.from_pretrained(
        meta.get("model_name", config.VIT_BASE_MODEL)
    )
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        logits = model(pixel_values=pixel_values).logits.cpu().float().numpy()[0]

    dx_probs     = _softmax(logits / T)
    pred_dx_idx  = int(np.argmax(dx_probs))
    dx_labels    = meta.get("dx_labels", config.HAM10000_DX_LABELS)

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
