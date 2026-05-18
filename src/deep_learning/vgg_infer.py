"""
Inference for the collaborator-trained VGG16 model (HAM10000, 7-class).

Label order in the VGG16 checkpoint (collaborators/vgg16/config.py):
    ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]   ← his order

Our canonical alphabetical order (config.HAM10000_DX_LABELS):
    ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]   ← our order

This module remaps probabilities from his order to ours so the rest of the
pipeline (binary collapse, confidence router, UI) behaves identically.

No calibration JSON is available for VGG16 (temperature = 1.0 = identity).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import VGG16_Weights

import config
from src.deep_learning.infer import InferenceResult, _binary_collapse, _softmax

# ── Label remapping ────────────────────────────────────────────────────────────
# His label order (from collaborators/vgg16/config.py CLASS_NAMES)
_VGG_LABELS = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]

# Build a permutation vector: _REMAP[i] = index in OUR order of his i-th output
# e.g. his index 0 = "mel" → our index 4 → _REMAP[0] = 4
_CANONICAL = config.HAM10000_DX_LABELS           # alphabetical
_REMAP: list[int] = [_CANONICAL.index(lbl) for lbl in _VGG_LABELS]
# _REMAP = [4, 5, 1, 0, 2, 3, 6]

# ── Preprocessing (matches his training pipeline) ──────────────────────────────
_VGG_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet
        std=[0.229, 0.224, 0.225],
    ),
])


def _build_vgg16() -> nn.Module:
    """Reconstruct the VGG16 architecture used during training."""
    model = models.vgg16(weights=None)                       # no pretrained weights
    model.classifier[6] = nn.Linear(4096, len(_VGG_LABELS)) # 7-class head
    return model


def _load_vgg16_checkpoint(checkpoint_path: Path, device: str) -> nn.Module:
    """
    Load a VGG16 checkpoint.

    Handles two save formats:
      1. Plain state_dict (what torch.save(model.state_dict(), path) produces)
      2. Dict with 'model_state_dict' key (what our README recommends)
    """
    raw = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    else:
        # Assume it's a plain state_dict
        state_dict = raw

    model = _build_vgg16()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def vgg_infer(
    image: Image.Image,
    checkpoint_path: Path,
    device: Optional[str] = None,
) -> InferenceResult:
    """
    Run the VGG16 collaborator checkpoint on a single PIL image.

    Returns an InferenceResult with probabilities remapped to the canonical
    alphabetical label order so it is a drop-in replacement for vit_infer().

    Note: No temperature calibration is applied (T=1.0).  The model's raw
    softmax is used directly.  If a calibration JSON is added later, wire it
    in here.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_vgg16_checkpoint(checkpoint_path, device)

    # Preprocess
    tensor = _VGG_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)

    # Forward pass
    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        logits_his_order = model(tensor).cpu().float().numpy()[0]  # shape (7,)

    # Softmax in his label space (T=1.0, no calibration)
    probs_his_order = _softmax(logits_his_order)

    # Remap to our canonical label order
    dx_probs = np.zeros(len(_CANONICAL), dtype=np.float32)
    for his_idx, our_idx in enumerate(_REMAP):
        dx_probs[our_idx] = probs_his_order[his_idx]

    # Also remap logits for completeness (debug only)
    logits_canonical = np.zeros(len(_CANONICAL), dtype=np.float32)
    for his_idx, our_idx in enumerate(_REMAP):
        logits_canonical[our_idx] = logits_his_order[his_idx]

    pred_dx_idx = int(np.argmax(dx_probs))
    binary_probs, bin_idx, bin_label, confidence, mal_prob = _binary_collapse(dx_probs)

    return InferenceResult(
        dx_probs=dx_probs,
        dx_labels=_CANONICAL,
        predicted_dx_idx=pred_dx_idx,
        predicted_dx=_CANONICAL[pred_dx_idx],
        binary_probs=binary_probs,
        predicted_class_idx=bin_idx,
        predicted_label=bin_label,
        confidence=confidence,
        malignant_prob=mal_prob,
        temperature=1.0,   # uncalibrated — T=1.0 is identity
        logits=logits_canonical,
    )
