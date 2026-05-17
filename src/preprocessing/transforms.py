"""
Inference-time preprocessing: resize + ImageNet normalize, nothing else.

Per the project plan, augmentation is forbidden at inference time. That rule
is physically enforced by keeping the inference pipeline in this module and
the training augmentations in `augmentation.py` — the inference path cannot
accidentally pick up an augmenter.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import config


def build_inference_transform() -> transforms.Compose:
    """Resize -> Tensor -> ImageNet-normalize. No randomness."""
    return transforms.Compose([
        transforms.Resize((config.CNN_INPUT_SIZE, config.CNN_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def preprocess(img: Image.Image) -> torch.Tensor:
    """Convert a PIL.Image to a model-ready tensor with a leading batch dim."""
    tfm = build_inference_transform()
    return tfm(img).unsqueeze(0)


def to_displayable(tensor: torch.Tensor) -> np.ndarray:
    """Reverse normalization for visualization (e.g., Grad-CAM overlays)."""
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    img = tensor.detach().cpu().squeeze(0) * std + mean
    img = img.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (img * 255.0).astype(np.uint8)
