"""
Training-time augmentation pipeline. NEVER imported by inference code.

Dermatoscopic images are rotation- and flip-invariant — a melanoma is a
melanoma at any orientation — so geometric augmentation is safe and helps
generalization on HAM10000's strong class imbalance. Color jitter is kept
mild because dermoscopy color cues carry diagnostic signal.

build_train_transform()       — standard augmentation (v2 training)
build_strong_train_transform() — stronger augmentation for v3 fine-tuning,
                                 targeting rare-class generalization
"""

from __future__ import annotations

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

import config


def build_train_transform() -> A.Compose:
    """Standard training augmentation."""
    return A.Compose([
        A.Resize(config.CNN_INPUT_SIZE, config.CNN_INPUT_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.5),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


def build_strong_train_transform() -> A.Compose:
    """Stronger augmentation for fine-tuning on rare classes.

    Additions over standard:
    - ElasticTransform + GridDistortion: simulate lesion shape variance
    - Stronger CoarseDropout: simulate dermoscopic hair / ruler artefacts
    - CLAHE: simulate different dermoscope lighting
    - Stronger color jitter: handle inter-device colour variation
    - RandomGamma + RandomBrightnessContrast: lighting robustness
    """
    return A.Compose([
        A.Resize(config.CNN_INPUT_SIZE, config.CNN_INPUT_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.6),
        A.OneOf([
            A.ElasticTransform(alpha=80, sigma=8, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
        ], p=0.4),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.CLAHE(clip_limit=3.0, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CoarseDropout(max_holes=8, max_height=48, max_width=48,
                        min_holes=2, min_height=16, min_width=16, p=0.5),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


def build_eval_transform() -> A.Compose:
    """Validation-time transform — same as inference but in albumentations form."""
    return A.Compose([
        A.Resize(config.CNN_INPUT_SIZE, config.CNN_INPUT_SIZE),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


def apply(transform: A.Compose, image_rgb: np.ndarray):
    """Albumentations expects a HWC uint8 RGB numpy array."""
    return transform(image=image_rgb)["image"]
