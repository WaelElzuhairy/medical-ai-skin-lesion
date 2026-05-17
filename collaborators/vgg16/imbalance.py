"""
imbalance.py — Centralised class-imbalance handling for HAM10000.

Four selectable strategies (set IMBALANCE_STRATEGY in config.py):
  A  weighted_loss    — per-class loss weights penalise minority misclassification
  B  oversampling     — WeightedRandomSampler equalises class frequency per batch
  C  augment_minority — heavy augmentation applied only to non-majority classes
  D  combined         — A + B + C together (recommended for HAM10000)
"""

import warnings
from typing import Dict, List, Optional

import torch
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms

from config import CLASS_NAMES, DEVICE, IMBALANCE_STRATEGY, NUM_CLASSES

# ── Valid strategy names ──────────────────────────────────────────────────────
VALID_STRATEGIES = {"weighted_loss", "oversampling", "augment_minority", "combined"}

# ── ImageNet normalisation constants ─────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# ── Pre-built transform pipelines ────────────────────────────────────────────
# Standard training transform — light augmentation, used for majority class
# and for strategies that don't differentiate by class.
_STANDARD_TRAIN = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# Heavy augmentation transform — applied to minority classes under
# augment_minority and combined strategies.
_HEAVY_AUGMENT = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=15, shear=10),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# Validation / test transform — deterministic, no augmentation.
_VAL_TEST = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


# ── Public helpers ────────────────────────────────────────────────────────────

def get_class_weights(class_counts: Dict[str, int]) -> torch.Tensor:
    """
    Return per-class loss weights as a FloatTensor on DEVICE.

    weighted_loss / combined:  weight[c] = total / (num_classes * count[c])
    oversampling / augment_minority:  uniform weights (1.0 for every class)
    """
    strategy = IMBALANCE_STRATEGY
    total = sum(class_counts.values())

    if strategy in ("weighted_loss", "combined"):
        weights = [
            total / (NUM_CLASSES * max(class_counts.get(cls, 1), 1))
            for cls in CLASS_NAMES
        ]
    else:
        weights = [1.0] * NUM_CLASSES

    return torch.FloatTensor(weights).to(DEVICE)


def get_sampler(labels: List[int]) -> Optional[WeightedRandomSampler]:
    """
    Return a WeightedRandomSampler for oversampling / combined, else None.

    Each sample receives weight 1/count(its_class), so every class is drawn
    with equal probability per epoch regardless of original frequency.
    """
    strategy = IMBALANCE_STRATEGY

    if strategy not in ("oversampling", "combined"):
        return None

    # Count occurrences of each integer label
    class_counts: Dict[int, int] = {}
    for lbl in labels:
        class_counts[lbl] = class_counts.get(lbl, 0) + 1

    sample_weights = [1.0 / max(class_counts[lbl], 1) for lbl in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )


def get_minority_transform(is_minority: bool) -> transforms.Compose:
    """
    Return the appropriate training transform for a single sample.

    augment_minority / combined + minority class  →  heavy augmentation
    all other cases                               →  standard train transform
    """
    strategy = IMBALANCE_STRATEGY
    if strategy in ("augment_minority", "combined") and is_minority:
        return _HEAVY_AUGMENT
    return _STANDARD_TRAIN


def get_val_transform() -> transforms.Compose:
    """Deterministic val/test transform (no augmentation)."""
    return _VAL_TEST


def print_imbalance_summary(class_counts: Dict[str, int], strategy: str) -> None:
    """
    Print a formatted table of class distribution and explain the active strategy.
    Issues a UserWarning if strategy is not one of the four valid options.
    """
    if strategy not in VALID_STRATEGIES:
        warnings.warn(
            f"\n[imbalance] Invalid IMBALANCE_STRATEGY='{strategy}'.\n"
            f"  Must be one of {sorted(VALID_STRATEGIES)}.\n"
            f"  Falling back to no imbalance handling (uniform weights, no sampler).",
            UserWarning,
            stacklevel=2,
        )

    total = sum(class_counts.values())

    _STRATEGY_DESC = {
        "weighted_loss":    "upweights minority-class errors in CrossEntropyLoss",
        "oversampling":     "oversamples minority classes via WeightedRandomSampler",
        "augment_minority": "applies heavy augmentation exclusively to minority classes",
        "combined":         "weighted_loss + oversampling + augment_minority (recommended)",
    }

    def _label_for(cls: str) -> str:
        is_min = cls != "nv"
        if strategy == "weighted_loss":
            return "weighted loss"
        if strategy == "oversampling":
            return "oversampled" if is_min else "standard"
        if strategy == "augment_minority":
            return "heavy augment" if is_min else "standard augment"
        if strategy == "combined":
            return "weighted + oversample + heavy aug" if is_min else "weighted + standard aug"
        return "none (invalid strategy)"

    sep = "=" * 70
    print(f"\n{sep}")
    print("  HAM10000 — Class Distribution & Imbalance Strategy")
    print(sep)
    print(f"  {'Class':<8} {'Count':>7}  {'% Dataset':>10}  Strategy Applied")
    print("  " + "-" * 66)
    for cls in CLASS_NAMES:
        count = class_counts.get(cls, 0)
        pct   = 100.0 * count / total if total > 0 else 0.0
        print(f"  {cls:<8} {count:>7}  {pct:>9.1f}%  {_label_for(cls)}")
    print("  " + "-" * 66)
    print(f"  {'Total':<8} {total:>7}")
    print(sep)
    desc = _STRATEGY_DESC.get(strategy, "unknown — check config.py")
    print(f"  Active strategy : '{strategy}'")
    print(f"  Why it helps    : {desc}")
    print(f"{sep}\n")
