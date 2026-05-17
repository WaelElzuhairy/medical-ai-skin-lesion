"""
EfficientNet-B4 backbone via timm with a configurable classification head.

Defaults to a 2-class head for the binary benign/malignant task; can be
constructed with 7 classes for HAM10000 fine-grained dx prediction by
passing `num_classes=config.NUM_FINEGRAINED_CLASSES`. Exposes the final
convolutional block as the Grad-CAM target layer so explainability code
doesn't need to know the backbone's internal layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn

import config


def build_model(num_classes: int = config.NUM_FINEGRAINED_CLASSES, pretrained: bool = True) -> nn.Module:
    return timm.create_model(
        config.CNN_BACKBONE,
        pretrained=pretrained,
        num_classes=num_classes,
    )


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def get_param_groups(model: nn.Module, lr: float) -> list[dict]:
    """Return two param groups: backbone at lr*BACKBONE_LR_MULT, head at lr."""
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    return [
        {"params": backbone_params, "lr": lr * config.BACKBONE_LR_MULT},
        {"params": head_params,     "lr": lr},
    ]


def target_layer_for_gradcam(model: nn.Module) -> nn.Module:
    """Return the last convolutional block — the right hook point for Grad-CAM.

    For timm EfficientNet variants this is `conv_head` (post-block 1x1 conv).
    Falling back to the last entry in `model.blocks` keeps this robust to
    minor timm version differences.
    """
    if hasattr(model, "conv_head"):
        return model.conv_head
    if hasattr(model, "blocks"):
        return model.blocks[-1]
    raise AttributeError(f"Cannot locate Grad-CAM target layer on {type(model).__name__}")


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weights.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=0 reduces to weighted cross-entropy.
    gamma=2 (default) focuses heavily on hard/rare examples.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma          = gamma
        self.weight         = weight  # per-class alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard CE with label smoothing gives log-probs per sample
        ce_loss = nn.functional.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # p_t = exp(-CE) for the correct class (approximation)
        pt = torch.exp(-ce_loss)
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


def save_checkpoint(model: nn.Module, path: Path, extras: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "backbone": config.CNN_BACKBONE,
        "num_classes": model.get_classifier().out_features,
    }
    if extras:
        payload.update(extras)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    num_classes = payload.get("num_classes", config.NUM_BINARY_CLASSES)
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload
