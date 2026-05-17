"""
Grad-CAM heatmap generation.

EfficientNet-B4: targets the final conv block (conv_head) — that is where
spatial information is most aligned with semantic class predictions.

ViT-Base-16: uses the last transformer encoder block's attention output with
a reshape transform to convert flattened patch tokens back to 2-D space.
Falls back to showing the original image if the CAM computation fails.

Per the project plan, the heatmap helps clinicians understand which image
region drove the prediction; it is never used as a hard clinical signal.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config
from src.deep_learning.model import target_layer_for_gradcam
from src.preprocessing.transforms import preprocess


# ---------------------------------------------------------------------------
# EfficientNet Grad-CAM (original)
# ---------------------------------------------------------------------------

def generate_heatmap(
    model: torch.nn.Module,
    image: Image.Image,
    target_class_idx: int,
    device: str | None = None,
) -> np.ndarray:
    """Return an RGB uint8 overlay (heatmap blended onto the resized original).

    Works for EfficientNet-B4 checkpoints.
    """
    device = device or str(next(model.parameters()).device)
    model.eval()

    input_tensor = preprocess(image).to(device)

    cam = GradCAM(
        model=model,
        target_layers=[target_layer_for_gradcam(model)],
    )
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(target_class_idx)],
    )[0]

    resized = image.resize((config.CNN_INPUT_SIZE, config.CNN_INPUT_SIZE))
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
    return overlay  # HxWx3 uint8


# ---------------------------------------------------------------------------
# ViT Grad-CAM
# ---------------------------------------------------------------------------

def _vit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14):
    """Reshape ViT token sequence back to a 2-D spatial map.

    ViT encoder outputs shape (batch, num_patches + 1, hidden_dim).
    Slice off the CLS token (index 0) then reshape to (B, H, W, C) → (B, C, H, W).
    """
    # Drop CLS token, reshape patches to spatial grid
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # (B, H, W, C) → (B, C, H, W)
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class _ViTLogitsWrapper(torch.nn.Module):
    """Thin wrapper so GradCAM gets a plain Tensor from a HF ViT model."""

    def __init__(self, vit_model):
        super().__init__()
        self._model = vit_model

    def forward(self, x):
        return self._model(pixel_values=x).logits


def generate_vit_heatmap(
    model,
    image: Image.Image,
    target_class_idx: int,
    processor,
    device: str | None = None,
) -> np.ndarray:
    """Return an RGB uint8 overlay for a ViT checkpoint.

    If CAM computation fails (e.g., gradient tracking issue), returns the
    resized original image so the UI never crashes.

    Parameters
    ----------
    model:             HF AutoModelForImageClassification loaded by load_vit_checkpoint
    image:             PIL.Image (RGB)
    target_class_idx:  index of the predicted class (0-6)
    processor:         HF AutoImageProcessor for the same model
    device:            'cuda', 'cpu', or None
    """
    device = device or str(next(model.parameters()).device)

    # Prepare input
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # Resize for overlay (ViT operates at VIT_INPUT_SIZE)
    resized = image.resize((config.VIT_INPUT_SIZE, config.VIT_INPUT_SIZE))
    rgb_f32 = np.asarray(resized, dtype=np.float32) / 255.0

    try:
        wrapped = _ViTLogitsWrapper(model).to(device)
        wrapped.eval()

        # Target: last encoder block's layer norm output (good spatial signal)
        target_layer = model.vit.encoder.layer[-1].layernorm_before

        cam = GradCAM(
            model=wrapped,
            target_layers=[target_layer],
            reshape_transform=_vit_reshape_transform,
        )
        grayscale_cam = cam(
            input_tensor=pixel_values,
            targets=[ClassifierOutputTarget(target_class_idx)],
        )[0]

        overlay = show_cam_on_image(rgb_f32, grayscale_cam, use_rgb=True)
        return overlay  # HxWx3 uint8

    except Exception:
        # Graceful fallback: return the plain resized image
        return (rgb_f32 * 255).astype(np.uint8)
