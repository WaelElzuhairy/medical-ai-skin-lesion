"""
Fine-tune google/vit-base-patch16-224 on HAM10000 7-class task.

Architecture: Vision Transformer (ViT-Base/16) — same base as the
Anwarkh1/Skin_Cancer-Image_Classification HF model, but trained from
scratch on our leakage-free lesion-id-grouped split so we own the
training process end-to-end.

Training strategy:
  - Stage 1 (epochs 1..VIT_UNFREEZE_EPOCH):
      Freeze all transformer blocks + patch embedding; train only the
      classification head until it stabilises.
  - Stage 2 (epoch VIT_UNFREEZE_EPOCH + 1 onwards):
      Unfreeze the full backbone with differential LR:
        transformer blocks  → lr * VIT_BACKBONE_LR_MULT
        classifier head     → full lr
  - Class-weighted CrossEntropyLoss + WeightedRandomSampler for HAM10000
    imbalance (nv ~67 %, df ~1 %).
  - Label smoothing 0.1 to prevent overconfidence on 7-class head.
  - AMP (float16) for memory efficiency.
  - Saves best checkpoint (val macro-F1) + raw val logits for temperature
    calibration.  Full epoch history written to summary JSON.

Augmentation:
  Albumentations (geometric + colour) is applied BEFORE the HF processor
  so the processor can handle resize + normalisation with the model's own
  per-channel statistics.  The aug pipeline therefore omits Normalize and
  ToTensorV2 (the processor provides those).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoImageProcessor, AutoModelForImageClassification

import config


# ---------------------------------------------------------------------------
# Augmentation pipelines — NO Normalize / ToTensorV2 (processor handles that)
# ---------------------------------------------------------------------------

def _build_vit_train_aug() -> A.Compose:
    """Standard geometric + colour augmentation for ViT training."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.CLAHE(clip_limit=2.0, p=0.25),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32,
                        min_holes=1, min_height=8, min_width=8, p=0.3),
        # Output: HWC uint8 numpy (processor handles resize + normalise)
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ViTDataset(Dataset):
    """HAM10000 dataset compatible with the HuggingFace ViT image processor.

    The processor (AutoImageProcessor) handles:
      - Resizing to VIT_INPUT_SIZE × VIT_INPUT_SIZE
      - Normalisation with the model's per-channel mean / std

    Training images are augmented with albumentations BEFORE the processor.
    """

    def __init__(
        self,
        manifest_path: Path,
        processor: AutoImageProcessor,
        augment: bool = False,
    ) -> None:
        self.df = pd.read_parquet(manifest_path)
        self.processor = processor
        self.augment = augment
        self._aug = _build_vit_train_aug() if augment else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_np = np.array(Image.open(row["image_path"]).convert("RGB"))

        if self.augment and self._aug is not None:
            img_np = self._aug(image=img_np)["image"]   # still uint8 HWC

        inputs = self.processor(images=img_np, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # → (3, H, W)
        return pixel_values, int(row["dx_label"])


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_vit(device: torch.device) -> AutoModelForImageClassification:
    """Load ViT-Base-16 with a fresh 7-class classification head."""
    model = AutoModelForImageClassification.from_pretrained(
        config.VIT_BASE_MODEL,
        num_labels=config.NUM_FINEGRAINED_CLASSES,
        id2label={i: lbl for i, lbl in enumerate(config.HAM10000_DX_LABELS)},
        label2id={lbl: i for i, lbl in enumerate(config.HAM10000_DX_LABELS)},
        ignore_mismatched_sizes=True,   # replaces the 1000-class ImageNet head
    )
    return model.to(device)


def _freeze_vit_backbone(model: AutoModelForImageClassification) -> None:
    """Freeze patch embedding + all transformer encoder blocks."""
    for param in model.vit.parameters():
        param.requires_grad = False


def _unfreeze_vit_backbone(model: AutoModelForImageClassification) -> None:
    """Unfreeze the full transformer body."""
    for param in model.vit.parameters():
        param.requires_grad = True


def _get_vit_param_groups(
    model: AutoModelForImageClassification,
    lr: float,
) -> list[dict]:
    backbone_params = [p for p in model.vit.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr * config.VIT_BACKBONE_LR_MULT})
    groups.append({"params": head_params, "lr": lr})
    return groups


def save_vit_checkpoint(
    model: AutoModelForImageClassification,
    path: Path,
    extras: dict | None = None,
) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "model_name": config.VIT_BASE_MODEL,
        "num_labels": config.NUM_FINEGRAINED_CLASSES,
        "dx_labels":  config.HAM10000_DX_LABELS,
        **(extras or {}),
    }, path)


def load_vit_checkpoint(
    path: Path,
    device: str | torch.device = "cpu",
) -> tuple[AutoModelForImageClassification, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = AutoModelForImageClassification.from_pretrained(
        ckpt.get("model_name", config.VIT_BASE_MODEL),
        num_labels=ckpt.get("num_labels", config.NUM_FINEGRAINED_CLASSES),
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ckpt


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_vit(
    model: AutoModelForImageClassification,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_targets = [], []
    for pixel_values, y in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(pixel_values=pixel_values).logits.float()
        out = torch.nan_to_num(out, nan=0.0, posinf=30.0, neginf=-30.0)
        all_logits.append(out.cpu().numpy())
        all_targets.append(y.numpy())
    logits  = np.concatenate(all_logits,  axis=0)
    targets = np.concatenate(all_targets, axis=0)
    preds   = logits.argmax(axis=1)
    f1      = f1_score(targets, preds, average="macro")
    return f1, logits, targets


# ---------------------------------------------------------------------------
# Training helpers (shared with train_v2)
# ---------------------------------------------------------------------------

def _make_sampler(labels: np.ndarray, max_weight_mult: float = 10.0) -> WeightedRandomSampler:
    """WeightedRandomSampler with a cap on per-class upsampling multiplier.

    Without a cap, tiny classes (e.g. vasc=108 vs nv=4691) get upsampled ~43x,
    which causes the model to massively over-predict them.  We cap at 10x the
    weight of the majority class so rare classes are still boosted, but not
    pathologically so.
    """
    counts = Counter(labels.tolist())
    max_count = max(counts.values())
    # base weight = 1/count; cap = 1/(max_count / max_weight_mult)
    cap = max_weight_mult / max_count
    w_per_class = {cls: min(1.0 / cnt, cap) for cls, cnt in counts.items()}
    sample_weights = np.array([w_per_class[int(y)] for y in labels], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def _class_weights(
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    counts = Counter(labels.tolist())
    freqs  = np.array([counts.get(i, 1) for i in range(num_classes)], dtype=np.float32)
    w      = 1.0 / freqs
    w      = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def _save_training_curves(epoch_history: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs     = [e["epoch"]        for e in epoch_history]
    train_loss = [e["train_loss"]   for e in epoch_history]
    val_f1     = [e["val_macro_f1"] for e in epoch_history]
    best_ep    = epoch_history[int(np.argmax(val_f1))]["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, "b-o", markersize=4, label="Train loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1, "g-o", markersize=4, label="Val macro-F1")
    ax2.axvline(best_ep, color="red", linestyle="--", alpha=0.6, label=f"Best epoch {best_ep}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro-F1")
    ax2.set_title("Validation Macro-F1"); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle("Training curves — ViT-Base-16 / HAM10000 7-class", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_vit(version_tag: str = "vit_v1") -> Path:
    """Fine-tune google/vit-base-patch16-224 on our HAM10000 7-class split.

    Returns the path to the best checkpoint saved.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}", flush=True)
    print(f"Base model: {config.VIT_BASE_MODEL}", flush=True)
    print(f"7-class labels: {config.HAM10000_DX_LABELS}", flush=True)

    # --- Processor (handles resize + normalise) ---
    print("Loading image processor…", flush=True)
    processor = AutoImageProcessor.from_pretrained(config.VIT_BASE_MODEL)

    # --- Datasets + loaders ---
    train_ds = ViTDataset(config.PROCESSED_DIR / "train.parquet", processor, augment=True)
    val_ds   = ViTDataset(config.PROCESSED_DIR / "val.parquet",   processor, augment=False)

    sampler = _make_sampler(train_ds.df["dx_label"].to_numpy())
    train_loader = DataLoader(
        train_ds,
        batch_size=config.VIT_TRAIN_BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.VIT_EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # --- Class weights ---
    cw = _class_weights(train_ds.df["dx_label"].to_numpy(), config.NUM_FINEGRAINED_CLASSES, device)
    print(f"Class weights: { {config.HAM10000_DX_LABELS[i]: round(float(cw[i]), 2) for i in range(len(cw))} }", flush=True)

    # --- Model ---
    print("Building ViT model…", flush=True)
    model = build_vit(device)

    # --- Stage 1: freeze backbone, warm-up head ---
    _freeze_vit_backbone(model)
    print(f"Transformer backbone frozen for first {config.VIT_UNFREEZE_EPOCH} epochs.", flush=True)
    optimizer = torch.optim.AdamW(
        _get_vit_param_groups(model, config.VIT_LEARNING_RATE),
        weight_decay=config.VIT_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.VIT_NUM_EPOCHS,
    )
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=config.VIT_LABEL_SMOOTHING)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # --- Paths ---
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    reports_dir = config.MODELS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    best_path       = config.CHECKPOINTS_DIR / f"{version_tag}.pt"
    val_logits_path = config.CALIBRATION_DIR / f"val_logits_{version_tag}.npz"

    best_f1                 = -1.0
    epochs_without_improve  = 0
    epoch_history: list[dict] = []
    backbone_unfrozen       = False

    for epoch in range(1, config.VIT_NUM_EPOCHS + 1):

        # --- Stage 2: unfreeze with differential LR ---
        if epoch == config.VIT_UNFREEZE_EPOCH + 1 and not backbone_unfrozen:
            _unfreeze_vit_backbone(model)
            optimizer = torch.optim.AdamW(
                _get_vit_param_groups(model, config.VIT_LEARNING_RATE),
                weight_decay=config.VIT_WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.VIT_NUM_EPOCHS - config.VIT_UNFREEZE_EPOCH,
            )
            backbone_unfrozen = True
            print(
                f"Backbone unfrozen at epoch {epoch} "
                f"(backbone lr={config.VIT_LEARNING_RATE * config.VIT_BACKBONE_LR_MULT:.1e}, "
                f"head lr={config.VIT_LEARNING_RATE:.1e}).",
                flush=True,
            )

        # --- Train one epoch ---
        model.train()
        t0      = time.time()
        running = 0.0

        for pixel_values, y in train_loader:
            pixel_values = pixel_values.to(device, non_blocking=True)
            y            = y.to(device,            non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(pixel_values=pixel_values).logits
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * pixel_values.size(0)

        scheduler.step()
        train_loss = running / len(train_ds)

        val_f1, val_logits, val_targets = evaluate_vit(model, val_loader, device)
        elapsed = time.time() - t0
        head_lr = optimizer.param_groups[-1]["lr"]

        print(
            f"epoch {epoch:>2}  train_loss={train_loss:.4f}  "
            f"val_macroF1={val_f1:.4f}  head_lr={head_lr:.2e}  ({elapsed:.1f}s)",
            flush=True,
        )

        epoch_history.append({
            "epoch":           epoch,
            "train_loss":      round(float(train_loss), 6),
            "val_macro_f1":    round(float(val_f1), 6),
            "head_lr":         float(head_lr),
            "elapsed_s":       round(float(elapsed), 1),
            "backbone_frozen": not backbone_unfrozen,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_vit_checkpoint(model, best_path, extras={
                "version_tag":  version_tag,
                "val_macro_f1": float(val_f1),
                "epoch":        epoch,
            })
            np.savez(val_logits_path, logits=val_logits, targets=val_targets)
            epochs_without_improve = 0
            print("  ** new best -- checkpoint saved.", flush=True)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= config.VIT_EARLY_STOP_PATIENCE:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {epochs_without_improve} epochs).",
                    flush=True,
                )
                break

    # --- Save training curves + summary ---
    _save_training_curves(epoch_history, reports_dir / f"training_curves_{version_tag}.png")

    best_idx = int(np.argmax([e["val_macro_f1"] for e in epoch_history]))
    summary  = {
        "version_tag":       version_tag,
        "base_model":        config.VIT_BASE_MODEL,
        "num_classes":       config.NUM_FINEGRAINED_CLASSES,
        "dx_labels":         config.HAM10000_DX_LABELS,
        "best_val_macro_f1": best_f1,
        "best_epoch":        epoch_history[best_idx]["epoch"],
        "total_epochs_run":  len(epoch_history),
        "checkpoint":        str(best_path),
        "val_logits":        str(val_logits_path),
        "epoch_history":     epoch_history,
        "training_config": {
            "base_model":           config.VIT_BASE_MODEL,
            "batch_size":           config.VIT_TRAIN_BATCH_SIZE,
            "learning_rate":        config.VIT_LEARNING_RATE,
            "backbone_lr_mult":     config.VIT_BACKBONE_LR_MULT,
            "unfreeze_epoch":       config.VIT_UNFREEZE_EPOCH,
            "label_smoothing":      config.VIT_LABEL_SMOOTHING,
            "weight_decay":         config.VIT_WEIGHT_DECAY,
            "early_stop_patience":  config.VIT_EARLY_STOP_PATIENCE,
            "num_epochs_max":       config.VIT_NUM_EPOCHS,
            "amp":                  True,
            "augmentation":         "standard",
        },
    }
    summary_path = config.CHECKPOINTS_DIR / f"summary_{version_tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nBest val macro-F1: {best_f1:.4f}  (epoch {summary['best_epoch']})", flush=True)
    print(f"Checkpoint:        {best_path}", flush=True)
    print(f"Val logits cached: {val_logits_path}", flush=True)
    print(f"Summary:           {summary_path}", flush=True)
    return best_path
