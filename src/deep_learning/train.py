"""
Train EfficientNet-B4 on the HAM10000 7-class fine-grained task.

Training strategy (all thresholds live in config.py):
  - Staged unfreezing: backbone frozen for first UNFREEZE_EPOCH epochs so the
    randomly-initialised head stabilises before we touch pretrained weights.
  - Differential LR: backbone at lr * BACKBONE_LR_MULT, head at full lr.
  - Explicit class weights + WeightedRandomSampler for HAM10000 imbalance
    (nv ~67%, df ~1%).
  - Label smoothing (LABEL_SMOOTHING) to reduce overconfidence on 7-class task.
  - AMP (automatic mixed precision) for memory efficiency on 6 GB GPU.
  - Saves best checkpoint by macro-F1 + raw val logits for calibration.
  - Full epoch history written to summary JSON for report generation.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from PIL import Image

import config
from src.deep_learning.model import (
    FocalLoss, build_model, freeze_backbone, get_param_groups,
    load_checkpoint, save_checkpoint, unfreeze_backbone,
)
from src.preprocessing.augmentation import apply, build_eval_transform, build_strong_train_transform, build_train_transform


class HAM10000Dataset(Dataset):
    """Reads a parquet manifest produced by scripts/prepare_ham10000.py.

    label_col controls which label column is used:
      'dx_label'     — 7-class integer (0-6)
      'binary_label' — binary integer (0-1)
    """

    def __init__(self, manifest_path: Path, transform, label_col: str = "dx_label"):
        self.df = pd.read_parquet(manifest_path)
        self.transform = transform
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["image_path"]).convert("RGB"))
        tensor = apply(self.transform, img)
        return tensor, int(row[self.label_col])


def _make_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = Counter(labels.tolist())
    weights_per_class = {cls: 1.0 / cnt for cls, cnt in counts.items()}
    sample_weights = np.array([weights_per_class[int(y)] for y in labels], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def _class_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean = 1."""
    counts = Counter(labels.tolist())
    freqs = np.array([counts.get(i, 1) for i in range(num_classes)], dtype=np.float32)
    weights = 1.0 / freqs
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _save_training_curves(epoch_history: list[dict], out_path: Path) -> None:
    epochs      = [e["epoch"]        for e in epoch_history]
    train_loss  = [e["train_loss"]   for e in epoch_history]
    val_f1      = [e["val_macro_f1"] for e in epoch_history]
    best_epoch  = epoch_history[int(np.argmax(val_f1))]["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, "b-o", markersize=4, label="Train loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1, "g-o", markersize=4, label="Val macro-F1")
    ax2.axvline(best_epoch, color="red", linestyle="--", alpha=0.6, label=f"Best epoch {best_epoch}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro-F1")
    ax2.set_title("Validation Macro-F1"); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle("Training curves — EfficientNet-B4 / HAM10000 7-class", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved: {out_path}", flush=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(x).float()  # cast to fp32 inside AMP context
        out = torch.nan_to_num(out, nan=0.0, posinf=30.0, neginf=-30.0)
        all_logits.append(out.cpu().numpy())
        all_targets.append(y.numpy())
    logits  = np.concatenate(all_logits,  axis=0)
    targets = np.concatenate(all_targets, axis=0)
    preds   = logits.argmax(axis=1)
    f1      = f1_score(targets, preds, average="macro")
    return f1, logits, targets


def train(version_tag: str = "v2") -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}", flush=True)
    print(f"7-class labels: {config.HAM10000_DX_LABELS}", flush=True)

    train_ds = HAM10000Dataset(config.PROCESSED_DIR / "train.parquet", build_train_transform(), label_col="dx_label")
    val_ds   = HAM10000Dataset(config.PROCESSED_DIR / "val.parquet",   build_eval_transform(),  label_col="dx_label")

    sampler = _make_sampler(train_ds.df["dx_label"].to_numpy())
    train_loader = DataLoader(
        train_ds, batch_size=config.TRAIN_BATCH_SIZE, sampler=sampler,
        num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    # Class weights from training split
    cw = _class_weights(train_ds.df["dx_label"].to_numpy(), config.NUM_FINEGRAINED_CLASSES, device)
    print(f"Class weights: { {config.HAM10000_DX_LABELS[i]: round(float(cw[i]),2) for i in range(len(cw))} }", flush=True)

    model = build_model(num_classes=config.NUM_FINEGRAINED_CLASSES, pretrained=True).to(device)

    # --- Phase 1: freeze backbone, train head only ---
    freeze_backbone(model)
    print(f"Backbone frozen for first {config.UNFREEZE_EPOCH} epochs.", flush=True)
    optimizer = torch.optim.AdamW(
        get_param_groups(model, config.LEARNING_RATE),
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=config.LABEL_SMOOTHING)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_f1     = -1.0
    best_path   = config.CHECKPOINTS_DIR / f"efficientnet_b4_{version_tag}.pt"
    val_logits_path = config.CALIBRATION_DIR / f"val_logits_{version_tag}.npz"
    epochs_without_improve = 0
    epoch_history: list[dict] = []
    backbone_unfrozen = False

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    reports_dir = config.MODELS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.NUM_EPOCHS + 1):

        # --- Phase 2: unfreeze backbone with differential LR ---
        if epoch == config.UNFREEZE_EPOCH + 1 and not backbone_unfrozen:
            unfreeze_backbone(model)
            # Rebuild optimizer with backbone + head param groups
            optimizer = torch.optim.AdamW(
                get_param_groups(model, config.LEARNING_RATE),
                weight_decay=config.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.NUM_EPOCHS - config.UNFREEZE_EPOCH,
            )
            backbone_unfrozen = True
            print(f"Backbone unfrozen at epoch {epoch} (backbone lr={config.LEARNING_RATE * config.BACKBONE_LR_MULT:.1e}, head lr={config.LEARNING_RATE:.1e}).", flush=True)

        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()
        train_loss = running / len(train_ds)

        val_f1, val_logits, val_targets = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        head_lr  = optimizer.param_groups[-1]["lr"]
        print(f"epoch {epoch:>2}  train_loss={train_loss:.4f}  val_macroF1={val_f1:.4f}  head_lr={head_lr:.2e}  ({elapsed:.1f}s)", flush=True)

        epoch_history.append({
            "epoch":        epoch,
            "train_loss":   round(float(train_loss), 6),
            "val_macro_f1": round(float(val_f1), 6),
            "head_lr":      float(head_lr),
            "elapsed_s":    round(float(elapsed), 1),
            "backbone_frozen": not backbone_unfrozen,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, best_path, extras={
                "version_tag":    version_tag,
                "val_macro_f1":   float(val_f1),
                "num_classes":    config.NUM_FINEGRAINED_CLASSES,
                "dx_labels":      config.HAM10000_DX_LABELS,
            })
            np.savez(val_logits_path, logits=val_logits, targets=val_targets)
            epochs_without_improve = 0
            print(f"  ** new best -- checkpoint saved.", flush=True)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= config.EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {epochs_without_improve} epochs).", flush=True)
                break

    # --- Save training curves ---
    _save_training_curves(epoch_history, reports_dir / f"training_curves_{version_tag}.png")

    summary = {
        "version_tag":        version_tag,
        "num_classes":        config.NUM_FINEGRAINED_CLASSES,
        "dx_labels":          config.HAM10000_DX_LABELS,
        "best_val_macro_f1":  best_f1,
        "best_epoch":         epoch_history[int(np.argmax([e["val_macro_f1"] for e in epoch_history]))]["epoch"],
        "total_epochs_run":   len(epoch_history),
        "checkpoint":         str(best_path),
        "val_logits":         str(val_logits_path),
        "epoch_history":      epoch_history,
        "training_config": {
            "batch_size":        config.TRAIN_BATCH_SIZE,
            "learning_rate":     config.LEARNING_RATE,
            "backbone_lr_mult":  config.BACKBONE_LR_MULT,
            "unfreeze_epoch":    config.UNFREEZE_EPOCH,
            "label_smoothing":   config.LABEL_SMOOTHING,
            "weight_decay":      config.WEIGHT_DECAY,
            "early_stop_patience": config.EARLY_STOP_PATIENCE,
            "num_epochs_max":    config.NUM_EPOCHS,
            "amp":               True,
        },
    }
    summary_path = config.CHECKPOINTS_DIR / f"summary_{version_tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nBest val macro-F1: {best_f1:.4f}  (epoch {summary['best_epoch']})")
    print(f"Checkpoint:        {best_path}")
    print(f"Val logits cached: {val_logits_path}")
    print(f"Summary:           {summary_path}")
    return best_path


def train_v3(warm_start_tag: str = "v2", version_tag: str = "v3") -> Path:
    """Fine-tune from a v2 checkpoint using focal loss + stronger augmentation.

    Key differences from train():
      - Warm-starts from an existing checkpoint (warm_start_tag)
      - FocalLoss(gamma=2) replaces CrossEntropy
      - build_strong_train_transform() — heavier augmentation
      - No backbone freezing phase (backbone already trained)
      - Lower learning rate (V3_LEARNING_RATE from config)
      - Squared class weights for more aggressive rare-class focus
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training v3 on device: {device}", flush=True)
    print(f"Warm-starting from checkpoint: {warm_start_tag}", flush=True)

    train_ds = HAM10000Dataset(config.PROCESSED_DIR / "train.parquet", build_strong_train_transform(), label_col="dx_label")
    val_ds   = HAM10000Dataset(config.PROCESSED_DIR / "val.parquet",   build_eval_transform(),         label_col="dx_label")

    sampler = _make_sampler(train_ds.df["dx_label"].to_numpy())
    train_loader = DataLoader(
        train_ds, batch_size=config.TRAIN_BATCH_SIZE, sampler=sampler,
        num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"),
    )

    # Squared inverse-frequency weights — more aggressive rare-class focus
    raw_cw = _class_weights(train_ds.df["dx_label"].to_numpy(), config.NUM_FINEGRAINED_CLASSES, device)
    cw = (raw_cw ** 1.5)
    cw = cw / cw.mean()
    print(f"Class weights (^1.5): { {config.HAM10000_DX_LABELS[i]: round(float(cw[i]),2) for i in range(len(cw))} }", flush=True)

    # Warm-start model
    warm_path = config.CHECKPOINTS_DIR / f"efficientnet_b4_{warm_start_tag}.pt"
    if not warm_path.is_file():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {warm_path}. Train v2 first.")
    model, _ = load_checkpoint(warm_path, device=device)
    model.train()
    unfreeze_backbone(model)
    print(f"Loaded warm-start weights from {warm_path}", flush=True)

    optimizer = torch.optim.AdamW(
        get_param_groups(model, config.V3_LEARNING_RATE),
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.V3_NUM_EPOCHS)
    criterion = FocalLoss(gamma=config.V3_FOCAL_GAMMA, weight=cw, label_smoothing=config.LABEL_SMOOTHING)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_f1     = -1.0
    best_path   = config.CHECKPOINTS_DIR / f"efficientnet_b4_{version_tag}.pt"
    val_logits_path = config.CALIBRATION_DIR / f"val_logits_{version_tag}.npz"
    epochs_without_improve = 0
    epoch_history: list[dict] = []

    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    reports_dir = config.MODELS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.V3_NUM_EPOCHS + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()
        train_loss = running / len(train_ds)

        val_f1, val_logits, val_targets = evaluate(model, val_loader, device)
        elapsed  = time.time() - t0
        head_lr  = optimizer.param_groups[-1]["lr"]
        print(f"epoch {epoch:>2}  train_loss={train_loss:.4f}  val_macroF1={val_f1:.4f}  head_lr={head_lr:.2e}  ({elapsed:.1f}s)", flush=True)

        epoch_history.append({
            "epoch":        epoch,
            "train_loss":   round(float(train_loss), 6),
            "val_macro_f1": round(float(val_f1), 6),
            "head_lr":      float(head_lr),
            "elapsed_s":    round(float(elapsed), 1),
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, best_path, extras={
                "version_tag": version_tag,
                "warm_start":  warm_start_tag,
                "val_macro_f1": float(val_f1),
                "num_classes": config.NUM_FINEGRAINED_CLASSES,
                "dx_labels":   config.HAM10000_DX_LABELS,
            })
            np.savez(val_logits_path, logits=val_logits, targets=val_targets)
            epochs_without_improve = 0
            print(f"  ** new best -- checkpoint saved.", flush=True)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= config.V3_EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {epochs_without_improve} epochs).", flush=True)
                break

    _save_training_curves(epoch_history, reports_dir / f"training_curves_{version_tag}.png")

    summary = {
        "version_tag":       version_tag,
        "warm_start_from":   warm_start_tag,
        "num_classes":       config.NUM_FINEGRAINED_CLASSES,
        "dx_labels":         config.HAM10000_DX_LABELS,
        "best_val_macro_f1": best_f1,
        "best_epoch":        epoch_history[int(np.argmax([e["val_macro_f1"] for e in epoch_history]))]["epoch"],
        "total_epochs_run":  len(epoch_history),
        "checkpoint":        str(best_path),
        "val_logits":        str(val_logits_path),
        "epoch_history":     epoch_history,
        "training_config": {
            "warm_start":       warm_start_tag,
            "batch_size":       config.TRAIN_BATCH_SIZE,
            "learning_rate":    config.V3_LEARNING_RATE,
            "backbone_lr_mult": config.V3_BACKBONE_LR_MULT,
            "focal_gamma":      config.V3_FOCAL_GAMMA,
            "label_smoothing":  config.LABEL_SMOOTHING,
            "augmentation":     "strong",
            "weight_decay":     config.WEIGHT_DECAY,
            "early_stop_patience": config.V3_EARLY_STOP_PATIENCE,
            "amp":              True,
        },
    }
    summary_path = config.CHECKPOINTS_DIR / f"summary_{version_tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nBest val macro-F1: {best_f1:.4f}  (epoch {summary['best_epoch']})")
    print(f"Checkpoint:        {best_path}")
    print(f"Val logits:        {val_logits_path}")
    print(f"Summary:           {summary_path}")
    return best_path
