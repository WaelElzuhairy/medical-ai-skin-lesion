"""
train.py — Two-phase VGG16 fine-tuning on HAM10000.

Usage:
    python train.py

Phase 1 — Feature Extraction  (EPOCHS_PHASE1 epochs)
    Backbone frozen; only the new 7-class head is trained.

Phase 2 — Fine-tuning          (up to EPOCHS_PHASE2 epochs)
    All layers unfrozen; differential learning rates + early stopping.
"""

import copy
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS_PHASE1,
    EPOCHS_PHASE2,
    LR_CLASSIFIER_FINETUNE,
    LR_FINETUNE,
    LR_HEAD,
    MODEL_DIR,
    NUM_CLASSES,
    OUTPUT_DIR,
    SEED,
)
from dataset import load_and_split


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── CSV logging helper ────────────────────────────────────────────────────────

def _append_csv(filepath, row, header=None) -> None:
    """Append one row to a CSV; write header on the first call (fresh file)."""
    write_header = not filepath.exists()
    with open(filepath, "a", newline="") as f:
        w = csv.writer(f)
        if write_header and header:
            w.writerow(header)
        w.writerow(row)


# ── Per-epoch helpers ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, desc: str) -> float:
    """Run one training epoch; return mean loss over the dataset."""
    model.train()
    running_loss = 0.0
    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for images, labels in bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader.dataset)


def evaluate(model, loader, device, criterion, desc: str):
    """Run validation/test pass; return (mean_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct      = 0
    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    with torch.no_grad():
        for images, labels in bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            running_loss += criterion(outputs, labels).item() * images.size(0)
            correct      += (outputs.argmax(dim=1) == labels).sum().item()
    n = len(loader.dataset)
    return running_loss / n, correct / n


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def run_phase1(train_loader, val_loader, class_weights) -> str:
    """Feature-extraction phase: freeze backbone, train head only."""
    print(f"\n{'='*55}")
    print(f"  PHASE 1 — Feature Extraction  ({EPOCHS_PHASE1} epochs)")
    print(f"{'='*55}")

    # Load pretrained VGG16 and freeze the convolutional backbone
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the final fully-connected layer with a 7-class head
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = Adam(model.classifier.parameters(), lr=LR_HEAD)
    # Halve the LR every 5 epochs to allow stable convergence of the head
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    log_path    = OUTPUT_DIR / "phase1_log.csv"
    log_header  = ["epoch", "train_loss", "val_loss", "val_acc"]
    # Remove stale log from a previous run
    if log_path.exists():
        log_path.unlink()

    best_val_acc   = 0.0
    checkpoint_path = MODEL_DIR / "phase1_best.pth"

    for epoch in range(1, EPOCHS_PHASE1 + 1):
        train_loss          = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            desc=f"P1 [{epoch:02d}/{EPOCHS_PHASE1}] train",
        )
        val_loss, val_acc   = evaluate(
            model, val_loader, DEVICE, criterion,
            desc=f"P1 [{epoch:02d}/{EPOCHS_PHASE1}] val  ",
        )
        scheduler.step()

        print(
            f"  Epoch {epoch:02d}/{EPOCHS_PHASE1} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}"
        )
        _append_csv(log_path, [epoch, train_loss, val_loss, val_acc], log_header)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    >> checkpoint saved  (val_acc={best_val_acc:.4f})")

    print(f"\n  Phase 1 complete — best val accuracy: {best_val_acc:.4f}")
    print(f"  Checkpoint : {checkpoint_path}")
    return str(checkpoint_path)


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def run_phase2(train_loader, val_loader, class_weights, phase1_ckpt: str) -> str:
    """Full fine-tuning: unfreeze all layers, differential LRs, early stopping."""
    print(f"\n{'='*55}")
    print(f"  PHASE 2 — Fine-tuning  (up to {EPOCHS_PHASE2} epochs)")
    print(f"{'='*55}")

    # Reconstruct architecture and load Phase 1 weights
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    model.load_state_dict(torch.load(phase1_ckpt, map_location=DEVICE))

    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # Differential learning rates: backbone gets a much smaller LR than the head
    optimizer = Adam([
        {"params": model.features.parameters(),    "lr": LR_FINETUNE},
        {"params": model.classifier.parameters(),  "lr": LR_CLASSIFIER_FINETUNE},
    ])
    # Cosine annealing decays LR smoothly to near-zero over all phase-2 epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE2)

    log_path    = OUTPUT_DIR / "phase2_log.csv"
    log_header  = ["epoch", "train_loss", "val_loss", "val_acc"]
    if log_path.exists():
        log_path.unlink()

    checkpoint_path  = MODEL_DIR / "phase2_best.pth"
    best_val_loss    = float("inf")
    best_weights     = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, EPOCHS_PHASE2 + 1):
        train_loss         = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            desc=f"P2 [{epoch:02d}/{EPOCHS_PHASE2}] train",
        )
        val_loss, val_acc  = evaluate(
            model, val_loader, DEVICE, criterion,
            desc=f"P2 [{epoch:02d}/{EPOCHS_PHASE2}] val  ",
        )
        scheduler.step()

        print(
            f"  Epoch {epoch:02d}/{EPOCHS_PHASE2} | "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}"
        )
        _append_csv(log_path, [epoch, train_loss, val_loss, val_acc], log_header)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_weights     = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(best_weights, checkpoint_path)
            print(f"    >> checkpoint saved  (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(
                f"    – no improvement  "
                f"(patience {patience_counter}/{EARLY_STOPPING_PATIENCE})"
            )
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  Early stopping triggered at epoch {epoch}.")
                break

    # Restore the best weights found during phase 2
    model.load_state_dict(best_weights)
    torch.save(best_weights, checkpoint_path)

    print(f"\n  Phase 2 complete — best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint : {checkpoint_path}")
    return str(checkpoint_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)
    print(f"Device : {DEVICE}")

    # Load datasets and imbalance artefacts
    print("Loading dataset …")
    train_ds, val_ds, _, sampler, class_weights = load_and_split()

    # When oversampling is active the sampler controls iteration order;
    # shuffle must be False to avoid conflicting with the sampler.
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,          # num_workers=0 is required on Windows
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    phase1_ckpt = run_phase1(train_loader, val_loader, class_weights)
    run_phase2(train_loader, val_loader, class_weights, phase1_ckpt)

    print("\n" + "=" * 55)
    print("  Training complete.")
    print(f"  Phase 1 best  -> {MODEL_DIR / 'phase1_best.pth'}")
    print(f"  Phase 2 best  -> {MODEL_DIR / 'phase2_best.pth'}")
    print(f"  Training logs -> {OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
