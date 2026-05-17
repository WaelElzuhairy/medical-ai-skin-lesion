"""
evaluate.py — Comprehensive evaluation of the trained VGG16 model on HAM10000.

Usage:
    python evaluate.py

Outputs written to outputs/:
    confusion_matrix.png  — seaborn heatmap
    roc_curves.png        — per-class ROC curves (one-vs-rest)
    metrics.json          — accuracy, roc_auc, per-class precision/recall/F1
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DEVICE,
    MODEL_DIR,
    NUM_CLASSES,
    OUTPUT_DIR,
)
from dataset import load_and_split


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path):
    """Reconstruct VGG16 head and load fine-tuned weights."""
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, loader):
    """
    Run model over `loader`; return:
        preds  — (N,) int array of argmax predictions
        labels — (N,) int array of ground-truth labels
        probs  — (N, C) float array of softmax probabilities
    """
    all_preds, all_labels, all_probs = [], [], []
    bar = tqdm(loader, desc="Inference", unit="batch")
    with torch.no_grad():
        for images, labels in bar:
            outputs = model(images.to(DEVICE))
            probs   = torch.softmax(outputs, dim=1).cpu().numpy()
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.concatenate(all_probs, axis=0),
    )


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _plot_confusion_matrix(cm: np.ndarray, save_path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title("Confusion Matrix — HAM10000", fontsize=13, pad=12)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {save_path}")


def _plot_roc_curves(all_labels: np.ndarray, all_probs: np.ndarray,
                     save_path) -> None:
    y_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        auc_val     = roc_auc_score(y_bin[:, i], all_probs[:, i])
        ax.plot(fpr, tpr, label=f"{cls}  (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curves — HAM10000 (One-vs-Rest)", fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ROC curves saved       -> {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    checkpoint = MODEL_DIR / "phase2_best.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Run  python train.py  first."
        )

    print(f"Loading model from {checkpoint} …")
    model = load_model(checkpoint)

    print("Loading test split …")
    _, _, test_ds, _, _ = load_and_split()
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    all_preds, all_labels, all_probs = run_inference(model, test_loader)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    accuracy = float((all_preds == all_labels).mean())
    print(f"\nOverall accuracy : {accuracy:.4f}")

    # ── Per-class accuracy ────────────────────────────────────────────────────
    print("\nPer-class accuracy:")
    per_class_acc = {}
    for i, cls in enumerate(CLASS_NAMES):
        mask    = all_labels == i
        cls_acc = float((all_preds[mask] == all_labels[mask]).mean()) if mask.any() else 0.0
        per_class_acc[cls] = round(cls_acc, 4)
        print(f"  {cls:<8}: {cls_acc:.4f}  (n={mask.sum()})")

    # ── Classification report ─────────────────────────────────────────────────
    print("\nClassification Report:")
    report_str  = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    report_dict = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True
    )
    print(report_str)

    # ── Macro ROC-AUC (one-vs-rest) ───────────────────────────────────────────
    y_bin   = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
    roc_auc = float(roc_auc_score(y_bin, all_probs, average="macro", multi_class="ovr"))
    print(f"Macro ROC-AUC (OvR) : {roc_auc:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print()
    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm,       OUTPUT_DIR / "confusion_matrix.png")
    _plot_roc_curves(all_labels, all_probs, OUTPUT_DIR / "roc_curves.png")

    # ── metrics.json ──────────────────────────────────────────────────────────
    metrics = {
        "accuracy": round(accuracy, 4),
        "roc_auc":  round(roc_auc,  4),
        "per_class": {
            cls: {
                "precision": round(report_dict[cls]["precision"], 4),
                "recall":    round(report_dict[cls]["recall"],    4),
                "f1":        round(report_dict[cls]["f1-score"],  4),
                "accuracy":  per_class_acc[cls],
            }
            for cls in CLASS_NAMES
        },
    }
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics JSON saved     -> {metrics_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
