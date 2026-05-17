"""
binary_eval.py  — Re-evaluate the model as a binary classifier:
    Malignant : mel, bcc, akiec
    Benign    : nv, bkl, df, vasc
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from torch.utils.data import DataLoader

from config import BATCH_SIZE, CLASS_NAMES, DEVICE, MODEL_DIR, NUM_CLASSES, OUTPUT_DIR
from dataset import load_and_split

# ── Binary mapping ─────────────────────────────────────────────────────────────
# 0 = Benign   1 = Malignant
MALIGNANT = {"mel", "bcc", "akiec"}   # clinically malignant / pre-malignant
BINARY_LABEL = {i: (1 if cls in MALIGNANT else 0) for i, cls in enumerate(CLASS_NAMES)}
BINARY_NAMES = ["Benign", "Malignant"]

def to_binary(arr):
    return np.array([BINARY_LABEL[x] for x in arr])

# ── Model ──────────────────────────────────────────────────────────────────────
def load_model():
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_DIR / "phase2_best.pth", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(model, loader):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            out   = model(images.to(DEVICE))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    return np.array(all_preds), np.array(all_labels), np.concatenate(all_probs)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading model ...")
    model = load_model()

    print("Loading test split ...")
    _, _, test_ds, _, _ = load_and_split()
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Running inference ...")
    preds7, labels7, probs7 = run_inference(model, loader)

    # ── Map to binary ──────────────────────────────────────────────────────────
    bin_preds  = to_binary(preds7)
    bin_labels = to_binary(labels7)

    # Malignant probability = sum of mel + bcc + akiec softmax scores
    malignant_idx = [i for i, c in enumerate(CLASS_NAMES) if c in MALIGNANT]
    malignant_prob = probs7[:, malignant_idx].sum(axis=1)   # shape (N,)

    # ── Metrics ────────────────────────────────────────────────────────────────
    acc    = accuracy_score(bin_labels, bin_preds)
    auc    = roc_auc_score(bin_labels, malignant_prob)
    cm     = confusion_matrix(bin_labels, bin_preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)   # recall for Malignant
    specificity = tn / (tn + fp)   # recall for Benign

    print("\n" + "="*60)
    print("  BINARY CLASSIFICATION RESULTS")
    print("  (Malignant: mel, bcc, akiec  |  Benign: nv, bkl, df, vasc)")
    print("="*60)
    print(f"  Overall Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC           : {auc:.4f}")
    print(f"  Sensitivity       : {sensitivity:.4f}  (Malignant recall)")
    print(f"  Specificity       : {specificity:.4f}  (Benign recall)")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print()
    print(classification_report(bin_labels, bin_preds, target_names=BINARY_NAMES))

    # ── Confusion matrix plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=BINARY_NAMES, yticklabels=BINARY_NAMES, ax=ax,
                annot_kws={"size": 16})
    ax.set_title("Binary Confusion Matrix\n(Malignant vs Benign)", fontsize=13, pad=12)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "binary_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix -> {cm_path}")

    # ── ROC curve plot ─────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(bin_labels, malignant_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="crimson", lw=2, label=f"Malignant (AUC = {auc:.3f})")
    ax.plot([0,1],[0,1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("Binary ROC Curve — Malignant vs Benign", fontsize=13, pad=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    roc_path = OUTPUT_DIR / "binary_roc_curve.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"  ROC curve        -> {roc_path}")

if __name__ == "__main__":
    main()
