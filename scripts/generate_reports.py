"""
Generate and save full training diagnostics for a model version.

Outputs (all in models/reports/):
  - confusion_matrix_7class_{tag}.png   — 7×7 confusion matrix
  - confusion_matrix_binary_{tag}.png   — 2×2 binary collapse
  - reliability_diagram_{tag}.png       — calibration curves
  - training_curves_{tag}.png           — loss + val F1 per epoch (if summary exists)
  - metrics_{tag}.json                  — full metrics for report generation

Run:
    python scripts/generate_reports.py [version_tag]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.deep_learning.calibration import expected_calibration_error, load_temperature
from src.deep_learning.model import load_checkpoint
from src.deep_learning.train import HAM10000Dataset
from src.preprocessing.augmentation import build_eval_transform

OUT_DIR = config.MODELS_DIR / "reports"


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _binary_collapse(probs: np.ndarray) -> np.ndarray:
    mal = probs[:, config.MALIGNANT_CLASS_INDICES].sum(axis=1, keepdims=True)
    return np.concatenate([1.0 - mal, mal], axis=1)


def plot_confusion_matrix(targets, preds, labels, title, out_path):
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_reliability_diagram(logits, targets_binary, T, out_path):
    n_bins = 15
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    raw_probs2 = _binary_collapse(_softmax(logits))
    cal_probs2 = _binary_collapse(_softmax(logits / T))

    for ax, (label, p2) in zip(axes, [
        ("Raw softmax", raw_probs2),
        (f"Calibrated (T={T:.3f})", cal_probs2),
    ]):
        pos_probs = p2[:, 1]
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (pos_probs >= lo) & (pos_probs < hi)
            if mask.sum() == 0:
                continue
            bin_accs.append((targets_binary[mask] == 1).mean())
            bin_confs.append(pos_probs[mask].mean())

        ece = expected_calibration_error(p2, targets_binary)
        ax.bar(bin_confs, bin_accs, width=0.05, alpha=0.7)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_title(f"{label}\nECE={ece:.4f}")
        ax.legend()

    fig.suptitle("Reliability Diagram — Binary (malignant class)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_training_curves(summary: dict, out_path: Path) -> None:
    history = summary.get("epoch_history", [])
    if not history:
        return
    epochs     = [e["epoch"]        for e in history]
    train_loss = [e["train_loss"]   for e in history]
    val_f1     = [e["val_macro_f1"] for e in history]
    best_epoch = summary.get("best_epoch", epochs[int(np.argmax(val_f1))])
    unfreeze   = summary.get("training_config", {}).get("unfreeze_epoch", None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(epochs, train_loss, "b-o", markersize=4)
    if unfreeze:
        ax1.axvline(unfreeze + 1, color="orange", linestyle="--", alpha=0.7, label=f"Unfreeze epoch {unfreeze+1}")
        ax1.legend()
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss"); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1, "g-o", markersize=4)
    ax2.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best epoch {best_epoch}")
    if unfreeze:
        ax2.axvline(unfreeze + 1, color="orange", linestyle="--", alpha=0.7, label=f"Unfreeze epoch {unfreeze+1}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro-F1")
    ax2.set_title("Validation Macro-F1"); ax2.legend(); ax2.grid(True, alpha=0.3)

    arch = summary.get("architecture", "EfficientNet-B4")
    fig.suptitle(f"Training Curves — {arch} HAM10000 7-class ({summary.get('version_tag','')})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _collect_vit_logits(version_tag: str):
    """Load pre-collected test logits from npz, or collect them via subprocess.

    Uses collect_test_logits.py in a clean subprocess to avoid the segfault
    caused by importing matplotlib+sklearn alongside transformers+CUDA in one
    process.
    """
    import subprocess
    npz_path = OUT_DIR / f"test_logits_{version_tag}.npz"

    if not npz_path.is_file():
        print(f"Collecting test logits for {version_tag} (subprocess) ...")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "collect_test_logits.py"),
             version_tag],
            capture_output=False,   # let output stream live
        )
        if result.returncode != 0:
            raise RuntimeError(f"collect_test_logits.py failed for {version_tag}")
        if not npz_path.is_file():
            raise RuntimeError(f"Expected {npz_path} after collect_test_logits.py — not found")

    data      = np.load(npz_path, allow_pickle=True)
    logits    = data["logits"]
    targets   = data["targets"].astype(int)
    T         = float(data["T"][0])
    dx_labels = list(data["dx_labels"])
    return logits, targets, T, dx_labels


def main(version_tag: str = "v2") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    is_vit = version_tag.startswith("vit_")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load logits from pre-collected npz (generated by collect_test_logits.py).
    # We use a subprocess for logit collection to avoid a segfault caused by
    # importing PyTorch + sklearn + matplotlib in the same process on CUDA.
    logits, targets, T, dx_labels = _collect_vit_logits(version_tag)

    if is_vit:
        arch_label   = "ViT-Base-16"
        inner_tag    = version_tag[len("vit_"):]
        summary_path = config.CHECKPOINTS_DIR / f"summary_vit_{inner_tag}.json"
    else:
        arch_label   = "EfficientNet-B4"
        summary_path = config.CHECKPOINTS_DIR / f"summary_{version_tag}.json"

    cal_probs7  = _softmax(logits / T)
    raw_probs7  = _softmax(logits)
    preds7      = cal_probs7.argmax(axis=1)

    cal_probs2  = _binary_collapse(cal_probs7)
    raw_probs2  = _binary_collapse(raw_probs7)
    bin_targets = np.array([1 if t in config.MALIGNANT_CLASS_INDICES else 0 for t in targets])
    bin_preds   = cal_probs2.argmax(axis=1)

    # --- Metrics --------------------------------------------------------------
    report7  = classification_report(targets,     preds7,    target_names=dx_labels,           digits=4, output_dict=True)
    report2  = classification_report(bin_targets, bin_preds, target_names=config.BINARY_LABELS, digits=4, output_dict=True)
    ece_raw  = expected_calibration_error(raw_probs2, bin_targets)
    ece_cal  = expected_calibration_error(cal_probs2, bin_targets)

    metrics = {
        "version_tag":   version_tag,
        "test_samples":  int(len(targets)),
        "temperature_T": float(T),
        "dx_labels":     dx_labels,
        "malignant_class_indices": config.MALIGNANT_CLASS_INDICES,
        "seven_class": {
            "accuracy":  float(report7["accuracy"]),
            "macro_f1":  float(f1_score(targets, preds7, average="macro")),
            "per_class": {
                lbl: {k: float(v) for k, v in report7[lbl].items()}
                for lbl in dx_labels
            },
        },
        "binary_collapse": {
            "accuracy":       float(report2["accuracy"]),
            "macro_f1":       float(f1_score(bin_targets, bin_preds, average="macro")),
            "ece_raw":        float(ece_raw),
            "ece_calibrated": float(ece_cal),
            "ece_target":     config.ECE_TARGET,
            "phase1_pass":    bool(ece_cal < config.ECE_TARGET),
            "per_class":      {
                lbl: {k: float(v) for k, v in report2[lbl].items()}
                for lbl in config.BINARY_LABELS
            },
        },
    }

    metrics_path = OUT_DIR / f"metrics_{version_tag}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved: {metrics_path}")

    # --- Print summary --------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  7-CLASS   accuracy={report7['accuracy']:.4f}  macro-F1={metrics['seven_class']['macro_f1']:.4f}")
    print(f"  BINARY    accuracy={report2['accuracy']:.4f}  macro-F1={metrics['binary_collapse']['macro_f1']:.4f}")
    print(f"  ECE raw={ece_raw:.4f}  calibrated={ece_cal:.4f}  ({'PASS' if ece_cal < config.ECE_TARGET else 'FAIL'})")
    print(f"{'='*60}")

    # update plot title to show architecture
    metrics["architecture"] = arch_label

    # --- Plots ----------------------------------------------------------------
    plot_confusion_matrix(targets, preds7, dx_labels,
                          f"7-Class Confusion Matrix — {arch_label} Test Set ({version_tag})",
                          OUT_DIR / f"confusion_matrix_7class_{version_tag}.png")
    plot_confusion_matrix(bin_targets, bin_preds, config.BINARY_LABELS,
                          f"Binary Confusion Matrix — {arch_label} Test Set ({version_tag})",
                          OUT_DIR / f"confusion_matrix_binary_{version_tag}.png")
    plot_reliability_diagram(logits, bin_targets, T,
                             OUT_DIR / f"reliability_diagram_{version_tag}.png")

    # Training curves from summary if available
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        plot_training_curves(summary, OUT_DIR / f"training_curves_{version_tag}.png")

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v2"
    main(version_tag=tag)
