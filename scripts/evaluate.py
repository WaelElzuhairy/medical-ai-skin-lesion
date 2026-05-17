"""
Evaluate a trained + calibrated checkpoint on the held-out test split.

Reports:
  - 7-class: accuracy, macro-F1, per-class precision/recall/F1
  - Binary collapse: benign/malignant accuracy, macro-F1, ECE (raw + calibrated)

Phase 1 acceptance: post-calibration ECE < config.ECE_TARGET (default 0.05).

Supports both EfficientNet (v2, v3) and ViT (vit_v1) version tags.
Version tags starting with "vit_" automatically use the ViT checkpoint and
ViT inference path.

Run:
    python scripts/evaluate.py [version_tag]
    python scripts/evaluate.py v2       # EfficientNet-B4 v2
    python scripts/evaluate.py vit_v1   # locally-trained ViT
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.deep_learning.calibration import expected_calibration_error, load_temperature


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _binary_collapse(probs: np.ndarray) -> np.ndarray:
    """Collapse 7-class probs to (N,2) binary [benign, malignant]."""
    mal = probs[:, config.MALIGNANT_CLASS_INDICES].sum(axis=1, keepdims=True)
    ben = 1.0 - mal
    return np.concatenate([ben, mal], axis=1)


def _run_efficientnet(version_tag: str, device: torch.device):
    from src.deep_learning.model import load_checkpoint
    from src.deep_learning.train import HAM10000Dataset
    from src.preprocessing.augmentation import build_eval_transform

    ckpt_path = config.CHECKPOINTS_DIR / f"efficientnet_b4_{version_tag}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No EfficientNet checkpoint at {ckpt_path}. Train first.")

    model, meta = load_checkpoint(ckpt_path, device=device)
    T           = load_temperature(version_tag=version_tag)
    dx_labels   = meta.get("dx_labels", config.HAM10000_DX_LABELS)

    test_ds = HAM10000Dataset(
        config.PROCESSED_DIR / "test.parquet",
        build_eval_transform(),
        label_col="dx_label",
    )
    loader = DataLoader(test_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False, num_workers=0)

    all_logits, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                all_logits.append(model(x).cpu().float().numpy())
            all_targets.append(y.numpy())

    return (
        np.concatenate(all_logits,  axis=0),
        np.concatenate(all_targets, axis=0),
        T,
        dx_labels,
    )


def _run_vit(version_tag: str, device: torch.device):
    """version_tag is the inner tag, e.g. 'v1' for a full tag of 'vit_v1'."""
    from transformers import AutoImageProcessor
    from src.deep_learning.vit_train import ViTDataset, evaluate_vit, load_vit_checkpoint

    ckpt_path = config.CHECKPOINTS_DIR / f"vit_{version_tag}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No ViT checkpoint at {ckpt_path}. Train first.")

    model, meta = load_vit_checkpoint(ckpt_path, device=device)
    # Calibration JSON is stored under the full tag (e.g. vit_v1.json)
    T           = load_temperature(version_tag=f"vit_{version_tag}")
    dx_labels   = meta.get("dx_labels", config.HAM10000_DX_LABELS)

    processor = AutoImageProcessor.from_pretrained(
        meta.get("model_name", config.VIT_BASE_MODEL)
    )
    test_ds = ViTDataset(config.PROCESSED_DIR / "test.parquet", processor, augment=False)
    loader  = DataLoader(test_ds, batch_size=config.VIT_EVAL_BATCH_SIZE, shuffle=False, num_workers=0)

    _, logits, targets = evaluate_vit(model, loader, device)
    return logits, targets, T, dx_labels


def main(version_tag: str = "v2") -> None:
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_vit    = version_tag.startswith("vit_")
    inner_tag = version_tag[len("vit_"):] if is_vit else version_tag

    if is_vit:
        print(f"Evaluating ViT checkpoint: vit_{inner_tag}  (T from calibration/{version_tag}.json)", flush=True)
        logits, targets, T, dx_labels = _run_vit(inner_tag, device)
    else:
        print(f"Evaluating EfficientNet checkpoint: efficientnet_b4_{version_tag}", flush=True)
        logits, targets, T, dx_labels = _run_efficientnet(version_tag, device)

    # 7-class metrics
    raw_probs7 = _softmax(logits)
    cal_probs7 = _softmax(logits / T)
    preds7     = cal_probs7.argmax(axis=1)

    # Binary collapse
    cal_probs2  = _binary_collapse(cal_probs7)
    raw_probs2  = _binary_collapse(raw_probs7)
    bin_targets = np.array([1 if t in config.MALIGNANT_CLASS_INDICES else 0 for t in targets])
    bin_preds   = cal_probs2.argmax(axis=1)

    ece_raw = expected_calibration_error(raw_probs2, bin_targets)
    ece_cal = expected_calibration_error(cal_probs2, bin_targets)

    arch = "ViT-Base-16" if is_vit else "EfficientNet-B4"
    print(f"\nTest set: {len(targets)} samples | {arch} | temperature T = {T:.4f}\n")
    print("=" * 60)
    print("7-CLASS RESULTS")
    print("=" * 60)
    print(classification_report(targets, preds7, target_names=dx_labels, digits=4))
    print(f"7-class macro-F1: {f1_score(targets, preds7, average='macro'):.4f}")

    print("\n" + "=" * 60)
    print("BINARY COLLAPSE  (malignant = akiec + bcc + mel)")
    print("=" * 60)
    print(classification_report(bin_targets, bin_preds,
                                target_names=config.BINARY_LABELS, digits=4))
    print(f"Binary macro-F1:        {f1_score(bin_targets, bin_preds, average='macro'):.4f}")
    print(f"ECE (raw softmax):      {ece_raw:.4f}")
    print(f"ECE (calibrated):       {ece_cal:.4f}    target < {config.ECE_TARGET}")

    passed = "PASS" if ece_cal < config.ECE_TARGET else "FAIL"
    print(f"\nPhase 1 acceptance: {passed}")


def save_test_logits(version_tag: str) -> Path:
    """Collect test logits and save to models/reports/test_logits_{tag}.npz.

    Called by generate_reports.py so ViT inference runs in a clean process
    (no matplotlib imported alongside transformers — avoids a CUDA segfault).

    Returns the path to the saved .npz file.
    """
    OUT_DIR = config.MODELS_DIR / "reports"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"test_logits_{version_tag}.npz"

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_vit    = version_tag.startswith("vit_")
    inner_tag = version_tag[len("vit_"):] if is_vit else version_tag

    if is_vit:
        logits, targets, T, dx_labels = _run_vit(inner_tag, device)
    else:
        logits, targets, T, dx_labels = _run_efficientnet(version_tag, device)

    np.savez(out_path, logits=logits, targets=targets,
             T=np.array([T]), dx_labels=np.array(dx_labels))
    print(f"Saved test logits -> {out_path}", flush=True)
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("version_tag", nargs="?", default="v2")
    parser.add_argument("--save-logits", action="store_true",
                        help="Save test logits to models/reports/ (used by generate_reports.py)")
    args = parser.parse_args()

    if args.save_logits:
        save_test_logits(args.version_tag)
    else:
        main(version_tag=args.version_tag)
