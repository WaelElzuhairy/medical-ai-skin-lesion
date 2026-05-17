"""
Evaluate the HuggingFace ViT model (Anwarkh1/Skin_Cancer-Image_Classification)
on our properly-split HAM10000 test set (split by lesion_id, no leakage).

This gives the real accuracy of that model on held-out data, vs the 96.95%
claimed on their own (likely leaky) validation split.

Run:
    python scripts/evaluate_hf_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from transformers import AutoImageProcessor, AutoModelForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.deep_learning.calibration import expected_calibration_error

HF_MODEL = "Anwarkh1/Skin_Cancer-Image_Classification"

# Map HF model labels -> our HAM10000_DX_LABELS indices
# HF label order from model card:
HF_LABELS = [
    "Benign keratosis-like lesions",   # bkl
    "Basal cell carcinoma",             # bcc
    "Actinic keratoses",                # akiec
    "Vascular lesions",                 # vasc
    "Melanocytic nevi",                 # nv
    "Melanoma",                         # mel
    "Dermatofibroma",                   # df
]
# Mapping from HF class index -> our dx_label index
HF_TO_DX = {
    0: config.HAM10000_DX_LABELS.index("bkl"),
    1: config.HAM10000_DX_LABELS.index("bcc"),
    2: config.HAM10000_DX_LABELS.index("akiec"),
    3: config.HAM10000_DX_LABELS.index("vasc"),
    4: config.HAM10000_DX_LABELS.index("nv"),
    5: config.HAM10000_DX_LABELS.index("mel"),
    6: config.HAM10000_DX_LABELS.index("df"),
}


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Loading {HF_MODEL}...", flush=True)

    processor = AutoImageProcessor.from_pretrained(HF_MODEL)
    model = AutoModelForImageClassification.from_pretrained(HF_MODEL)
    model = model.to(device).eval()
    print(f"Model loaded. Labels: {model.config.id2label}", flush=True)

    test_df = pd.read_parquet(config.PROCESSED_DIR / "test.parquet")
    print(f"Test set: {len(test_df)} images", flush=True)

    all_logits, all_targets = [], []

    for i, row in test_df.iterrows():
        if i % 100 == 0:
            print(f"  {i}/{len(test_df)}...", flush=True)
        img = Image.open(row["image_path"]).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        all_logits.append(outputs.logits.cpu().float().numpy())
        all_targets.append(int(row["dx_label"]))

    logits_hf = np.concatenate(all_logits, axis=0)   # shape (N, 7) in HF label order
    targets    = np.array(all_targets)                 # in our dx_label order

    # Reorder logits from HF label order -> our dx_label order
    logits_ours = np.zeros_like(logits_hf)
    for hf_idx, our_idx in HF_TO_DX.items():
        logits_ours[:, our_idx] = logits_hf[:, hf_idx]

    probs  = _softmax(logits_ours)
    preds  = probs.argmax(axis=1)

    # Binary collapse
    mal_prob   = probs[:, config.MALIGNANT_CLASS_INDICES].sum(axis=1)
    bin_probs  = np.stack([1 - mal_prob, mal_prob], axis=1)
    bin_targets = np.array([1 if t in config.MALIGNANT_CLASS_INDICES else 0 for t in targets])
    bin_preds   = bin_probs.argmax(axis=1)
    ece = expected_calibration_error(bin_probs, bin_targets)

    print(f"\n{'='*60}")
    print(f"  HF ViT model — tested on OUR leakage-free test split")
    print(f"{'='*60}")
    print("\n7-CLASS:")
    print(classification_report(targets, preds, target_names=config.HAM10000_DX_LABELS, digits=4))
    print(f"7-class macro-F1: {f1_score(targets, preds, average='macro'):.4f}")
    print(f"\nBINARY COLLAPSE:")
    print(classification_report(bin_targets, bin_preds, target_names=config.BINARY_LABELS, digits=4))
    print(f"Binary macro-F1: {f1_score(bin_targets, bin_preds, average='macro'):.4f}")
    print(f"ECE (uncalibrated): {ece:.4f}")

    print(f"\n--- COMPARISON ---")
    print(f"Our v2:   7-class F1=0.4431  binary F1=0.7477  ECE=0.029")
    print(f"HF ViT:   7-class F1={f1_score(targets, preds, average='macro'):.4f}  binary F1={f1_score(bin_targets, bin_preds, average='macro'):.4f}  ECE={ece:.4f} (uncal)")


if __name__ == "__main__":
    main()
