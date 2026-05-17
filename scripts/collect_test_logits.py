"""
Collect test-set logits for a ViT or EfficientNet checkpoint and save to
models/reports/test_logits_{version_tag}.npz

This script is intentionally minimal — no matplotlib, no sklearn — so that
ViT inference (transformers + PyTorch CUDA) runs without the module conflicts
that cause segfaults in the full generate_reports.py process.

Usage:
    python scripts/collect_test_logits.py vit_v1
    python scripts/collect_test_logits.py v2
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

OUT_DIR = config.MODELS_DIR / "reports"


def collect_vit(version_tag: str, device: torch.device) -> Path:
    from transformers import AutoImageProcessor
    from src.deep_learning.vit_train import ViTDataset, load_vit_checkpoint
    from src.deep_learning.calibration import load_temperature

    inner_tag = version_tag[len("vit_"):] if version_tag.startswith("vit_") else version_tag
    ckpt_path = config.CHECKPOINTS_DIR / f"vit_{inner_tag}.pt"

    model, meta = load_vit_checkpoint(ckpt_path, device=device)
    model.eval()
    T         = load_temperature(version_tag=f"vit_{inner_tag}")
    dx_labels = meta.get("dx_labels", config.HAM10000_DX_LABELS)

    processor = AutoImageProcessor.from_pretrained(
        meta.get("model_name", config.VIT_BASE_MODEL)
    )

    test_ds = ViTDataset(
        config.PROCESSED_DIR / "test.parquet",
        processor,
        augment=False,
    )
    loader = DataLoader(
        test_ds,
        batch_size=config.VIT_EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    all_logits, all_targets = [], []
    with torch.no_grad():
        for i, (pv, y) in enumerate(loader):
            pv = pv.to(device)
            logits = model(pixel_values=pv).logits.float().cpu().numpy()
            all_logits.append(logits)
            all_targets.append(y.numpy())
            if (i + 1) % 5 == 0:
                print(f"  batch {i+1}/{len(loader)}", flush=True)

    logits  = np.concatenate(all_logits)
    targets = np.concatenate(all_targets).astype(int)
    return logits, targets, T, dx_labels


def collect_efficientnet(version_tag: str, device: torch.device):
    from src.deep_learning.model import load_checkpoint
    from src.deep_learning.calibration import load_temperature
    from src.deep_learning.train import HAM10000Dataset
    from src.preprocessing.augmentation import build_eval_transform

    ckpt_path = config.CHECKPOINTS_DIR / f"efficientnet_b4_{version_tag}.pt"
    model, meta = load_checkpoint(ckpt_path, device=device)
    model.eval()
    T         = load_temperature(version_tag=version_tag)
    dx_labels = meta.get("dx_labels", config.HAM10000_DX_LABELS)

    test_ds = HAM10000Dataset(
        config.PROCESSED_DIR / "test.parquet",
        build_eval_transform(),
        label_col="dx_label",
    )
    loader = DataLoader(test_ds, batch_size=config.EVAL_BATCH_SIZE, shuffle=False, num_workers=0)

    all_logits, all_targets = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x).cpu().float().numpy()
            all_logits.append(logits)
            all_targets.append(y.numpy())
            if (i + 1) % 5 == 0:
                print(f"  batch {i+1}/{len(loader)}", flush=True)

    logits  = np.concatenate(all_logits)
    targets = np.concatenate(all_targets).astype(int)
    return logits, targets, T, dx_labels


def main(version_tag: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"test_logits_{version_tag}.npz"

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_vit    = version_tag.startswith("vit_")
    arch      = "ViT-Base-16" if is_vit else "EfficientNet-B4"

    print(f"Collecting test logits: {arch} / {version_tag} on {device}", flush=True)

    if is_vit:
        logits, targets, T, dx_labels = collect_vit(version_tag, device)
    else:
        logits, targets, T, dx_labels = collect_efficientnet(version_tag, device)

    np.savez(
        out_path,
        logits=logits,
        targets=targets,
        T=np.array([T]),
        dx_labels=np.array(dx_labels),
    )
    print(f"Saved: {out_path}  (logits={logits.shape}, T={T:.4f})", flush=True)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "vit_v1"
    main(tag)
