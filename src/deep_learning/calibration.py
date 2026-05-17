"""
Temperature scaling for confidence calibration (Guo et al., 2017).

Raw softmax probabilities from a deep net are systematically overconfident.
The Confidence Router routes cases by these probabilities, so an uncalibrated
model would mis-route — sending genuinely uncertain cases down the
high-confidence "auto-report" path. Temperature scaling fits a single scalar T
on validation logits to flatten the distribution. ECE (Expected Calibration
Error) measures how much it helped.

Per the project plan, raw softmax is never used downstream — `infer.py`
always divides logits by T before softmax.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config


def expected_calibration_error(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = config.ECE_NUM_BINS,
) -> float:
    """ECE on the predicted class confidence."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if not in_bin.any():
            continue
        bin_size = in_bin.sum()
        avg_conf = confidences[in_bin].mean()
        avg_acc = accuracies[in_bin].mean()
        ece += (bin_size / n) * abs(avg_conf - avg_acc)
    return float(ece)


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Fit scalar T minimizing NLL via LBFGS."""
    logits_t = torch.from_numpy(logits).float()
    targets_t = torch.from_numpy(targets).long()
    log_T = nn.Parameter(torch.zeros(1))  # T = exp(log_T) stays positive
    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T)
        loss = nll(logits_t / T, targets_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_T).detach().item())


def _binary_collapse(probs: np.ndarray) -> np.ndarray:
    """Collapse (N,7) probs to (N,2) [benign, malignant]."""
    mal = probs[:, config.MALIGNANT_CLASS_INDICES].sum(axis=1, keepdims=True)
    return np.concatenate([1.0 - mal, mal], axis=1)


def _binary_targets(targets7: np.ndarray) -> np.ndarray:
    """Map 7-class targets to binary (1=malignant, 0=benign)."""
    return np.array([1 if t in config.MALIGNANT_CLASS_INDICES else 0 for t in targets7])


def _fit_temperature_binary(logits7: np.ndarray, bin_targets: np.ndarray) -> float:
    """Fit T by minimising binary NLL on the collapsed malignant probability."""
    logits_t   = torch.from_numpy(logits7).float()
    targets_t  = torch.from_numpy(bin_targets).long()
    mal_idx    = torch.tensor(config.MALIGNANT_CLASS_INDICES)
    log_T      = nn.Parameter(torch.zeros(1))
    optimizer  = torch.optim.LBFGS([log_T], lr=0.1, max_iter=200)
    bce        = nn.BCELoss()

    def closure():
        optimizer.zero_grad()
        T          = torch.exp(log_T)
        probs7     = torch.softmax(logits_t / T, dim=1)
        mal_prob   = probs7[:, mal_idx].sum(dim=1)
        mal_prob   = mal_prob.clamp(1e-7, 1 - 1e-7)
        loss       = bce(mal_prob, targets_t.float())
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_T).detach().item())


def calibrate_from_saved_logits(version_tag: str = "v1") -> dict:
    """Load val logits saved by train.py, fit T on binary collapse, write json.

    For 7-class models we fit T to minimise binary NLL (benign vs malignant)
    because that is what the confidence router actually sees. Fitting on 7-class
    NLL with label-smoothed training gives a T barely above 1 that can worsen
    ECE on the binary task.
    """
    npz_path = config.CALIBRATION_DIR / f"val_logits_{version_tag}.npz"
    out_path = config.CALIBRATION_DIR / f"{version_tag}.json"

    data = np.load(npz_path)
    logits, targets = data["logits"], data["targets"]

    # Drop rows with any NaN/Inf (can occur from AMP fp16 overflow)
    valid = np.isfinite(logits).all(axis=1)
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"Warning: dropping {n_dropped} NaN/Inf logit rows before calibration.")
        logits, targets = logits[valid], targets[valid]

    is_7class = logits.shape[1] == config.NUM_FINEGRAINED_CLASSES

    if is_7class:
        # Fit T on binary collapse probabilities
        bin_targets = _binary_targets(targets)
        pre_probs2  = _binary_collapse(_softmax(logits))
        pre_ece     = expected_calibration_error(pre_probs2, bin_targets)

        # Build binary logits for fitting: log(malignant_prob / benign_prob)
        # We fit T on the 7-class logits but evaluate on binary collapse
        T = _fit_temperature_binary(logits, bin_targets)

        post_probs2 = _binary_collapse(_softmax(logits / T))
        post_ece    = expected_calibration_error(post_probs2, bin_targets)
        task_label  = "binary-collapse"
    else:
        pre_probs  = _softmax(logits)
        pre_ece    = expected_calibration_error(pre_probs, targets)
        T          = fit_temperature(logits, targets)
        post_probs = _softmax(logits / T)
        post_ece   = expected_calibration_error(post_probs, targets)
        task_label = "binary"

    payload = {
        "version_tag":   version_tag,
        "temperature":   T,
        "ece_before":    pre_ece,
        "ece_after":     post_ece,
        "ece_target":    config.ECE_TARGET,
        "calibrated_on": task_label,
        "n_bins":        config.ECE_NUM_BINS,
        "n_samples":     int(len(targets)),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"T = {T:.4f}    ECE ({task_label}): {pre_ece:.4f} -> {post_ece:.4f}    target<{config.ECE_TARGET}")
    print(f"Calibration written to {out_path}")
    return payload


def load_temperature(version_tag: str = "v1") -> float:
    path = config.CALIBRATION_DIR / f"{version_tag}.json"
    if not path.is_file():
        return 1.0
    return float(json.loads(path.read_text())["temperature"])


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
