"""
Confidence router — RULE-BASED, NO LLM.

Routes a calibrated confidence score to one of three tiers based on
hard thresholds defined in config.py. This is a pure function with
zero side-effects and zero LLM calls — it must stay that way.

Confidence = max(P(benign), P(malignant)) from the calibrated softmax.
A clearly benign case (P(benign)=0.95) has confidence=0.95 → HIGH.
A genuinely uncertain case (P(benign)≈P(malignant)≈0.50) → LOW/MEDIUM.

Tier meanings:
  HIGH   (> 0.85) → full agentic report path
  MEDIUM (0.50–0.85) → escalate to clinician, no auto-report
  LOW    (< 0.50) → reject, model is genuinely uncertain
"""

from __future__ import annotations

from enum import Enum

import config


class Tier(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


def route(confidence: float) -> Tier:
    """Map a calibrated confidence score to a routing tier.

    Parameters
    ----------
    confidence: float in [0, 1] — max(P(benign), P(malignant))

    Returns
    -------
    Tier.LOW / Tier.MEDIUM / Tier.HIGH
    """
    if confidence < config.ROUTER_LOW_MAX:
        return Tier.LOW
    elif confidence <= config.ROUTER_HIGH_MIN:
        return Tier.MEDIUM
    else:
        return Tier.HIGH


def route_description(tier: Tier, result) -> str:
    """Return a human-readable routing decision for logs / UI."""
    if tier == Tier.LOW:
        return (
            f"LOW confidence ({result.confidence:.2%}) — model is genuinely uncertain. "
            f"Predicted {result.predicted_label.upper()} ({result.predicted_dx.upper()}). "
            f"Case rejected — manual review required."
        )
    elif tier == Tier.MEDIUM:
        return (
            f"MEDIUM confidence ({result.confidence:.2%}) — escalating to clinician. "
            f"Predicted {result.predicted_label.upper()} ({result.predicted_dx.upper()}). "
            f"Malignant probability: {result.malignant_prob:.2%}."
        )
    else:
        return (
            f"HIGH confidence ({result.confidence:.2%}) — proceeding to full report. "
            f"Predicted {result.predicted_label.upper()} ({result.predicted_dx.upper()}). "
            f"Malignant probability: {result.malignant_prob:.2%}."
        )
