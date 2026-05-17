"""
Uncertainty Agent — activated ONLY for MEDIUM confidence tier.

Produces a structured escalation report for clinician review.
FORBIDDEN from making clinical recommendations.
"""

from __future__ import annotations

from typing import Any

from src.agentic.anthropic_client import call_llm
from src.deep_learning.infer import InferenceResult

SYSTEM_PROMPT = """You are a clinical decision support assistant preparing an escalation report.
The AI model has MEDIUM confidence in its prediction — this case requires human clinician review.

RULES:
- Do NOT make any clinical recommendation or suggest a diagnosis.
- Do NOT suggest treatment or next steps beyond "refer to clinician".
- Only describe what the model found and why it is uncertain.
- Output must be valid JSON matching this exact schema:
  {
    "confidence_value": 0.0,
    "predicted_dx": "string",
    "predicted_binary": "string",
    "malignant_probability": 0.0,
    "ambiguities": ["string", ...],
    "escalation_reason": "string",
    "recommended_action": "Refer to qualified clinician for review."
  }
- "ambiguities" = list of reasons why the model is uncertain (max 3 items).
- "escalation_reason" = one sentence explaining why this case cannot be auto-reported.
- "recommended_action" must always be exactly: "Refer to qualified clinician for review."
"""


def run(result: InferenceResult, metadata: dict[str, Any]) -> dict:
    """Generate a structured escalation report for MEDIUM tier cases.

    Parameters
    ----------
    result:   InferenceResult from inference
    metadata: patient metadata dict

    Returns
    -------
    dict with escalation report fields
    """
    dx_breakdown = "\n".join(
        f"  {lbl.upper()}: {prob:.2%}"
        for lbl, prob in zip(result.dx_labels, result.dx_probs)
    )

    user_msg = f"""MEDIUM CONFIDENCE CASE — requires escalation.

MODEL OUTPUT:
- Predicted binary: {result.predicted_label.upper()}
- Most likely dx: {result.predicted_dx.upper()}
- Router confidence: {result.confidence:.2%}
- Malignant probability: {result.malignant_prob:.2%}

7-CLASS BREAKDOWN:
{dx_breakdown}

PATIENT:
- Age: {metadata.get('age', 'unknown')}
- Sex: {metadata.get('sex', 'unknown')}
- Localization: {metadata.get('localization', 'unknown')}

Generate the escalation report. Identify why this case is uncertain."""

    schema = {
        "type": "object",
        "required": [
            "confidence_value", "predicted_dx", "predicted_binary",
            "malignant_probability", "ambiguities",
            "escalation_reason", "recommended_action"
        ],
    }

    for attempt in range(2):
        try:
            response = call_llm(SYSTEM_PROMPT, user_msg, schema=schema)
            # Force recommended_action to the exact string
            response["recommended_action"] = "Refer to qualified clinician for review."
            # Fill in values from result directly (don't trust LLM for numbers)
            response["confidence_value"]      = round(float(result.confidence), 4)
            response["malignant_probability"] = round(float(result.malignant_prob), 4)
            response["predicted_dx"]          = result.predicted_dx
            response["predicted_binary"]      = result.predicted_label
            return response
        except ValueError as e:
            if attempt == 0:
                print(f"[UncertaintyAgent] Parse failed, retrying: {e}", flush=True)
                continue
            raise RuntimeError(f"UncertaintyAgent failed after 2 attempts: {e}") from e

    raise RuntimeError("UncertaintyAgent: unexpected exit")
