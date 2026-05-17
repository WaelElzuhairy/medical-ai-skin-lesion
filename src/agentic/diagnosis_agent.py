"""
Diagnosis Agent — cross-references CNN output with patient metadata.

Input:  InferenceResult + patient metadata dict
Output: structured JSON with agreement flag, contradictions, rationale

Rules:
  - Schema-validated output; retries once on parse failure.
  - LLM is forbidden from overriding the CNN prediction.
  - Returns a structured dict, never free text.
"""

from __future__ import annotations

from typing import Any

from src.agentic.anthropic_client import call_llm
from src.deep_learning.infer import InferenceResult

SYSTEM_PROMPT = """You are a clinical decision support assistant reviewing a CNN skin lesion classification.
Your role is to cross-reference the model's prediction with patient metadata and flag any contradictions.

RULES:
- You CANNOT override or change the CNN prediction.
- You may only flag contradictions or supporting evidence from the metadata.
- Output must be valid JSON matching this exact schema:
  {
    "agrees": true/false,
    "contradictions": ["string", ...],
    "supporting_factors": ["string", ...],
    "rationale": "one sentence"
  }
- "agrees" = true if metadata is consistent with the CNN prediction.
- "contradictions" = list of metadata factors that conflict with the prediction (empty list if none).
- "supporting_factors" = list of metadata factors that support the prediction (empty list if none).
- "rationale" = one concise sentence summarising the cross-reference.
- Do NOT make clinical recommendations.
- Do NOT invent information not present in the metadata.
"""


def run(result: InferenceResult, metadata: dict[str, Any]) -> dict:
    """Cross-reference CNN prediction with patient metadata.

    Parameters
    ----------
    result:   InferenceResult from infer() or vit_infer()
    metadata: patient metadata dict with keys like age, sex, localization

    Returns
    -------
    dict with keys: agrees, contradictions, supporting_factors, rationale
    """
    age          = metadata.get("age", "unknown")
    sex          = metadata.get("sex", "unknown")
    localization = metadata.get("localization", "unknown")

    dx_breakdown = "\n".join(
        f"  {lbl.upper()}: {prob:.2%}"
        for lbl, prob in zip(result.dx_labels, result.dx_probs)
    )

    user_msg = f"""CNN PREDICTION:
- Binary: {result.predicted_label.upper()}
- Most likely dx: {result.predicted_dx.upper()}
- Malignant probability: {result.malignant_prob:.2%}
- Router confidence: {result.confidence:.2%}

7-CLASS BREAKDOWN:
{dx_breakdown}

PATIENT METADATA:
- Age: {age}
- Sex: {sex}
- Lesion localization: {localization}

Cross-reference the prediction with the metadata. Identify any contradictions or supporting factors."""

    schema = {
        "type": "object",
        "properties": {
            "agrees":             {"type": "boolean"},
            "contradictions":     {"type": "array", "items": {"type": "string"}},
            "supporting_factors": {"type": "array", "items": {"type": "string"}},
            "rationale":          {"type": "string"},
        },
        "required": ["agrees", "contradictions", "supporting_factors", "rationale"],
    }

    for attempt in range(2):
        try:
            response = call_llm(SYSTEM_PROMPT, user_msg, schema=schema)
            # Validate required keys
            for key in ("agrees", "contradictions", "supporting_factors", "rationale"):
                if key not in response:
                    raise ValueError(f"Missing key: {key}")
            return response
        except ValueError as e:
            if attempt == 0:
                print(f"[DiagnosisAgent] Parse failed, retrying: {e}", flush=True)
                continue
            raise RuntimeError(f"DiagnosisAgent failed after 2 attempts: {e}") from e

    raise RuntimeError("DiagnosisAgent: unexpected exit")
