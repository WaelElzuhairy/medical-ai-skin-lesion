"""
Report Agent — template slot-filling only, NO free clinical narrative.

The LLM is only permitted to copy validated content into named slots.
It cannot generate new clinical claims, invent citations, or deviate
from the template structure. The system prompt enforces this explicitly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import config
from src.agentic.anthropic_client import call_llm
from src.deep_learning.infer import InferenceResult

SYSTEM_PROMPT = """You are a report formatting assistant. Your ONLY job is to fill in
the provided template slots with the validated information given to you.

STRICT RULES:
- Copy information exactly as given — do NOT paraphrase or add clinical interpretation.
- Do NOT generate new clinical claims, diagnoses, or recommendations.
- Do NOT add citations not present in the evidence provided.
- Do NOT modify the disclaimer text.
- Fill every slot with the exact data provided.
- Output valid JSON matching the slot schema exactly.
"""

REPORT_TEMPLATE = """# AI Clinical Decision Support Report

**Date:** {date}
**Case ID:** {case_id}

---

## Model Prediction

| Field | Value |
|-------|-------|
| Binary Classification | **{predicted_label}** |
| Most Likely Dx | {predicted_dx} ({predicted_dx_full}) |
| Malignant Probability | {malignant_prob} |
| Router Confidence | {confidence} |
| Temperature (T) | {temperature} |

### 7-Class Breakdown
{dx_breakdown}

---

## Patient Information
- **Age:** {age}
- **Sex:** {sex}
- **Lesion Localization:** {localization}

---

## Diagnosis Cross-Reference
{diagnosis_rationale}

{contradictions_section}

---

## Supporting Evidence
{evidence_section}

---

## Disclaimer

{disclaimer}
"""

DX_FULLNAMES = {
    "akiec": "Actinic Keratosis / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevus",
    "vasc":  "Vascular Lesion",
}


def run(
    result: InferenceResult,
    metadata: dict[str, Any],
    diagnosis_result: dict,
    evidence_result: dict,
    case_id: str | None = None,
) -> str:
    """Fill the report template with validated data.

    Returns the completed report as a markdown string.
    The LLM is used only to format the evidence quotes section.
    """
    import uuid
    case_id = case_id or str(uuid.uuid4())[:8].upper()

    # --- Build dx breakdown table ---
    dx_rows = "\n".join(
        f"| {lbl.upper()} | {DX_FULLNAMES.get(lbl, lbl)} | {prob:.2%} |"
        for lbl, prob in zip(result.dx_labels, result.dx_probs)
    )
    dx_breakdown = f"| Code | Full Name | Probability |\n|------|-----------|-------------|\n{dx_rows}"

    # --- Contradictions section ---
    contradictions = diagnosis_result.get("contradictions", [])
    if contradictions:
        items = "\n".join(f"- {c}" for c in contradictions)
        contradictions_section = f"**Metadata contradictions flagged:**\n{items}"
    else:
        contradictions_section = "*No metadata contradictions detected.*"

    # --- Evidence section (LLM formats quotes only) ---
    evidence_section = _format_evidence(evidence_result)

    report = REPORT_TEMPLATE.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        case_id=case_id,
        predicted_label=result.predicted_label.upper(),
        predicted_dx=result.predicted_dx.upper(),
        predicted_dx_full=DX_FULLNAMES.get(result.predicted_dx, result.predicted_dx),
        malignant_prob=f"{result.malignant_prob:.2%}",
        confidence=f"{result.confidence:.2%}",
        temperature=f"{result.temperature:.4f}",
        dx_breakdown=dx_breakdown,
        age=metadata.get("age", "Not provided"),
        sex=metadata.get("sex", "Not provided"),
        localization=metadata.get("localization", "Not provided"),
        diagnosis_rationale=diagnosis_result.get("rationale", "No rationale provided."),
        contradictions_section=contradictions_section,
        evidence_section=evidence_section,
        disclaimer=config.CLINICAL_DISCLAIMER,
    )

    return report


def _format_evidence(evidence_result: dict) -> str:
    """Format the evidence quotes section from evidence agent output."""
    if evidence_result.get("status") == "insufficient_evidence":
        return "*Insufficient relevant literature found (similarity below threshold).*"

    quotes = evidence_result.get("quotes", [])
    if not quotes:
        return "*No supporting literature retrieved.*"

    lines = []
    for q in quotes:
        text  = q.get("text", "")
        doi   = q.get("doi", "")
        year  = q.get("pub_date", "")
        citation = f"(DOI: {doi}, {year})" if doi else f"({year})" if year else ""
        lines.append(f'> "{text}" {citation}')

    return "\n\n".join(lines)
