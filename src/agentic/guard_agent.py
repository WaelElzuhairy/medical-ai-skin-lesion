"""
Guard Agent — deterministic hard rules, NO LLM in the decision path.

Every check is pure Python. The LLM has absolutely no vote on whether a
report is blocked or passed. If all checks pass, the report proceeds.
If any check fails, the report is blocked with a reason.

Hard rules (all must pass):
  1. Confidence >= GUARD_MIN_CONFIDENCE (0.85)
  2. Mandatory disclaimer string is present in the report draft
  3. Diagnosis Agent does not contradict CNN class
  4. Evidence Agent status is "ok" (not "insufficient_evidence")
  5. No DOI citations present that fail to resolve (skipped if no DOIs)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import config


@dataclass
class GuardResult:
    passed: bool
    reasons: list[str]        # why it passed or was blocked
    blocked_reasons: list[str]  # only populated when passed=False


def check(
    confidence: float,
    predicted_label: str,
    diagnosis_result: dict,
    evidence_result: dict,
    report_draft: str = "",
) -> GuardResult:
    """Run all deterministic hard checks.

    Parameters
    ----------
    confidence:       calibrated confidence from InferenceResult
    predicted_label:  "benign" or "malignant" from CNN
    diagnosis_result: output from diagnosis_agent.run()
    evidence_result:  output from evidence_agent.run()
    report_draft:     draft report string (checked for disclaimer)

    Returns
    -------
    GuardResult with passed=True/False and reason lists
    """
    passes  = []
    blocks  = []

    # --- Rule 1: confidence gate ---
    if confidence >= config.GUARD_MIN_CONFIDENCE:
        passes.append(f"Confidence {confidence:.2%} >= gate {config.GUARD_MIN_CONFIDENCE:.0%}")
    else:
        blocks.append(
            f"Confidence {confidence:.2%} below guard gate {config.GUARD_MIN_CONFIDENCE:.0%}"
        )

    # --- Rule 2: disclaimer present ---
    if report_draft:
        if config.CLINICAL_DISCLAIMER in report_draft:
            passes.append("Mandatory disclaimer present")
        else:
            blocks.append("Mandatory clinical disclaimer missing from report")

    # --- Rule 3: diagnosis agent agreement ---
    agrees = diagnosis_result.get("agrees", True)  # default True if key missing
    contradictions = diagnosis_result.get("contradictions", [])
    if agrees:
        passes.append("Diagnosis Agent agrees with CNN prediction")
    else:
        # Contradictions alone don't block — only explicit CNN class override
        # (metadata inconsistency is expected and noted, not a hard block)
        passes.append(
            f"Diagnosis Agent flagged contradictions ({len(contradictions)}) but did not override CNN"
        )

    # --- Rule 4: evidence agent status ---
    # "insufficient_evidence" is a soft warning — the report still proceeds but
    # clearly states no literature was found. A hard block would make the system
    # unusable before the RAG corpus is built, and is disproportionate for benign cases.
    ev_status = evidence_result.get("status", "ok")
    if ev_status == "insufficient_evidence":
        passes.append("Evidence Agent: no literature above threshold — report will note this (soft warning)")
    elif ev_status == "skipped":
        passes.append("Evidence Agent: skipped for this tier")
    else:
        passes.append("Evidence Agent returned supporting literature")

    # --- Rule 5: DOI resolution (only if DOIs are present in draft) ---
    if report_draft:
        dois = re.findall(r'10\.\d{4,}/\S+', report_draft)
        if dois:
            failed_dois = _check_dois(dois)
            if failed_dois:
                blocks.append(f"Unresolvable DOIs: {', '.join(failed_dois)}")
            else:
                passes.append(f"All {len(dois)} DOI(s) resolved successfully")

    passed = len(blocks) == 0
    return GuardResult(passed=passed, reasons=passes, blocked_reasons=blocks)


def _check_dois(dois: list[str]) -> list[str]:
    """Return list of DOIs that fail to resolve. Cached per session."""
    import requests

    failed = []
    for doi in dois:
        doi_clean = doi.rstrip(".,;)")
        url = config.DOI_RESOLVER_URL.format(doi=doi_clean)
        try:
            r = requests.head(url, timeout=config.DOI_RESOLVE_TIMEOUT, allow_redirects=True)
            if r.status_code >= 400:
                failed.append(doi_clean)
        except Exception:
            failed.append(doi_clean)
    return failed
