"""
Orchestrator — wires the confidence router to the agent pipeline.

No LLM calls here. This is pure routing logic.

Flow:
  HIGH   → diagnosis_agent → evidence_agent → guard_agent → report_agent
  MEDIUM → uncertainty_agent → return escalation report
  LOW    → return rejection

Evidence agent is intentionally skipped for LOW tier (no LLM cost on rejects).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agentic.confidence_router import Tier, route, route_description
from src.deep_learning.infer import InferenceResult

EVIDENCE_PLACEHOLDER = {"status": "skipped", "quotes": []}


@dataclass
class OrchestratorResult:
    tier:              Tier
    route_description: str
    report:            str | None = None          # markdown report (HIGH tier only)
    escalation:        dict | None = None         # uncertainty report (MEDIUM tier)
    diagnosis:         dict | None = None         # diagnosis agent output
    evidence:          dict | None = None         # evidence agent output
    guard_passed:      bool | None = None
    guard_reasons:     list[str] = field(default_factory=list)
    guard_blocks:      list[str] = field(default_factory=list)
    error:             str | None = None


def run(
    result: InferenceResult,
    metadata: dict[str, Any],
    case_id: str | None = None,
) -> OrchestratorResult:
    """Run the full agentic pipeline for a single inference result.

    Parameters
    ----------
    result:   InferenceResult from vit_infer() or infer()
    metadata: patient metadata dict (age, sex, localization)
    case_id:  optional case identifier for the report

    Returns
    -------
    OrchestratorResult with tier + appropriate outputs
    """
    from src.agentic import (
        diagnosis_agent,
        evidence_agent,
        guard_agent,
        report_agent,
        uncertainty_agent,
    )

    tier  = route(result.confidence)
    descr = route_description(tier, result)
    print(f"[Orchestrator] {descr}", flush=True)

    # -----------------------------------------------------------------------
    # LOW — reject
    # -----------------------------------------------------------------------
    if tier == Tier.LOW:
        return OrchestratorResult(
            tier=tier,
            route_description=descr,
            report=None,
        )

    # -----------------------------------------------------------------------
    # MEDIUM — escalate
    # -----------------------------------------------------------------------
    if tier == Tier.MEDIUM:
        try:
            escalation = uncertainty_agent.run(result, metadata)
        except Exception as e:
            return OrchestratorResult(
                tier=tier,
                route_description=descr,
                error=f"UncertaintyAgent failed: {e}",
            )
        return OrchestratorResult(
            tier=tier,
            route_description=descr,
            escalation=escalation,
        )

    # -----------------------------------------------------------------------
    # HIGH — full pipeline
    # -----------------------------------------------------------------------
    # Step 1: Diagnosis Agent
    try:
        diagnosis = diagnosis_agent.run(result, metadata)
    except Exception as e:
        return OrchestratorResult(
            tier=tier,
            route_description=descr,
            error=f"DiagnosisAgent failed: {e}",
        )

    # Step 2: Evidence Agent
    try:
        evidence = evidence_agent.run(result, metadata)
    except Exception as e:
        return OrchestratorResult(
            tier=tier,
            route_description=descr,
            diagnosis=diagnosis,
            error=f"EvidenceAgent failed: {e}",
        )

    # Step 3: Guard Agent (pre-report check, no report draft yet)
    guard = guard_agent.check(
        confidence=result.confidence,
        predicted_label=result.predicted_label,
        diagnosis_result=diagnosis,
        evidence_result=evidence,
        report_draft="",  # no draft yet — disclaimer check happens after
    )

    if not guard.passed:
        return OrchestratorResult(
            tier=tier,
            route_description=descr,
            diagnosis=diagnosis,
            evidence=evidence,
            guard_passed=False,
            guard_reasons=guard.reasons,
            guard_blocks=guard.blocked_reasons,
            report=None,
        )

    # Step 4: Report Agent
    try:
        report = report_agent.run(result, metadata, diagnosis, evidence, case_id=case_id)
    except Exception as e:
        return OrchestratorResult(
            tier=tier,
            route_description=descr,
            diagnosis=diagnosis,
            evidence=evidence,
            guard_passed=True,
            error=f"ReportAgent failed: {e}",
        )

    # Step 5: Final Guard check (with report draft — verifies disclaimer)
    final_guard = guard_agent.check(
        confidence=result.confidence,
        predicted_label=result.predicted_label,
        diagnosis_result=diagnosis,
        evidence_result=evidence,
        report_draft=report,
    )

    if not final_guard.passed:
        return OrchestratorResult(
            tier=tier,
            route_description=descr,
            diagnosis=diagnosis,
            evidence=evidence,
            guard_passed=False,
            guard_reasons=final_guard.reasons,
            guard_blocks=final_guard.blocked_reasons,
            report=None,
        )

    return OrchestratorResult(
        tier=tier,
        route_description=descr,
        report=report,
        diagnosis=diagnosis,
        evidence=evidence,
        guard_passed=True,
        guard_reasons=final_guard.reasons,
        guard_blocks=[],
    )
