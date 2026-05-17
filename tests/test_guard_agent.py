"""
Unit tests for the Guard Agent.

Critical: all block/pass decisions are DETERMINISTIC — no LLM in the
decision path. These tests verify every hard rule independently.

Rules tested:
  1. Confidence >= GUARD_MIN_CONFIDENCE
  2. Mandatory disclaimer present in report draft
  3. Diagnosis Agent agreement (contradictions noted but don't hard-block)
  4. Evidence Agent status (insufficient_evidence = soft warning, not hard block)
  5. DOI resolution (unresolvable DOIs block the report)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import config
from src.agentic.guard_agent import check, GuardResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ok_diagnosis():
    return {"agrees": True, "contradictions": [], "supporting_factors": [], "rationale": "OK"}

def _ok_evidence():
    return {"status": "ok", "quotes": [{"text": "some quote", "doi": "", "pub_date": "2023"}]}

def _insufficient_evidence():
    return {"status": "insufficient_evidence", "quotes": []}


# ---------------------------------------------------------------------------
# Rule 1: Confidence gate
# ---------------------------------------------------------------------------

class TestConfidenceGate:

    def test_passes_above_gate(self):
        result = check(
            confidence=0.90,
            predicted_label="benign",
            diagnosis_result=_ok_diagnosis(),
            evidence_result=_ok_evidence(),
        )
        assert result.passed is True

    def test_passes_at_gate_boundary(self):
        result = check(
            confidence=config.GUARD_MIN_CONFIDENCE,
            predicted_label="benign",
            diagnosis_result=_ok_diagnosis(),
            evidence_result=_ok_evidence(),
        )
        assert result.passed is True

    def test_blocks_just_below_gate(self):
        result = check(
            confidence=config.GUARD_MIN_CONFIDENCE - 0.0001,
            predicted_label="benign",
            diagnosis_result=_ok_diagnosis(),
            evidence_result=_ok_evidence(),
        )
        assert result.passed is False
        assert any("confidence" in r.lower() for r in result.blocked_reasons)

    def test_blocks_low_confidence(self):
        result = check(
            confidence=0.40,
            predicted_label="benign",
            diagnosis_result=_ok_diagnosis(),
            evidence_result=_ok_evidence(),
        )
        assert result.passed is False


# ---------------------------------------------------------------------------
# Rule 2: Mandatory disclaimer
# ---------------------------------------------------------------------------

class TestDisclaimer:

    def test_passes_with_disclaimer(self):
        draft = f"Some report content.\n\n{config.CLINICAL_DISCLAIMER}"
        result = check(
            confidence=0.90,
            predicted_label="benign",
            diagnosis_result=_ok_diagnosis(),
            evidence_result=_ok_evidence(),
            report_draft=draft,
        )
        assert result.passed is True

    def test_blocks_missing_disclaimer(self):
        draft = "Some report content without the disclaimer."
        result = check(
            confidence=0.90,
            predicted_label="benign",
            diagnosis_result=_ok_diagnosis(),
            evidence_result=_ok_evidence(),
            report_draft=draft,
        )
        assert result.passed is False
        assert any("disclaimer" in r.lower() for r in result.blocked_reasons)

    def test_no_draft_skips_disclaimer_check(self):
        """If no draft is provided, disclaimer check is skipped (pre-report guard call)."""
        result = check(
            confidence=0.90,
            predicted_label="benign",
            diagnosis_result=_ok_diagnosis(),
            evidence_result=_ok_evidence(),
            report_draft="",
        )
        assert result.passed is True


# ---------------------------------------------------------------------------
# Rule 3: Diagnosis Agent
# ---------------------------------------------------------------------------

class TestDiagnosisAgent:

    def test_passes_when_agrees(self):
        diag = {"agrees": True, "contradictions": [], "rationale": "Consistent."}
        result = check(0.90, "benign", diag, _ok_evidence())
        assert result.passed is True

    def test_passes_with_contradictions_noted(self):
        """Contradictions are flagged but do NOT hard-block (metadata ≠ CNN override)."""
        diag = {
            "agrees": False,
            "contradictions": ["Patient age atypical for this dx"],
            "rationale": "Some inconsistency noted.",
        }
        result = check(0.90, "benign", diag, _ok_evidence())
        assert result.passed is True
        assert any("contradict" in r.lower() for r in result.reasons)


# ---------------------------------------------------------------------------
# Rule 4: Evidence Agent
# ---------------------------------------------------------------------------

class TestEvidenceAgent:

    def test_passes_with_ok_evidence(self):
        result = check(0.90, "benign", _ok_diagnosis(), _ok_evidence())
        assert result.passed is True

    def test_passes_with_insufficient_evidence_soft_warning(self):
        """insufficient_evidence is a soft warning — report still proceeds."""
        result = check(0.90, "benign", _ok_diagnosis(), _insufficient_evidence())
        assert result.passed is True
        assert any("soft warning" in r.lower() or "no literature" in r.lower()
                   for r in result.reasons)

    def test_insufficient_evidence_does_not_appear_in_blocks(self):
        result = check(0.90, "benign", _ok_diagnosis(), _insufficient_evidence())
        assert not any("insufficient" in r.lower() for r in result.blocked_reasons)


# ---------------------------------------------------------------------------
# Rule 5: DOI resolution
# ---------------------------------------------------------------------------

class TestDOIResolution:

    def test_passes_with_no_dois_in_draft(self):
        draft = f"Report with no DOIs.\n\n{config.CLINICAL_DISCLAIMER}"
        result = check(0.90, "benign", _ok_diagnosis(), _ok_evidence(), report_draft=draft)
        assert result.passed is True

    def test_blocks_unresolvable_doi(self, monkeypatch):
        """Patch _check_dois to simulate a failed DOI without network call."""
        import src.agentic.guard_agent as ga
        monkeypatch.setattr(ga, "_check_dois", lambda dois: dois)  # all fail

        draft = f"See 10.9999/fake-doi-that-does-not-exist.\n\n{config.CLINICAL_DISCLAIMER}"
        result = check(0.90, "benign", _ok_diagnosis(), _ok_evidence(), report_draft=draft)
        assert result.passed is False
        assert any("doi" in r.lower() for r in result.blocked_reasons)

    def test_passes_resolvable_doi(self, monkeypatch):
        """Patch _check_dois to simulate all DOIs resolving."""
        import src.agentic.guard_agent as ga
        monkeypatch.setattr(ga, "_check_dois", lambda dois: [])  # all pass

        draft = f"See 10.1000/real-doi.\n\n{config.CLINICAL_DISCLAIMER}"
        result = check(0.90, "benign", _ok_diagnosis(), _ok_evidence(), report_draft=draft)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Combined: multiple failures
# ---------------------------------------------------------------------------

class TestMultipleFailures:

    def test_all_failures_reported(self):
        draft = "No disclaimer here."
        result = check(
            confidence=0.40,           # fails rule 1
            predicted_label="benign",
            diagnosis_result=_ok_diagnosis(),
            evidence_result=_ok_evidence(),
            report_draft=draft,        # fails rule 2
        )
        assert result.passed is False
        assert len(result.blocked_reasons) >= 2
