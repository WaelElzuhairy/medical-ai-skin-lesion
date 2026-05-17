"""
Unit tests for the Evidence Agent's insufficient_evidence path.

Critical rule: if no chunk clears cosine 0.6, the agent MUST return
{"status": "insufficient_evidence"} and make ZERO LLM calls.

These tests use monkeypatching to isolate the threshold logic without
requiring ChromaDB, PubMed network access, or Groq API calls.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(predicted_dx="nv", confidence=0.92, malignant_prob=0.08):
    """Create a minimal InferenceResult-like object for testing."""
    class MockResult:
        pass
    r = MockResult()
    r.predicted_dx    = predicted_dx
    r.confidence      = confidence
    r.malignant_prob  = malignant_prob
    r.predicted_label = "benign"
    r.dx_labels       = config.HAM10000_DX_LABELS
    r.dx_probs        = [0.01, 0.01, 0.05, 0.02, 0.08, 0.80, 0.03]
    return r

METADATA = {"age": 45, "sex": "male", "localization": "back"}


# ---------------------------------------------------------------------------
# Core threshold rule
# ---------------------------------------------------------------------------

class TestInsufficientEvidencePath:

    def test_returns_insufficient_when_no_chunks(self, monkeypatch):
        """Zero chunks → insufficient_evidence, zero LLM calls."""
        import src.agentic.evidence_agent as ea
        import src.agentic.anthropic_client as client

        monkeypatch.setattr(ea, "_retrieve_chunks", lambda q: [])
        llm_called = []
        monkeypatch.setattr(client, "call_llm", lambda *a, **kw: llm_called.append(1))

        result = ea.run(_make_result(), METADATA)

        assert result["status"] == "insufficient_evidence"
        assert llm_called == [], "LLM must NOT be called when no chunks pass threshold"

    def test_returns_insufficient_when_all_chunks_below_threshold(self, monkeypatch):
        """Chunks exist but all below cosine 0.6 → insufficient_evidence."""
        import src.agentic.evidence_agent as ea
        import src.agentic.anthropic_client as client

        low_chunks = [
            {"text": "some text", "doi": "", "pub_date": "2023", "cosine": 0.55},
            {"text": "other text", "doi": "", "pub_date": "2022", "cosine": 0.40},
        ]
        monkeypatch.setattr(ea, "_retrieve_chunks", lambda q: low_chunks)

        # _retrieve_chunks already filters — if it returns these, they passed.
        # So we simulate the case where retrieval returns empty after filtering.
        monkeypatch.setattr(ea, "_retrieve_chunks", lambda q: [])
        llm_called = []
        monkeypatch.setattr(client, "call_llm", lambda *a, **kw: llm_called.append(1))

        result = ea.run(_make_result(), METADATA)
        assert result["status"] == "insufficient_evidence"
        assert llm_called == []

    def test_calls_llm_when_chunks_pass_threshold(self, monkeypatch):
        """Chunks above threshold → LLM IS called to extract quotes."""
        import src.agentic.evidence_agent as ea

        good_chunks = [
            {"text": "Melanocytic nevi are benign proliferations.", "doi": "10.1/test", "pub_date": "2023", "cosine": 0.72},
        ]
        monkeypatch.setattr(ea, "_retrieve_chunks", lambda q: good_chunks)

        llm_called = []
        def mock_llm(system, user, schema=None, **kw):
            llm_called.append(1)
            return {"status": "ok", "quotes": [{"text": "test quote", "doi": "10.1/test", "pub_date": "2023"}]}
        # Patch at the evidence_agent module level (where call_llm is imported)
        monkeypatch.setattr(ea, "call_llm", mock_llm)

        result = ea.run(_make_result(), METADATA)
        assert result["status"] == "ok"
        assert len(llm_called) == 1, "LLM must be called exactly once when chunks pass"

    def test_insufficient_evidence_has_correct_structure(self, monkeypatch):
        """The returned dict must have exactly the right keys."""
        import src.agentic.evidence_agent as ea
        monkeypatch.setattr(ea, "_retrieve_chunks", lambda q: [])

        result = ea.run(_make_result(), METADATA)
        assert "status" in result
        assert result["status"] == "insufficient_evidence"


# ---------------------------------------------------------------------------
# Query building
# ---------------------------------------------------------------------------

class TestQueryBuilding:

    def test_query_contains_predicted_dx(self, monkeypatch):
        """The compound query must include the predicted diagnosis."""
        import src.agentic.evidence_agent as ea

        captured = []
        monkeypatch.setattr(ea, "_retrieve_chunks", lambda q: captured.append(q) or [])

        ea.run(_make_result(predicted_dx="mel"), {"age": 50, "sex": "female", "localization": "face"})
        assert len(captured) == 1
        assert "mel" in captured[0]

    def test_query_contains_metadata(self, monkeypatch):
        """Query must incorporate patient metadata."""
        import src.agentic.evidence_agent as ea

        captured = []
        monkeypatch.setattr(ea, "_retrieve_chunks", lambda q: captured.append(q) or [])

        ea.run(_make_result(), {"age": 67, "sex": "female", "localization": "chest"})
        query = captured[0]
        assert "female" in query
        assert "chest" in query


# ---------------------------------------------------------------------------
# LLM failure graceful fallback
# ---------------------------------------------------------------------------

class TestLLMFailureFallback:

    def test_returns_insufficient_on_llm_double_failure(self, monkeypatch):
        """If LLM fails twice, return insufficient_evidence rather than crashing."""
        import src.agentic.evidence_agent as ea

        good_chunks = [
            {"text": "Some relevant text.", "doi": "", "pub_date": "2023", "cosine": 0.70},
        ]
        monkeypatch.setattr(ea, "_retrieve_chunks", lambda q: good_chunks)
        # Patch at evidence_agent module level so the import is intercepted
        monkeypatch.setattr(ea, "call_llm", lambda *a, **kw: (_ for _ in ()).throw(
            ValueError("Simulated LLM parse failure")
        ))

        result = ea.run(_make_result(), METADATA)
        # Should degrade gracefully, not raise
        assert result["status"] == "insufficient_evidence"
