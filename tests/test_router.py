"""
Unit tests for the confidence router.

Critical: the router is RULE-BASED and must be LLM-free.
Tests verify exact tier boundaries from config.py:
  LOW    < 0.50
  MEDIUM 0.50 – 0.85
  HIGH   > 0.85

Every boundary is tested both sides to catch off-by-one errors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import config
from src.agentic.confidence_router import Tier, route, route_description


# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------

class TestRouterBoundaries:

    def test_below_low_boundary(self):
        assert route(0.0) == Tier.LOW

    def test_just_below_low_max(self):
        assert route(0.4999) == Tier.LOW

    def test_at_low_max_is_medium(self):
        """0.50 is the start of MEDIUM (inclusive lower bound)."""
        assert route(0.50) == Tier.MEDIUM

    def test_just_above_low_max(self):
        assert route(0.5001) == Tier.MEDIUM

    def test_mid_medium(self):
        assert route(0.70) == Tier.MEDIUM

    def test_at_high_min_is_medium(self):
        """0.85 is still MEDIUM (inclusive upper bound of MEDIUM)."""
        assert route(0.85) == Tier.MEDIUM

    def test_just_above_high_min(self):
        assert route(0.8501) == Tier.HIGH

    def test_clearly_high(self):
        assert route(0.95) == Tier.HIGH

    def test_max_confidence_is_high(self):
        assert route(1.0) == Tier.HIGH


# ---------------------------------------------------------------------------
# Config consistency
# ---------------------------------------------------------------------------

class TestRouterConfig:

    def test_low_max_matches_config(self):
        """Boundary values must come from config, never be hardcoded."""
        assert route(config.ROUTER_LOW_MAX - 0.0001) == Tier.LOW
        assert route(config.ROUTER_LOW_MAX) == Tier.MEDIUM

    def test_high_min_matches_config(self):
        assert route(config.ROUTER_HIGH_MIN) == Tier.MEDIUM
        assert route(config.ROUTER_HIGH_MIN + 0.0001) == Tier.HIGH

    def test_guard_min_is_high_threshold(self):
        """Guard gate must equal the HIGH threshold — not a separate value."""
        assert config.GUARD_MIN_CONFIDENCE == config.ROUTER_HIGH_MIN


# ---------------------------------------------------------------------------
# No LLM calls
# ---------------------------------------------------------------------------

class TestRouterIsLLMFree:

    def test_route_makes_no_network_calls(self, monkeypatch):
        """Patch the LLM client to raise if called — router must never touch it."""
        import src.agentic.anthropic_client as client
        monkeypatch.setattr(client, "call_llm", lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("route() must NOT call the LLM")
        ))
        # These must all complete without triggering the patch
        assert route(0.3) == Tier.LOW
        assert route(0.6) == Tier.MEDIUM
        assert route(0.9) == Tier.HIGH


# ---------------------------------------------------------------------------
# Route description
# ---------------------------------------------------------------------------

class TestRouteDescription:

    def _mock_result(self, confidence, malignant_prob, label="benign", dx="nv"):
        class R:
            pass
        r = R()
        r.confidence      = confidence
        r.malignant_prob  = malignant_prob
        r.predicted_label = label
        r.predicted_dx    = dx
        return r

    def test_low_description_contains_reject(self):
        r = self._mock_result(0.3, 0.3, "benign", "nv")
        desc = route_description(Tier.LOW, r)
        assert "reject" in desc.lower()

    def test_medium_description_contains_escalat(self):
        r = self._mock_result(0.7, 0.4, "benign", "bkl")
        desc = route_description(Tier.MEDIUM, r)
        assert "escalat" in desc.lower()

    def test_high_description_contains_report(self):
        r = self._mock_result(0.92, 0.08, "benign", "nv")
        desc = route_description(Tier.HIGH, r)
        assert "report" in desc.lower()
