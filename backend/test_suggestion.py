"""
Unit tests for JARVIS 3.0 Suggestion Engine.

Covers:
   1. WAIT decision → no suggestion
   2. IGNORE decision → no suggestion
   3. ASK_PERMISSION + dev hypothesis → suggestion created
   4. accept() returns Suggestion with PreparedRequest
   5. decline() clears pending
   6. expire() clears pending after TTL
   7. only one pending at a time
   8. provider failure does not crash engine
   9. lifecycle events published (Created, Asked, Accepted, Declined)
  10. get_pending_suggestion returns defensive copy
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent))

from runtime.runtime import Runtime, set_runtime
from suggestion import (
    SuggestionEngine,
    SuggestionStatus,
    GeneralDevelopmentProvider,
    PreparedRequest,
    Suggestion,
)
from suggestion.providers import SuggestionProvider
from planner import Planner, Decision, Plan
from inference.models import Hypothesis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rt():
    """Create a Runtime and register it as the singleton."""
    rt = Runtime()
    rt.start()
    set_runtime(rt)
    return rt


def _fake_context():
    return MagicMock(
        active_window="main.py",
        active_application="Code.exe",
        is_idle=False,
        is_locked=False,
        confidence=1.0,
    )


def _fake_interpretation():
    return MagicMock(
        interaction_level="high",
        context_stability="steady",
        context_stability_s=120.0,
        application_switch_frequency=0.5,
        current_focus_duration_s=120.0,
    )


def _dev_hypothesis(confidence: float = 0.9):
    return Hypothesis(
        id="software_development",
        confidence=confidence,
        evidence=["VS Code active", "High interaction", "Stable context"],
    )


def _fake_plan(decision):
    return Plan(
        decision=decision,
        reason=f"Test: {decision.value}",
        confidence=0.9,
        supporting_hypotheses=("software_development",),
    )


def _publish_decision(rt, decision, cycle_id="cycle_1", plan=None):
    """Publish a PlanProduced-style decision event."""
    from planner.planner import PlanProduced

    rt.event_bus.publish(PlanProduced(
        cycle_id=cycle_id,
        decision=decision.value if isinstance(decision, Decision) else decision,
        reason="test",
        confidence=0.9,
        hypothesis_ids=("software_development",),
        timestamp=time.time(),
    ))


def _engine_with_lookups(rt):
    """Create a SuggestionEngine wired to fake lookups."""
    engine = SuggestionEngine(
        lookup_context=_fake_context,
        lookup_interpretation=_fake_interpretation,
        lookup_hypotheses=lambda: [_dev_hypothesis()],
        lookup_plan=lambda: _fake_plan(Decision.ASK_PERMISSION),
    )
    engine.start()
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_wait_decision_no_suggestion():
    rt = _make_rt()
    try:
        engine = _engine_with_lookups(rt)
        _publish_decision(rt, Decision.WAIT)
        time.sleep(0.05)
        assert engine.get_pending_suggestion() is None
        engine.stop()
        print("Test 1: WAIT -> no suggestion: PASS")
    finally:
        rt.stop()


def test_ignore_decision_no_suggestion():
    rt = _make_rt()
    try:
        engine = _engine_with_lookups(rt)
        _publish_decision(rt, Decision.IGNORE)
        time.sleep(0.05)
        assert engine.get_pending_suggestion() is None
        engine.stop()
        print("Test 2: IGNORE -> no suggestion: PASS")
    finally:
        rt.stop()


def test_ask_permission_creates_suggestion():
    rt = _make_rt()
    try:
        engine = _engine_with_lookups(rt)
        _publish_decision(rt, Decision.ASK_PERMISSION)
        time.sleep(0.05)
        suggestion = engine.get_pending_suggestion()
        assert suggestion is not None
        assert suggestion.status == SuggestionStatus.PENDING
        assert suggestion.provider == "general_development"
        assert "Code.exe" in suggestion.message
        assert suggestion.prepared_request is not None
        engine.stop()
        print("Test 3: ASK_PERMISSION -> suggestion created: PASS")
    finally:
        rt.stop()


def test_accept_returns_prepared_request():
    rt = _make_rt()
    try:
        engine = _engine_with_lookups(rt)
        _publish_decision(rt, Decision.ASK_PERMISSION)
        time.sleep(0.05)
        accepted = engine.accept()
        assert accepted is not None
        assert accepted.status == SuggestionStatus.ACCEPTED
        assert isinstance(accepted.prepared_request, PreparedRequest)
        assert accepted.prepared_request.request_type == "development_context_assist"
        assert engine.get_pending_suggestion() is None
        engine.stop()
        print("Test 4: accept() returns PreparedRequest: PASS")
    finally:
        rt.stop()


def test_decline_clears_pending():
    rt = _make_rt()
    try:
        engine = _engine_with_lookups(rt)
        _publish_decision(rt, Decision.ASK_PERMISSION)
        time.sleep(0.05)
        assert engine.has_pending
        engine.decline()
        assert not engine.has_pending
        assert engine.get_pending_suggestion() is None
        engine.stop()
        print("Test 5: decline() clears pending: PASS")
    finally:
        rt.stop()


def test_expire_clears_after_ttl():
    rt = _make_rt()
    try:
        engine = SuggestionEngine(
            lookup_context=_fake_context,
            lookup_interpretation=_fake_interpretation,
            lookup_hypotheses=lambda: [_dev_hypothesis()],
            lookup_plan=lambda: _fake_plan(Decision.ASK_PERMISSION),
            ttl_s=0.1,
        )
        engine.start()
        _publish_decision(rt, Decision.ASK_PERMISSION)
        time.sleep(0.05)
        assert engine.has_pending
        time.sleep(0.15)
        engine.expire()
        assert not engine.has_pending
        engine.stop()
        print("Test 6: expire() clears after TTL: PASS")
    finally:
        rt.stop()


def test_only_one_pending_at_a_time():
    rt = _make_rt()
    try:
        engine = _engine_with_lookups(rt)
        _publish_decision(rt, Decision.ASK_PERMISSION)
        time.sleep(0.05)
        assert engine.has_pending
        s1 = engine.get_pending_suggestion()
        cid1 = s1.cycle_id
        # Fire another ASK_PERMISSION — should not replace.
        _publish_decision(rt, Decision.ASK_PERMISSION, cycle_id="cycle_2")
        time.sleep(0.05)
        s2 = engine.get_pending_suggestion()
        assert s2 is not None
        assert s2.cycle_id == cid1  # first one kept
        assert s2.id == s1.id
        engine.stop()
        print("Test 7: only one pending at a time: PASS")
    finally:
        rt.stop()


def test_provider_failure_does_not_crash():
    rt = _make_rt()
    try:
        class BrokenProvider(SuggestionProvider):
            name = "broken"
            def generate(self, ctx, interp, hyp, plan):
                raise RuntimeError("boom")

        engine = SuggestionEngine(
            lookup_context=_fake_context,
            lookup_interpretation=_fake_interpretation,
            lookup_hypotheses=lambda: [_dev_hypothesis()],
            lookup_plan=lambda: _fake_plan(Decision.ASK_PERMISSION),
            provider_classes=(BrokenProvider, GeneralDevelopmentProvider),
        )
        engine.start()
        _publish_decision(rt, Decision.ASK_PERMISSION)
        time.sleep(0.05)
        # Broken provider fails, GeneralDevelopmentProvider succeeds.
        assert engine.has_pending
        assert engine.get_pending_suggestion().provider == "general_development"
        engine.stop()
        print("Test 8: provider failure -> next provider used: PASS")
    finally:
        rt.stop()


def test_lifecycle_events_published():
    rt = _make_rt()
    try:
        engine = _engine_with_lookups(rt)
        received = []

        def on_event(event):
            received.append(event)

        rt.event_bus.subscribe(on_event)

        _publish_decision(rt, Decision.ASK_PERMISSION)
        time.sleep(0.05)
        suggestion = engine.get_pending_suggestion()
        assert suggestion is not None
        sid = suggestion.id

        # Ask
        engine.ask()
        time.sleep(0.05)

        # Accept
        engine.accept()
        time.sleep(0.05)

        types = [e.type for e in received]
        assert "suggestion_created" in types
        assert "suggestion_asked" in types
        assert "suggestion_accepted" in types

        engine.stop()
        print("Test 9: lifecycle events published: PASS")
    finally:
        rt.stop()


def test_get_pending_returns_copy():
    rt = _make_rt()
    try:
        engine = _engine_with_lookups(rt)
        _publish_decision(rt, Decision.ASK_PERMISSION)
        time.sleep(0.05)
        s1 = engine.get_pending_suggestion()
        s2 = engine.get_pending_suggestion()
        assert s1 is not s2  # defensive copy
        assert s1.id == s2.id
        engine.stop()
        print("Test 10: get_pending returns defensive copy: PASS")
    finally:
        rt.stop()


def test_no_dev_hypothesis_no_suggestion():
    rt = _make_rt()
    try:
        engine = SuggestionEngine(
            lookup_context=_fake_context,
            lookup_interpretation=_fake_interpretation,
            # No software_development hypothesis
            lookup_hypotheses=lambda: [Hypothesis(id="browsing", confidence=0.8, evidence=["Chrome"])],
            lookup_plan=lambda: _fake_plan(Decision.ASK_PERMISSION),
        )
        engine.start()
        _publish_decision(rt, Decision.ASK_PERMISSION)
        time.sleep(0.05)
        assert engine.get_pending_suggestion() is None
        engine.stop()
        print("Test 11: non-dev hypothesis -> no suggestion: PASS")
    finally:
        rt.stop()


if __name__ == "__main__":
    test_wait_decision_no_suggestion()
    test_ignore_decision_no_suggestion()
    test_ask_permission_creates_suggestion()
    test_accept_returns_prepared_request()
    test_decline_clears_pending()
    test_expire_clears_after_ttl()
    test_only_one_pending_at_a_time()
    test_provider_failure_does_not_crash()
    test_lifecycle_events_published()
    test_get_pending_returns_copy()
    test_no_dev_hypothesis_no_suggestion()
    print()
    print("=== All suggestion tests passed ===")
