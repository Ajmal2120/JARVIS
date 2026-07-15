"""
Tests for JARVIS 3.0 Planner.

Covers:
    - Import check
    - Default construction, start/stop lifecycle
    - Default plan: WAIT when not wired
    - No hypotheses -> IGNORE
    - Low confidence -> WAIT
    - Strong confidence -> ASK_PERMISSION
    - User idle -> IGNORE
    - User locked -> IGNORE
    - Rapid app switching -> WAIT
    - Rule merge: highest weight*confidence wins
    - Tie-break: ASK_PERMISSION > WAIT > IGNORE
    - get_current_plan() returns a frozen copy
    - Thread-safe under concurrent evaluation
    - EventBus subscription triggers evaluation
"""
from __future__ import annotations

import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from planner import (
    Planner,
    Plan,
    Decision,
    PlanningRule,
    PlanningVerdict,
    DEFAULT_CONFIDENCE_THRESHOLD,
)
from inference.models import Hypothesis
from context import Context, ContextInterpretation, InteractionLevel, StabilityLevel
from datetime import datetime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_hypothesis(id_, confidence, evidence):
    return Hypothesis(id=id_, confidence=confidence, evidence=evidence,
                      inferred_at=datetime.now())

def _make_context(
    is_idle=False, is_locked=False,
    application_switch_frequency=0.0, current_focus_duration_s=120.0,
    context_stability_s=120.0,
):
    return ContextInterpretation(
        interaction_level=InteractionLevel.HIGH,
        session_duration_s=600.0,
        context_stability_s=context_stability_s,
        context_stability=StabilityLevel.STEADY,
        application_switch_frequency=application_switch_frequency,
        current_focus_duration_s=current_focus_duration_s,
        confidence=1.0,
        interpreted_at=datetime.now(),
    )

def _make_context_snap(is_idle=False, is_locked=False):
    return Context(
        active_window="main.py", active_application="Code.exe",
        last_activity_timestamp=datetime.now(),
        is_idle=is_idle, idle_seconds=0.0,
        is_locked=is_locked, system_uptime_s=100.0,
        confidence=1.0, last_updated=datetime.now(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_default_plan_is_wait():
    """Planner emits WAIT when not wired to lookups."""
    planner = Planner()
    plan = planner.evaluate()
    assert plan.decision == Decision.WAIT, f"expected WAIT, got {plan.decision}"
    print("Test 1: Default plan is WAIT -> OK")


def test_no_hypotheses_ignores():
    """With no hypotheses, the Planner emits IGNORE."""
    planner = Planner(
        lookup_hypotheses=lambda: [],
        lookup_context=_make_context_snap,
        lookup_interpretation=_make_context,
    )
    plan = planner.evaluate()
    assert plan.decision == Decision.IGNORE, f"expected IGNORE, got {plan.decision}"
    print("Test 2: No hypotheses -> IGNORE -> OK")


def test_low_confidence_waits():
    """Top hypothesis below 0.7 -> WAIT."""
    planner = Planner(
        lookup_hypotheses=lambda: [_make_hypothesis("dev", 0.4, ["a"])],
        lookup_context=_make_context_snap,
        lookup_interpretation=_make_context,
    )
    plan = planner.evaluate()
    assert plan.decision == Decision.WAIT, f"expected WAIT, got {plan.decision}"
    assert "0.40" in plan.reason
    print("Test 3: Low confidence -> WAIT -> OK")


def test_strong_confidence_asks_permission():
    """Top hypothesis >= 0.7 AND stable context -> ASK_PERMISSION."""
    planner = Planner(
        lookup_hypotheses=lambda: [_make_hypothesis("software_development", 0.82,
                                                     ["VS Code active", "High interaction"])],
        lookup_context=_make_context_snap,
        lookup_interpretation=_make_context,
    )
    plan = planner.evaluate()
    assert plan.decision == Decision.ASK_PERMISSION, f"expected ASK_PERMISSION, got {plan.decision}"
    assert plan.confidence == 0.82
    assert "software_development" in plan.reason
    assert "software_development" in plan.supporting_hypotheses
    print("Test 4: Strong confidence -> ASK_PERMISSION -> OK")


def test_user_idle_ignores():
    """User is idle -> IGNORE regardless of hypothesis strength."""
    planner = Planner(
        lookup_hypotheses=lambda: [_make_hypothesis("dev", 0.9, ["a"])],
        lookup_context=lambda: _make_context_snap(is_idle=True),
        lookup_interpretation=_make_context,
    )
    plan = planner.evaluate()
    assert plan.decision == Decision.IGNORE, f"expected IGNORE, got {plan.decision}"
    assert "idle" in plan.reason.lower()
    print("Test 5: User idle -> IGNORE -> OK")


def test_user_locked_ignores():
    """Screen locked -> IGNORE."""
    planner = Planner(
        lookup_hypotheses=lambda: [_make_hypothesis("dev", 0.9, ["a"])],
        lookup_context=lambda: _make_context_snap(is_locked=True),
        lookup_interpretation=_make_context,
    )
    plan = planner.evaluate()
    assert plan.decision == Decision.IGNORE, f"expected IGNORE, got {plan.decision}"
    assert "lock" in plan.reason.lower()
    print("Test 6: User locked -> IGNORE -> OK")


def test_app_flickering_waits():
    """Rapid app switching -> WAIT."""
    planner = Planner(
        lookup_hypotheses=lambda: [_make_hypothesis("dev", 0.85, ["a"])],
        lookup_context=_make_context_snap,
        lookup_interpretation=lambda: _make_context(application_switch_frequency=8.0),
    )
    plan = planner.evaluate()
    assert plan.decision == Decision.WAIT, f"expected WAIT, got {plan.decision}"
    print("Test 7: App flickering -> WAIT -> OK")


def test_merge_highest_weight_star_confidence():
    """The verdict with the highest weight × confidence wins."""

    # Two rules fire: one WAIT with high weight, one IGNORE with low weight.
    # The WAIT should win because 3.0 * 0.9 > 2.0 * 1.0.
    wait_verdict = PlanningVerdict(decision=Decision.WAIT, reason="W",
                                   confidence=0.9, weight=3.0)
    ignore_verdict = PlanningVerdict(decision=Decision.IGNORE, reason="I",
                                     confidence=1.0, weight=2.0)
    verdicts = [(wait_verdict, "rule1"), (ignore_verdict, "rule2")]
    hypotheses = [_make_hypothesis("h1", 0.9, ["x"])]

    plan = Planner._merge_verdicts(verdicts, hypotheses)
    assert plan.decision == Decision.WAIT, f"expected WAIT, got {plan.decision}"
    print("Test 8: Highest weight×confidence wins -> OK")


def test_tie_break_ask_permission():
    """Equal score: ASK_PERMISSION wins via priority."""

    ap = PlanningVerdict(decision=Decision.ASK_PERMISSION, reason="A",
                         confidence=0.5, weight=1.0)
    wait = PlanningVerdict(decision=Decision.WAIT, reason="W",
                           confidence=0.5, weight=1.0)
    verdicts = [(ap, "rule1"), (wait, "rule2")]
    hypotheses = [_make_hypothesis("h1", 0.8, ["x"])]

    plan = Planner._merge_verdicts(verdicts, hypotheses)
    assert plan.decision == Decision.ASK_PERMISSION
    print("Test 9: Tie -> ASK_PERMISSION wins -> OK")


def test_get_current_plan_returns_copy():
    """get_current_plan() returns a fresh frozen Plan (no mutation risk)."""
    planner = Planner(
        lookup_hypotheses=lambda: [_make_hypothesis("dev", 0.9, ["a"])],
        lookup_context=_make_context_snap,
        lookup_interpretation=_make_context,
    )
    planner.evaluate()
    plan1 = planner.get_current_plan()
    plan2 = planner.get_current_plan()
    assert plan1 is not plan2, "should return different Plan instances"
    assert plan1.decision == plan2.decision
    assert plan1.confidence == plan2.confidence
    assert plan1.plan_id == plan2.plan_id
    print("Test 10: get_current_plan returns copy -> OK")


def test_thread_safety():
    """Concurrent evaluate() calls do not corrupt state."""
    planner = Planner(
        lookup_hypotheses=lambda: [_make_hypothesis("dev", 0.9, ["a"])],
        lookup_context=_make_context_snap,
        lookup_interpretation=_make_context,
    )
    errors = []

    def worker():
        try:
            for _ in range(50):
                planner.evaluate()
                planner.get_current_plan()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread errors: {errors}"
    plan = planner.get_current_plan()
    assert plan.decision in (Decision.ASK_PERMISSION, Decision.WAIT, Decision.IGNORE)
    print("Test 11: Thread safety -> OK")


def test_eventbus_subscription():
    """Planner subscribes to EventBus and re-evaluates on events."""
    from runtime.runtime import Runtime, set_runtime
    from inference.engine import HypothesisGenerated

    rt = Runtime()
    rt.start()
    set_runtime(rt)

    planner = Planner(
        lookup_hypotheses=lambda: [_make_hypothesis("dev", 0.9, ["x"])],
        lookup_context=_make_context_snap,
        lookup_interpretation=_make_context,
    )
    planner.start()
    time.sleep(0.1)

    # Explicit eval with strong hypothesis -> ASK_PERMISSION.
    planner.evaluate()
    plan0 = planner.get_current_plan()
    assert plan0.decision == Decision.ASK_PERMISSION, f"expected ASK, got {plan0.decision}"

    # Fire a HypothesisGenerated event with low confidence -> re-eval -> WAIT
    planner._lookup_hypotheses = lambda: [_make_hypothesis("dev", 0.4, ["x"])]
    rt.event_bus.publish(HypothesisGenerated(
        cycle_id="cycle_1",
        ranked=(),
        top=None,
        timestamp=time.time(),
    ))
    time.sleep(0.2)

    plan1 = planner.get_current_plan()
    assert plan1.decision == Decision.WAIT, f"expected WAIT, got {plan1.decision}"

    planner.stop()
    rt.stop()
    print("Test 12: EventBus subscription -> OK")


def test_plan_idempotent_same_cycle():
    """Same cycle_id twice -> only evaluate once."""
    from runtime.runtime import Runtime, set_runtime

    rt = Runtime()
    rt.start()
    set_runtime(rt)

    count = [0]
    orig = Planner.evaluate

    def counting_evaluate(self):
        count[0] += 1
        return orig(self)

    Planner.evaluate = counting_evaluate  # type: ignore[assignment]

    planner = Planner(
        lookup_hypotheses=lambda: [_make_hypothesis("dev", 0.9, ["x"])],
        lookup_context=_make_context_snap,
        lookup_interpretation=_make_context,
    )
    planner.start()
    time.sleep(0.1)

    evt = type("E", (), {"type": "hypothesis_generated",
                          "cycle_id": "cycle_A",
                          "snapshot": {"window_title": "x"}})()
    rt.event_bus.publish(evt)
    rt.event_bus.publish(evt)
    time.sleep(0.2)

    # Should evaluate exactly once despite two publishes.
    assert count[0] == 1, f"expected 1 eval, got {count[0]}"

    Planner.evaluate = orig  # type: ignore[assignment]
    planner.stop()
    rt.stop()
    print("Test 13: Same cycle idempotent -> OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_default_plan_is_wait()
    test_no_hypotheses_ignores()
    test_low_confidence_waits()
    test_strong_confidence_asks_permission()
    test_user_idle_ignores()
    test_user_locked_ignores()
    test_app_flickering_waits()
    test_merge_highest_weight_star_confidence()
    test_tie_break_ask_permission()
    test_get_current_plan_returns_copy()
    test_thread_safety()
    test_eventbus_subscription()
    test_plan_idempotent_same_cycle()
    print()
    print("=== All Planner tests passed ===")
