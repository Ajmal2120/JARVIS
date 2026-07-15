"""Verification tests for the Inference Engine."""
from __future__ import annotations

import time
from unittest.mock import patch

from context import (
    Context, ContextInterpretation, ContextEngine,
    InteractionLevel, StabilityLevel,
)
from datetime import datetime
from inference import InferenceEngine, InferenceRule, Hypothesis
from inference.rules.software_development import SoftwareDevelopmentRule
from observer import WindowChanged, ApplicationChanged
from runtime.runtime import Runtime


def main():
    # ==========================================================================
    # Setup: Runtime + Context Engine + Inference Engine
    # ==========================================================================
    rt = Runtime(heartbeat_interval_s=0.5)
    rt.start()
    import runtime.runtime as rt_mod
    rt_mod._runtime = rt

    ce = ContextEngine()
    ce.start()

    ie = InferenceEngine(
        lookup_context=ce.get_current_context,
        lookup_interpretation=ce.get_current_interpretation,
    )
    ie.start()

    # ==========================================================================
    # Test 1: Fresh state — no hypotheses
    # ==========================================================================
    hypotheses = ie.get_current_hypotheses()
    assert hypotheses == [], f"expected empty, got {hypotheses}"
    print("Test 1: Fresh state -> no hypotheses: PASS")

    # ==========================================================================
    # Test 2: Single WindowChanged — only process match + low switch rate.
    #         Confidence = 3/6 = 0.5 (above threshold 0.35).
    # ==========================================================================
    rt.event_bus.publish(WindowChanged(
        snapshot={
            "window_title": "foo", "process_name": "Code.exe",
            "idle_seconds": 5.0, "is_locked": False,
            "system_uptime_s": 100.0,
        },
        window_title="foo", confidence=1.0,
    ))
    time.sleep(0.05)
    hypotheses = ie.get_current_hypotheses()
    assert len(hypotheses) == 1, f"expected 1, got {hypotheses}"
    assert hypotheses[0].id == "software_development"
    assert hypotheses[0].confidence >= 0.35
    print(f"Test 2: Single WindowChanged -> partial hypothesis "
          f"(confidence={hypotheses[0].confidence:.2f}): PASS")

    # ==========================================================================
    # Test 3: Full dev context — all 5 signals
    # ==========================================================================
    fake_context = Context(
        active_application="Code.exe",
        active_window="main.py",
        idle_seconds=0.0,
        is_locked=False,
        system_uptime_s=200.0,
        confidence=1.0,
    )
    fake_interp = ContextInterpretation(
        interaction_level=InteractionLevel.HIGH,
        session_duration_s=60.0,
        context_stability_s=120.0,
        context_stability=StabilityLevel.STEADY,
        application_switch_frequency=0.5,
        current_focus_duration_s=120.0,
        confidence=1.0,
        interpreted_at=datetime.now(),
    )
    rule = SoftwareDevelopmentRule()
    h = rule.evaluate(fake_context, fake_interp)
    assert h is not None, "expected hypothesis"
    assert h.id == "software_development"
    assert h.confidence == 1.0, f"expected 1.0, got {h.confidence}"
    assert len(h.evidence) == 5, f"expected 5 evidence items, got {len(h.evidence)}"
    print(f"Test 3: Full dev context -> confidence={h.confidence}, "
          f"evidence={len(h.evidence)}: PASS")
    print(f"    Evidence: {h.evidence}")

    # ==========================================================================
    # Test 4: Partial evidence — process + stable + long focus = 4/6 = 0.667
    # ==========================================================================
    fake_interp2 = ContextInterpretation(
        interaction_level=InteractionLevel.LOW,
        session_duration_s=10.0,
        context_stability_s=120.0,
        context_stability=StabilityLevel.STEADY,
        application_switch_frequency=0.5,
        current_focus_duration_s=120.0,
        confidence=1.0,
        interpreted_at=datetime.now(),
    )
    h2 = rule.evaluate(fake_context, fake_interp2)
    assert h2 is not None
    assert h2.id == "software_development"
    expected = round(5.0 / 6.0, 4)
    assert h2.confidence == expected, f"expected {expected}, got {h2.confidence}"
    assert h2.confidence >= rule.MIN_CONFIDENCE
    assert len(h2.evidence) == 4  # process + stable + long focus + low switch
    print(f"Test 4: Partial (process+stable+focus+switch) -> "
          f"confidence={h2.confidence:.4f}: PASS")

    # ==========================================================================
    # Test 5: Confidence cap — no process match, max other signals = 4/6
    # ==========================================================================
    fake_ctx_no_app = Context(
        active_application="notepad.exe",
        idle_seconds=0.0, is_locked=False,
        system_uptime_s=100.0, confidence=1.0,
    )
    h3 = rule.evaluate(fake_ctx_no_app, fake_interp)
    assert h3 is not None
    expected3 = round(4.0 / 6.0, 4)
    assert h3.confidence == expected3, f"expected {expected3}, got {h3.confidence}"
    print(f"Test 5: No process match, max signals -> confidence={h3.confidence:.4f}: PASS")

    # ==========================================================================
    # Test 6: Below MIN_CONFIDENCE -> None
    # ==========================================================================
    fake_interp4 = ContextInterpretation(
        interaction_level=InteractionLevel.IDLE,
        session_duration_s=1.0,
        context_stability_s=1.0,
        context_stability=StabilityLevel.FLUCTUATING,
        application_switch_frequency=20.0,
        current_focus_duration_s=1.0,
        confidence=1.0,
        interpreted_at=datetime.now(),
    )
    h4 = rule.evaluate(fake_context, fake_interp4)
    assert h4 is None, f"expected None, got {h4}"
    print("Test 6: Low evidence -> None: PASS")

    # ==========================================================================
    # Test 7: InferenceEngine.evaluate() returns ranked list
    # ==========================================================================
    # Persistence is active (start() was called in setup).  Reset the store
    # so this test starts from a clean slate — it's testing the ranking
    # logic, not cross-test accumulation.
    ie.reset_persistent()
    with patch.object(ie, "_lookup_context", return_value=fake_context), \
         patch.object(ie, "_lookup_interpretation", return_value=fake_interp):
        ranked = ie.evaluate()
    assert len(ranked) == 1
    assert ranked[0].id == "software_development"
    assert ranked[0].confidence == 1.0
    print("Test 7: Engine.evaluate() returns ranked hypothesis: PASS")

    # ==========================================================================
    # Test 8: Engine merges + sorts multiple rules
    # ==========================================================================
    class FakeRuleA(InferenceRule):
        name = "fake_a"
        def evaluate(self, ctx, interp):
            return Hypothesis(id="fake_a", confidence=0.6,
                              evidence=["A1", "A2"], inferred_at=datetime.now())

    class FakeRuleB(InferenceRule):
        name = "fake_b"
        def evaluate(self, ctx, interp):
            return Hypothesis(id="fake_b", confidence=0.9,
                              evidence=["B1"], inferred_at=datetime.now())

    ie2 = InferenceEngine(
        lookup_context=lambda: fake_context,
        lookup_interpretation=lambda: fake_interp,
        rules=(FakeRuleA, FakeRuleB),
    )
    ie2.start()
    ranked2 = ie2.evaluate()
    assert len(ranked2) == 2
    assert ranked2[0].id == "fake_b" and ranked2[0].confidence == 0.9
    assert ranked2[1].id == "fake_a" and ranked2[1].confidence == 0.6
    print("Test 8: Engine merges + sorts 2 rules (0.9, 0.6): PASS")

    # ==========================================================================
    # Test 9: Deduplication — same id, keep highest confidence
    # ==========================================================================
    class DupRule(InferenceRule):
        name = "dup"
        def evaluate(self, ctx, interp):
            return Hypothesis(id="dup", confidence=0.3, evidence=["dup"], inferred_at=datetime.now())

    ie3 = InferenceEngine(
        lookup_context=lambda: fake_context,
        lookup_interpretation=lambda: fake_interp,
        rules=(DupRule, DupRule),
    )
    ie3.start()
    ranked3 = ie3.evaluate()
    dup_ids = [h.id for h in ranked3]
    assert dup_ids.count("dup") == 1, (
        f"expected 1 dup, got {dup_ids.count('dup')}"
    )
    print("Test 9: Deduplication by id (1 dup kept): PASS")

    # ==========================================================================
    # Test 10: Rule failure is isolated
    # ==========================================================================
    class BrokenRule(InferenceRule):
        name = "broken"
        def evaluate(self, ctx, interp):
            raise RuntimeError("boom")

    class GoodRule(InferenceRule):
        name = "good"
        def evaluate(self, ctx, interp):
            return Hypothesis(id="good", confidence=0.7, evidence=["ok"], inferred_at=datetime.now())

    ie4 = InferenceEngine(
        lookup_context=lambda: fake_context,
        lookup_interpretation=lambda: fake_interp,
        rules=(BrokenRule, GoodRule),
    )
    ie4.start()
    ranked4 = ie4.evaluate()
    assert len(ranked4) == 1
    assert ranked4[0].id == "good"
    assert ie4.last_evaluated_at is not None
    print("Test 11: last_evaluated_at set: PASS")

    # ==========================================================================
    # Test 12: Confidence flows Context -> Interpreter -> Inference
    # ==========================================================================
    rt.event_bus.publish(ApplicationChanged(
        snapshot={
            "window_title": "foo", "process_name": "Code.exe",
            "idle_seconds": 0.0, "is_locked": False,
            "system_uptime_s": 300.0,
        },
        process_name="Code.exe", confidence=0.8,
    ))
    time.sleep(0.1)
    interp = ce.get_current_interpretation()
    assert interp.confidence == 0.8
    print("Test 12: Confidence flows through chain: PASS")

    # ==========================================================================
    # Cleanup
    # ==========================================================================
    ie.stop()
    ie2.stop()
    ie3.stop()
    ie4.stop()
    ce.stop()
    rt.stop()
    print("\n=== All inference tests passed ===")


if __name__ == "__main__":
    main()
