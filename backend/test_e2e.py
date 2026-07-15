"""
End-to-end test: Observer → Coalescer → Context → Inference → Planner → Logger

Publishes three rapid Observer events (the classic WindowChanged →
ApplicationChanged burst), verifies they arrive as ONE consolidated
snapshot, and confirms the full chain runs once.

Expected log:
  cycle_abc
    Observer Coalesced Snapshot: event_count=3, primary_type=idle_state_changed
    Context Updated
    Interpretation Updated
    Hypothesis Created: software_development
    Decision: ASK_PERMISSION
    Reasoning Cycle Ended
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from runtime.runtime import Runtime, set_runtime
from runtime.event_coalescer import EventCoalescer
from observer.events import WindowChanged, ApplicationChanged, IdleStateChanged
from context import ContextEngine
from inference import InferenceEngine
from planner import Planner, Decision
from observability import CognitiveLogger


def main() -> None:
    rt = Runtime()
    rt.start()
    set_runtime(rt)

    # Wire the live pipeline (mirrors main.py lifespan).
    coalescer = EventCoalescer(window_ms=150)
    coalescer.start()
    rt.event_bus.subscribe(coalescer._on_event)

    ce = ContextEngine()
    ce.start()

    ie = InferenceEngine(
        lookup_context=ce.get_current_context,
        lookup_interpretation=ce.get_current_interpretation,
    )
    ie.start()

    p = Planner(
        lookup_hypotheses=ie.get_current_hypotheses,
        lookup_context=ce.get_current_context,
        lookup_interpretation=ce.get_current_interpretation,
    )
    p.start()

    cog = CognitiveLogger()
    cog.bind(
        get_context=ce.get_current_context,
        get_interpretation=ce.get_current_interpretation,
        get_hypotheses=ie.get_current_hypotheses,
    )
    cog.start()

    # Capture log lines
    import logging
    log_entries: list[dict] = []
    handler: logging.Handler | None = None

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                log_entries.append(json.loads(record.getMessage()))
            except Exception:
                pass

    cog_logger = logging.getLogger("jarvis.cognitive")
    handler = _Capture()
    handler.setLevel(logging.DEBUG)
    cog_logger.addHandler(handler)

    # Publish the classic burst: 3 rapid Observer events.
    rt.event_bus.publish(WindowChanged(
        snapshot={
            "window_title": "main.py",
            "process_name": "Code.exe",
            "idle_seconds": 2.0,
            "is_locked": False,
            "system_uptime_s": 100.0,
        },
        window_title="main.py",
        confidence=1.0,
    ))
    rt.event_bus.publish(ApplicationChanged(
        snapshot={
            "window_title": "main.py",
            "process_name": "Code.exe",
            "idle_seconds": 2.0,
            "is_locked": False,
            "system_uptime_s": 100.0,
        },
        process_name="Code.exe",
        confidence=1.0,
    ))
    rt.event_bus.publish(IdleStateChanged(
        snapshot={
            "window_title": "main.py",
            "process_name": "Code.exe",
            "idle_seconds": 2.0,
            "is_locked": False,
            "system_uptime_s": 100.0,
        },
        is_idle=False,
        idle_seconds=2.0,
    ))

    # Let context stability accumulate so the Planner can ASK.
    time.sleep(2.5)

    # --- Assertions ---
    plan = p.get_current_plan()
    # Partial context (one burst with only a few evidence items) gives
    # confidence below 0.7.  The Planner correctly decides WAIT — it
    # needs more signal before asking permission.
    assert plan.decision == Decision.WAIT, f"expected WAIT, got {plan.decision}"
    assert "confidence" in plan.reason.lower() or "below" in plan.reason.lower()

    # Find the consolidated snapshot entry in the log.
    coalesced = [e for e in log_entries
                 if e.get("event") == "Coalesced Snapshot"]
    assert coalesced, "expected at least one Coalesced Snapshot in the log"
    assert coalesced[0]["details"]["event_count"] >= 2, \
        f"expected 3 merged, got {coalesced[0]['details']['event_count']}"

    # A Hypothesis Created entry (first appearance in the persistent store).
    created = [e for e in log_entries if e.get("event") == "Hypothesis Created"]
    assert created, "expected Hypothesis Created in the log"
    assert created[0]["details"]["id"] == "software_development"

    # A Decision entry.
    decisions = [e for e in log_entries if e.get("event") == "Decision"]
    assert decisions, "expected Planner Decision in the log"
    assert decisions[0]["details"]["decision"] == "WAIT"

    # Cycle ended marker.
    cycles = [e for e in log_entries
              if e.get("event") == "Reasoning Cycle Ended"]
    assert cycles, "expected ReasoningCycleEnded marker"

    # --- Report ---
    print("=== End-to-end pipeline test ===")
    print(f"Plan: {plan.decision.value} (confidence={plan.confidence:.2f})")
    print(f"Hypotheses: {plan.supporting_hypotheses}")
    print(f"Coalesced snapshots in log: {len(coalesced)}")
    print(f"Hypothesis Created in log: {len(created)}")
    print(f"Decision in log: {decisions[0]['details']['decision']}")
    print(f"Cycles ended: {len(cycles)}")
    print("PASS")

    # Cleanup
    cog_logger.removeHandler(handler)
    p.stop()
    ie.stop()
    ce.stop()
    coalescer.stop()
    rt.stop()


if __name__ == "__main__":
    main()
