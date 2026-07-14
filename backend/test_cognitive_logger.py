"""
Tests for the CognitiveLogger.

Covers:
  - Direct method calls write structured JSON entries
  - EventBus subscription logs Observer events
  - Bound service callbacks log downstream state
  - Future extension points work (Planner, Suggestion, Memory)
  - Disabled mode is a complete no-op (no file created, no I/O)
  - Duplicate-handler guard works
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Make sure we have a clean log path for each run.
LOG_PATH = Path(__file__).parent / "logs" / "cognitive.log"


def cleanup():
    """Remove the log file before each test."""
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    if LOG_PATH.parent.exists() and not any(LOG_PATH.parent.iterdir()):
        LOG_PATH.parent.rmdir()


def reset():
    """Close the handler and remove the log file."""
    import observability.cognitive_logger as m
    if m._cognitive_handler is not None:
        try:
            m._cognitive_handler.flush()
            m._cognitive_handler.close()
        except Exception:
            pass
        m._cognitive_handler = None
    cleanup()
    # Remove the handler from the logger too.
    cog_logger = logging.getLogger("jarvis.cognitive")
    cog_logger.handlers.clear()


def read_log_entries():
    """Read all JSON entries from the log file."""
    if not LOG_PATH.exists():
        return []
    out = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_direct_method_calls():
    """Direct calls to log methods write structured entries."""
    reset()
    import observability.cognitive_logger as m
    m.ENABLED = True

    # Ensure handler is attached (simulates what start() does).
    m._ensure_handler()

    cog = m.CognitiveLogger()
    cog.window_changed("test.py", "Code.exe", confidence=1.0)
    cog.application_changed("Code.exe", "test.py")
    cog.idle_state_changed(False, 2.0)
    cog.lock_state_changed(False)
    cog.context_updated(active_application="Code.exe", active_window="test.py")
    cog.interpretation_updated(interaction_level="high", context_stability="steady")
    cog.hypothesis_generated("software_development", 0.82, ["a", "b"])
    cog.decision("ACT", "high conf")
    cog.suggestion_event("accepted", "s1")
    cog.memory_event("stored", "k1")

    entries = read_log_entries()
    components = [e["component"] for e in entries]
    assert "Observer" in components, f"missing Observer entries: {components}"
    assert "Context" in components, f"missing Context entries: {components}"
    assert "Interpreter" in components, f"missing Interpreter entries: {components}"
    assert "Inference" in components, f"missing Inference entries: {components}"
    assert "Planner" in components, f"missing Planner entries: {components}"
    assert "Suggestion" in components, f"missing Suggestion entries: {components}"
    assert "Memory" in components, f"missing Memory entries: {components}"
    print("Test 1: Direct method calls -> OK")


def test_eventbus_subscription():
    """Observer events on the EventBus are routed to the right logger methods."""
    reset()
    import observability.cognitive_logger as m
    m.ENABLED = True

    from runtime.runtime import Runtime, set_runtime
    from observer.events import (
        WindowChanged, ApplicationChanged, IdleStateChanged, LockStateChanged,
    )

    rt = Runtime()
    rt.start()
    set_runtime(rt)

    cog = m.CognitiveLogger()
    cog.start()
    time.sleep(0.2)

    rt.event_bus.publish(WindowChanged(
        snapshot={"window_title": "x", "process_name": "y"},
        window_title="x", confidence=1.0,
    ))
    rt.event_bus.publish(ApplicationChanged(
        snapshot={"window_title": "x", "process_name": "y"},
        process_name="y", confidence=1.0,
    ))
    rt.event_bus.publish(IdleStateChanged(
        snapshot={"window_title": "x", "process_name": "y", "idle_seconds": 60.0},
        is_idle=True, idle_seconds=60.0,
    ))
    rt.event_bus.publish(LockStateChanged(
        snapshot={"window_title": "x", "process_name": "y", "is_locked": True},
        is_locked=True,
    ))
    time.sleep(0.3)

    entries = read_log_entries()
    events = [e["event"] for e in entries]
    assert "WindowChanged" in events, f"missing WindowChanged: {events}"
    assert "ApplicationChanged" in events, f"missing ApplicationChanged: {events}"
    assert "IdleStateChanged" in events, f"missing IdleStateChanged: {events}"
    assert "LockStateChanged" in events, f"missing LockStateChanged: {events}"

    cog.stop()
    rt.stop()
    print("Test 2: EventBus subscription -> OK")


def test_bind_log_downstream():
    """Bound callbacks enrich Observer events with downstream state."""
    reset()
    import observability.cognitive_logger as m
    m.ENABLED = True

    from runtime.runtime import Runtime, set_runtime
    from observer.events import WindowChanged
    from context import Context, ContextInterpretation, InteractionLevel, StabilityLevel
    from inference.models import Hypothesis
    from datetime import datetime

    rt = Runtime()
    rt.start()
    set_runtime(rt)

    def mk_ctx():
        return Context(
            active_window="x", active_application="y",
            is_idle=False, idle_seconds=2.0, is_locked=False, system_uptime_s=100.0,
            confidence=1.0, last_updated=datetime.now(),
        )

    def mk_interp():
        return ContextInterpretation(
            interaction_level=InteractionLevel.HIGH,
            session_duration_s=600.0, context_stability_s=120.0,
            context_stability=StabilityLevel.STEADY,
            application_switch_frequency=0.5, current_focus_duration_s=120.0,
            confidence=1.0,
            interpreted_at=datetime.now(),
        )

    def mk_hyp():
        return [Hypothesis(
            id="software_development", confidence=0.9,
            evidence=["x", "y"], inferred_at=datetime.now(),
        )]

    cog = m.CognitiveLogger()
    cog.bind(get_context=mk_ctx, get_interpretation=mk_interp, get_hypotheses=mk_hyp)
    cog.start()
    time.sleep(0.2)

    rt.event_bus.publish(WindowChanged(
        snapshot={"window_title": "x", "process_name": "y"},
        window_title="x", confidence=1.0,
    ))
    time.sleep(0.3)

    entries = read_log_entries()
    components = [e["component"] for e in entries]
    # All four components should appear for one Observer event.
    for expected in ("Observer", "Context", "Interpreter", "Inference"):
        assert expected in components, f"missing {expected}: {components}"

    cog.stop()
    rt.stop()
    print("Test 3: Bound callbacks enrich Observer events -> OK")


def test_disabled_mode():
    """Disabled mode creates no file and runs no I/O."""
    reset()
    import observability.cognitive_logger as m
    m.ENABLED = False

    cog = m.CognitiveLogger()
    cog.window_changed("x", "y")
    cog.context_updated()
    cog.hypothesis_generated("h", 1.0, [])

    assert not LOG_PATH.exists(), "log file should NOT exist when disabled"
    print("Test 4: Disabled mode -> OK")


def test_no_duplicate_handlers():
    """Multiple handler creations don't add duplicates."""
    reset()
    import observability.cognitive_logger as m
    m.ENABLED = True

    # Manually ensure handler multiple times
    m._ensure_handler()
    m._ensure_handler()
    m._ensure_handler()

    cog_logger = logging.getLogger("jarvis.cognitive")
    file_handlers = [h for h in cog_logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1, f"expected 1 file handler, got {len(file_handlers)}"
    print("Test 5: No duplicate handlers -> OK")


if __name__ == "__main__":
    test_direct_method_calls()
    test_eventbus_subscription()
    test_bind_log_downstream()
    test_disabled_mode()
    test_no_duplicate_handlers()
    print()
    print("=== All CognitiveLogger tests passed ===")
