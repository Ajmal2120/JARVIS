"""
Tests for JARVIS 3.0 ReasoningCycle.

Covers:
  - Factory creates unique IDs
  - Context manager produces a cycle_id
  - Static helpers
  - ReasoningCycleEnded event is published
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from runtime.cycle import ReasoningCycle, ReasoningCycleEnded


def test_factory_creates_unique_ids():
    ids = {ReasoningCycle.new_id() for _ in range(100)}
    assert len(ids) == 100, "IDs should be unique"


def test_context_manager():
    collected = []

    with ReasoningCycle() as cycle_id:
        collected.append(cycle_id)
        assert len(cycle_id) == 36  # UUID format

    assert len(collected) == 1


def test_explicit_begin_end():
    cycle = ReasoningCycle()
    assert cycle.current_id is None
    cid = cycle.begin()
    assert cycle.current_id == cid
    cycle.end()
    assert cycle.current_id is None


def test_cycle_ended_event():
    """ReasoningCycleEnded is published to the EventBus."""
    from runtime.runtime import Runtime, set_runtime

    rt = Runtime()
    rt.start()
    set_runtime(rt)

    received = []

    def on_event(event):
        received.append(event)

    rt.event_bus.subscribe(on_event)

    cycle_id = ReasoningCycle.new_id()
    ReasoningCycle.publish_end(cycle_id)
    time.sleep(0.05)

    assert any(
        isinstance(e, ReasoningCycleEnded) and e.cycle_id == cycle_id
        for e in received
    ), f"expected ReasoningCycleEnded event, got {received}"

    rt.stop()
    print("Cycle test: OK")


if __name__ == "__main__":
    test_factory_creates_unique_ids()
    test_context_manager()
    test_explicit_begin_end()
    test_cycle_ended_event()
    print("=== All cycle tests passed ===")
