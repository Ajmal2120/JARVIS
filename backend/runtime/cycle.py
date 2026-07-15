"""
JARVIS 3.0 — ReasoningCycle.

A single complete thought in JARVIS 3.0 goes through:

    Observer -> Context -> Interpreter -> Inference -> Planner
                                                  -> (future Suggestion Engine)

Each full pass is a "Reasoning Cycle".  This module provides:

    - A factory for unique cycle identifiers (UUIDs).
    - A context-manager for test / explicit use that pairs a cycle id
      with begin/end lifecycle hooks.
    - An event dataclass that downstream layers publish to signal the
      end of a cycle (used by CognitiveLogger and the future Suggestion
      Engine).

Design goals:
    - Zero allocation at idle (the ContextManager creates nothing
      until enter() is called).
    - Thread-safe: cycle IDs are unique and never collide.
    - Minimal dependencies: only the standard library and runtime.events.

No LLM. No memory. No suggestions. Pure infrastructure.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from runtime.events import RuntimeEvent


logger = logging.getLogger("jarvis.runtime.cycle")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class ReasoningCycle:
    """Factory + lifecycle context manager for reasoning cycle IDs.

    Every reasoning pass through the cognitive pipeline receives a
    unique ``cycle_id``.  This class is the single source of truth for
    those identifiers.

    Usage in a pipeline:
        # The Coalescer assigns the cycle_id at flush time.
        with ReasoningCycle.begin() as cycle_id:
            # ... context update, inference, planner ...
            ReasoningCycle.publish_end(cycle_id)

    For test / manual use:
        cycle = ReasoningCycle()
        cycle_id = cycle.begin()
        try:
            ...
        finally:
            cycle.end()
    """

    _local = threading.local()
    _generator = staticmethod(uuid.uuid4)  # type: ignore[assignment]

    def __init__(self) -> None:
        self._cycle_id: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Context manager                                                    #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> str:
        """Begin a new cycle and return its cycle_id."""
        self._cycle_id = str(self._generator())
        return self._cycle_id

    def __exit__(self, *_exc_info: object) -> None:
        self._cycle_id = None

    # ------------------------------------------------------------------ #
    # Explicit lifecycle                                                 #
    # ------------------------------------------------------------------ #

    def begin(self) -> str:
        """Begin a new cycle.  Returns the assigned cycle_id."""
        self._cycle_id = str(self._generator())
        return self._cycle_id

    def end(self) -> None:
        """End the current cycle (release the id)."""
        self._cycle_id = None

    @property
    def current_id(self) -> Optional[str]:
        """The cycle_id of the currently-open cycle, if any."""
        return self._cycle_id

    # ------------------------------------------------------------------ #
    # Static helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def new_id() -> str:
        """Generate a fresh UUID-based cycle_id.

        Used by the EventCoalescer when it flushes a burst of Observer
        events.  The Coalescer calls this instead of instantiating the
        context manager — it is cheap and stateless.
        """
        return str(uuid.uuid4())

    @staticmethod
    def publish_end(cycle_id: str) -> None:
        """Publish a ReasoningCycleEnded event to the Runtime EventBus.

        Called by the last cognitive component (Planner, or later the
        Suggestion Engine) so a listener can see the full chain from
        start to finish for a single thought.
        """
        try:
            from runtime.runtime import get_runtime
            runtime = get_runtime()
        except RuntimeError:
            return
        try:
            runtime.event_bus.publish(
                ReasoningCycleEnded(cycle_id=cycle_id, timestamp=_now())
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("ReasoningCycle.publish_end failed: %s", exc)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReasoningCycleEnded(RuntimeEvent):
    """Published by the last pipeline component when a cycle is done.

    Subscribers (CognitiveLogger, Suggestion Engine) use this as a
    natural delimiter for one complete chain of inference.
    """

    type: str = "reasoning_cycle_ended"
    cycle_id: str = ""
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _now() -> float:
    import time as _t
    return _t.time()
