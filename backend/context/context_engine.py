"""
JARVIS 3.0 — Context Engine service.

The Context Engine subscribes to Observer events and maintains the
authoritative live context snapshot. It is the single source of truth
for "What is happening right now?"

The engine is purely event-driven. It holds no polling thread, makes
no OS calls, stores no history, and makes no decisions.

Responsibilities:
    - Subscribe to Observer events on the Runtime EventBus
    - Update the context snapshot from each event
    - Expose get_current_context() for any downstream component

Responsibilities it does NOT have:
    - It does not read the OS (Observer does that)
    - It does not call the LLM
    - It does not execute skills
    - It does not write to Memory
    - It does not infer activity ("user is coding", "user is browsing")
    - It does not keep history or deltas beyond the current snapshot

Integration:
    - Implements RuntimeService (name, start, stop)
    - Subscribes to Runtime.event_bus in start(), unsubscribes in stop()
    - Registered after Observer in main.py lifespan
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Callable

from runtime.runtime import Runtime, RuntimeState, get_runtime
from runtime.events import RuntimeEvent

from observer.events import (
    ApplicationChanged,
    IdleStateChanged,
    LockStateChanged,
    WindowChanged,
)

from .context import Context

logger = logging.getLogger("jarvis.context")


class ContextEngine:
    """Runtime service that maintains the live context snapshot."""

    name: str = "context_engine"

    def __init__(self) -> None:
        self._context = Context()
        self._lock = threading.Lock()
        self._unsubscribe: Callable[[], None] | None = None

    # ------------------------------------------------------------------ #
    # RuntimeService interface                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Subscribe to Observer events on the Runtime EventBus."""
        runtime = get_runtime()
        self._unsubscribe = runtime.event_bus.subscribe(self._on_event)
        logger.info("context_engine: started, subscribed to EventBus")

    def stop(self) -> None:
        """Unsubscribe from the EventBus."""
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        logger.info("context_engine: stopped")

    # ------------------------------------------------------------------ #
    # Event handling                                                       #
    # ------------------------------------------------------------------ #

    def _on_event(self, event: RuntimeEvent) -> None:
        """Process an event from the EventBus.

        Only Observer events are relevant. All others are ignored.
        """
        # Fast path: only events with a snapshot dict come from Observer.
        snapshot = getattr(event, "snapshot", None)
        if not isinstance(snapshot, dict):
            return

        self._update_context(snapshot, event)

    def _update_context(self, snapshot: dict, event: RuntimeEvent) -> None:
        """Merge an Observer snapshot into the live context.

        Only fields present in the snapshot are updated. Missing fields
        are left as-is so a transient read failure from the Observer
        doesn't blank out context fields that were previously known.
        """
        now = datetime.now()
        confidence = getattr(event, "confidence", 1.0)

        with self._lock:
            if snapshot.get("window_title") is not None:
                self._context.active_window = snapshot["window_title"]
            if snapshot.get("process_name") is not None:
                self._context.active_application = snapshot["process_name"]
            if snapshot.get("idle_seconds") is not None:
                self._context.idle_seconds = float(snapshot["idle_seconds"])

            # IdleStateChanged carries the Observer's authoritative
            # classification (above/below threshold). Use that, not a
            # raw > 0 check on idle_seconds.
            if isinstance(event, IdleStateChanged):
                self._context.is_idle = bool(event.is_idle)

            if snapshot.get("is_locked") is not None:
                self._context.is_locked = bool(snapshot["is_locked"])
            if snapshot.get("system_uptime_s") is not None:
                self._context.system_uptime_s = float(snapshot["system_uptime_s"])

            self._context.confidence = float(confidence)
            self._context.last_updated = now

            # Window or application change means the user just did
            # something — mark it as activity.
            if isinstance(event, (WindowChanged, ApplicationChanged)):
                self._context.last_activity_timestamp = now

    # ------------------------------------------------------------------ #
    # Public accessor                                                      #
    # ------------------------------------------------------------------ #

    def get_current_context(self) -> Context:
        """Return a copy of the current context snapshot.

        Returns a new Context instance so callers can't mutate internal
        state. Under the lock for thread safety.
        """
        with self._lock:
            return Context(
                active_window=self._context.active_window,
                active_application=self._context.active_application,
                last_activity_timestamp=self._context.last_activity_timestamp,
                is_idle=self._context.is_idle,
                idle_seconds=self._context.idle_seconds,
                system_uptime_s=self._context.system_uptime_s,
                is_locked=self._context.is_locked,
                confidence=self._context.confidence,
                last_updated=self._context.last_updated,
            )