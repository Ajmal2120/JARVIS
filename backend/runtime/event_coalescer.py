"""
JARVIS 3.0 — Event Coalescer.

A single user action often produces a burst of Observer events:

    WindowChanged -> ApplicationChanged -> ...

The Coalescer sits between the Observer and downstream layers (Context
Engine, Inference Engine, Planner).  Within a configurable window
(default 150 ms) it:

    1. Buffers incoming Observer events.
    2. Merges their `snapshot` dicts into a single canonical snapshot.
    3. Records the most-recent concrete event type and its dedicated
       fields (e.g. ``is_idle``, ``process_name``).
    4. Assigns one ``cycle_id`` to the whole burst.
    5. Publishes ONE consolidated event downstream via the Runtime
       EventBus.

Downstream components that subscribe to *this consolidated event*
(class ``ConsolidatedSnapshot``) get exactly one reasoning cycle per
user action, no matter how many raw Observer events fired.

Cost: a single ``threading.Timer`` instance per flush.  Idle cost is
effectively zero (the timer thread sleeps).

Public API:
    coalescer = EventCoalescer(window_ms=150)
    coalescer.start()
    runtime.event_bus.subscribe(coalescer._on_event)
    coalescer.stop()
"""
from __future__ import annotations

import logging
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional

from runtime.events import RuntimeEvent
from runtime.runtime import get_runtime


logger = logging.getLogger("jarvis.runtime.coalescer")


# ---------------------------------------------------------------------------
# Consolidated event
# ---------------------------------------------------------------------------

#: Set of event types the Coalescer is willing to buffer.
DEFAULT_COALESCE_TYPES = frozenset({
    "window_changed",
    "application_changed",
    "idle_state_changed",
    "lock_state_changed",
})


@dataclass(frozen=True)
class ConsolidatedSnapshot(RuntimeEvent):
    """One merged Observer payload for a single user action.

    Downstream services that subscribe to ``runtime.event_bus`` and
    filter on ``type == "consolidated_snapshot"`` receive one of these
    per coalescing window.

    Fields:
        type:           Always "consolidated_snapshot".
        cycle_id:       Stable identifier for this reasoning cycle —
                        propagated into Context, Inference, Planner,
                        and CognitiveLogger.
        snapshot:       The merged snapshot dict (last-write-wins).
        primary_type:   The most recent concrete Observer event type
                        within the window ("window_changed" etc.).
        primary_event:  A reference to the last concrete event (kept
                        for callers that need its dedicated fields).
        event_count:    Number of raw events merged into this one.
        confidence:     Minimum confidence across the merged events.
        window_ms:      Actual configured window used.
        timestamp:      Unix epoch seconds when the flush happened.
    """

    type: str = "consolidated_snapshot"
    cycle_id: str = ""
    snapshot: Dict = None  # type: ignore[assignment]
    primary_type: str = ""
    primary_event: Optional[RuntimeEvent] = None
    event_count: int = 0
    confidence: float = 1.0
    window_ms: int = 150
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# EventCoalescer
# ---------------------------------------------------------------------------

class EventCoalescer:
    """Coalesce bursts of Observer events into single downstream events.

    Usage in ``main.py``::

        coalescer = EventCoalescer(window_ms=150)
        coalescer.start()
        rt.event_bus.subscribe(coalescer._on_event)
        ...
        coalescer.stop()

    Other services subscribe to ``runtime.event_bus`` but filter on
    ``type == "consolidated_snapshot"`` instead of the raw Observer
    event types.  They keep receiving a ``snapshot`` dict — they just
    receive one call per coalescing window instead of one per raw
    change.
    """

    def __init__(
        self,
        window_ms: int = 150,
        coalesce_types: Optional[frozenset] = None,
        max_buffer: int = 16,
    ) -> None:
        """Construct the coalescer.

        Args:
            window_ms:     Milliseconds to buffer events before flushing.
                           Tunable via config.  150 ms is a good default
                           that swallows the typical ApplicationChanged
                           / WindowChanged burst while adding barely-
                           noticeable latency.
            coalesce_types: Set of event types to buffer.  Default is
                           the four standard Observer types.
            max_buffer:    Safety upper bound.  When the buffer reaches
                           this size the coalescer flushes immediately,
                           so a runaway publisher cannot cause unbounded
                           memory growth.
        """
        if window_ms < 0:
            raise ValueError(f"window_ms must be >= 0, got {window_ms}")
        if max_buffer < 1:
            raise ValueError(f"max_buffer must be >= 1, got {max_buffer}")

        self.window_ms = int(window_ms)
        self.coalesce_types = (
            coalesce_types
            if coalesce_types is not None
            else DEFAULT_COALESCE_TYPES
        )
        self.max_buffer = max_buffer

        # Buffer + flush state.  Guarded by the lock.
        self._lock = threading.Lock()
        self._buffer: list[RuntimeEvent] = []
        self._snapshot: Dict[str, object] = {}
        self._primary: Optional[RuntimeEvent] = None
        self._current_cycle_id: Optional[str] = None
        self._current_window_id: int = 0  # tracks which window owns the cycle_id

        # Flush-time threading primitives.
        self._timer: Optional[threading.Timer] = None
        self._running = False
        self._flush_now = threading.Event()
        self._flush_complete = threading.Event()
        self._flush_complete.set()  # idle at startup

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._running = True
        logger.info("event_coalescer: started (window_ms=%d)", self.window_ms)

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._cancel_timer_locked()
        # Drain any pending events synchronously so callers don't lose them.
        self._flush(force=True)
        logger.info("event_coalescer: stopped")

    # ------------------------------------------------------------------ #
    # Event entrypoint (subscribed to the Runtime EventBus)               #
    # ------------------------------------------------------------------ #

    def _on_event(self, event: RuntimeEvent) -> None:
        """Filter, buffer, and schedule a flush."""
        event_type = getattr(event, "type", "")
        if event_type not in self.coalesce_types:
            # Non-coalescable events pass through unchanged by leaving
            # this callback alone — the Coalescer is one subscriber
            # among many, not a wrapper.
            return

        if not self._running:
            return

        with self._lock:
            # Start a fresh cycle on the first event of a window.
            if not self._buffer:
                self._current_cycle_id = str(uuid.uuid4())
                self._snapshot = {}
                self._primary = None

            self._buffer.append(event)
            self._current_window_id += 1

            # Merge snapshot (last-write-wins).
            snap = getattr(event, "snapshot", None)
            if isinstance(snap, dict):
                self._snapshot.update(snap)

            # Preserve the concrete payload (e.g. is_idle, process_name)
            # by remembering the most recent concrete event.
            self._primary = event

            # If we've hit max_buffer, force-flush immediately.
            if len(self._buffer) >= self.max_buffer:
                self._flush_locked(force=True)
            else:
                # Otherwise schedule a deferred flush.
                self._schedule_flush_locked()

    # ------------------------------------------------------------------ #
    # Internal flush                                                      #
    # ------------------------------------------------------------------ #

    def _schedule_flush_locked(self) -> None:
        """Arm the timer for the current window. Caller must hold lock."""
        self._cancel_timer_locked()
        if self.window_ms <= 0:
            # Flush on the next bit of free time, but never recurse.
            threading.Timer(0.001, self._flush).start()
            return
        t = threading.Timer(
            self.window_ms / 1000.0,
            self._flush,
        )
        t.daemon = True
        self._timer = t
        t.start()

    def _cancel_timer_locked(self) -> None:
        """Cancel any pending timer. Caller must hold lock."""
        if self._timer is not None:
            try:
                self._timer.cancel()
            except Exception:
                pass
            self._timer = None

    def _flush(self, force: bool = False) -> None:
        """Public flush entrypoint (used by Timer)."""
        with self._lock:
            if not self._running and not force:
                return
            self._flush_locked(force=force)

    def _flush_locked(self, force: bool = False) -> None:
        """Build the consolidated event and publish it.

        Caller must hold the lock (or guarantee no race).  We re-acquire
        the snapshot under the lock, build the event outside any lock,
        then publish.
        """
        # Cancel any pending timer — we're flushing now.
        self._cancel_timer_locked()

        if not self._buffer:
            return  # nothing to flush (e.g. called twice)

        # Snapshot the buffer state.
        buffer_count = len(self._buffer)
        merged_snapshot = deepcopy(self._snapshot)
        primary = self._primary
        cycle_id = self._current_cycle_id or str(uuid.uuid4())

        # Reset buffer (events are owned by the merged event now).
        self._buffer.clear()
        self._primary = None
        self._current_cycle_id = None
        self._snapshot = {}

        # Publish on the Runtime EventBus.  We avoid holding the lock
        # during publish() in case any subscriber is slow.
        self._publish_consolidated(
            cycle_id=cycle_id,
            merged_snapshot=merged_snapshot,
            primary=primary,
            event_count=buffer_count,
        )

    def _publish_consolidated(
        self,
        *,
        cycle_id: str,
        merged_snapshot: Dict,
        primary: Optional[RuntimeEvent],
        event_count: int,
    ) -> None:
        """Publish the consolidated event to the Runtime EventBus."""
        try:
            runtime = get_runtime()
        except RuntimeError:
            logger.debug("event_coalescer: no Runtime singleton; event dropped")
            return

        primary_type = getattr(primary, "type", "") if primary is not None else ""
        conf = (
            float(getattr(primary, "confidence", 1.0))
            if primary is not None else 1.0
        )

        try:
            runtime.event_bus.publish(ConsolidatedSnapshot(
                cycle_id=cycle_id,
                snapshot=merged_snapshot,
                primary_type=primary_type,
                primary_event=primary,
                event_count=event_count,
                confidence=conf,
                window_ms=self.window_ms,
                timestamp=_now_seconds(),
            ))
        except Exception as exc:  # noqa: BLE001
            logger.exception("event_coalescer: publish failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #

    @property
    def pending_count(self) -> int:
        """Current buffered event count. Useful for tests / metrics."""
        with self._lock:
            return len(self._buffer)

    def flush_now(self) -> None:
        """Synchronous flush (used by tests and graceful shutdown)."""
        self._flush(force=True)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _now_seconds() -> float:
    import time as _time
    return _time.time()
