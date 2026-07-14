"""
JARVIS 3.0 — Observer service.

The Observer is the first Runtime service. Its job is narrow:

    1. Read OS state (window, process, idle, lock, uptime) at a fixed
       polling interval.
    2. Compare the new state to the previous state.
    3. Publish an event only when something changed.

What the Observer does NOT do:
    - It does not interpret observations.
    - It does not classify activities.
    - It does not call the LLM.
    - It does not write to Memory.
    - It does not make suggestions.
    - It does not block the Runtime or the heartbeat thread.

Integration with Runtime:
    - Implements RuntimeService (name, start, stop).
    - Uses Runtime.set_state(OBSERVING) when active, IDLE when paused.
    - Publishes events through Runtime.event_bus.
    - Runs its own daemon thread. start() and stop() are synchronous
      and block until the thread is fully up or down.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from runtime.runtime import Runtime, RuntimeState, get_runtime
from runtime.events import RuntimeEvent

from .events import (
    ApplicationChanged,
    IdleStateChanged,
    LockStateChanged,
    ObserverEvent,
    WindowChanged,
)
from .models import SystemSnapshot
from .sources import (
    get_active_process_name,
    get_active_window_title,
    get_idle_seconds,
    get_locked,
    get_system_uptime_s,
    read_lock_state,
    reset_lock_state,
)

logger = logging.getLogger("jarvis.observer")


class Observer:
    """Runtime service that observes OS state and publishes change events.

    Polling interval is configurable. Default is 1.0s — fast enough to
    catch window switches, slow enough to stay invisible on CPU.
    """

    name: str = "observer"

    def __init__(
        self,
        poll_interval_s: float = 1.0,
        idle_threshold_s: float = 60.0,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be > 0")
        self._poll_interval = poll_interval_s
        self._idle_threshold = idle_threshold_s
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._snapshot = SystemSnapshot()
        self._was_idle: bool = False

    # ------------------------------------------------------------------ #
    # RuntimeService interface                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Begin observing. Spawns a daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.debug("observer: already running")
            return

        # Seed the snapshot so we don't emit spurious events on the first
        # real comparison.
        reset_lock_state()
        self._snapshot = self._read_snapshot()
        self._was_idle = (self._snapshot.idle_seconds >= self._idle_threshold)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="Observer",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "observer: started (poll=%.1fs, idle_threshold=%.0fs)",
            self._poll_interval, self._idle_threshold,
        )

    def stop(self) -> None:
        """Stop observing. Blocks until the thread exits."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._poll_interval + 0.5)
            self._thread = None
        logger.info("observer: stopped")

    # ------------------------------------------------------------------ #
    # Observation loop                                                     #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        runtime = get_runtime()
        if runtime.is_running:
            runtime.set_state(RuntimeState.OBSERVING)

        while not self._stop_event.wait(self._poll_interval):
            try:
                self._poll(runtime)
            except Exception as exc:
                logger.exception("observer: poll error: %s", exc)

        # Transition back to IDLE if we're the reason it was OBSERVING.
        if runtime.is_running and runtime.state is RuntimeState.OBSERVING:
            runtime.set_state(RuntimeState.IDLE)

    def _poll(self, runtime: Runtime) -> None:
        """Read current state, compare, publish changes."""
        current = self._read_snapshot()
        events = self._diff(self._snapshot, current)
        self._snapshot = current

        if events:
            if runtime.is_running:
                runtime.set_state(RuntimeState.OBSERVING)
            for event in events:
                self._publish(event, runtime)
        elif runtime.state is RuntimeState.OBSERVING:
            runtime.set_state(RuntimeState.IDLE)

    # ------------------------------------------------------------------ #
    # State reading                                                       #
    # ------------------------------------------------------------------ #

    def _read_snapshot(self) -> SystemSnapshot:
        """Read the current OS state. Failed reads become None/0 defaults."""
        return SystemSnapshot(
            window_title=get_active_window_title() or "",
            process_name=get_active_process_name() or "",
            idle_seconds=get_idle_seconds() or 0.0,
            is_locked=get_locked() or False,
            system_uptime_s=get_system_uptime_s() or 0.0,
        )

    # ------------------------------------------------------------------ #
    # Change detection                                                     #
    # ------------------------------------------------------------------ #

    def _diff(
        self, old: SystemSnapshot, new: SystemSnapshot
    ) -> list[ObserverEvent]:
        """Compare snapshots and return a list of change events.

        Returns an empty list when nothing changed. This is the Observer's
        guarantee: it emits events only on change.
        """
        events: list[ObserverEvent] = []
        snapshot_dict = new.to_dict()

        # Window title
        if new.window_title != old.window_title:
            events.append(WindowChanged(
                snapshot=snapshot_dict,
                window_title=new.window_title,
            ))

        # Process name
        if new.process_name != old.process_name:
            events.append(ApplicationChanged(
                snapshot=snapshot_dict,
                process_name=new.process_name,
            ))

        # Lock state — handled via the lock source's own change detection
        lock_now, lock_changed = read_lock_state()
        if lock_now is not None and lock_changed:
            events.append(LockStateChanged(
                snapshot=snapshot_dict,
                is_locked=lock_now,
            ))
        elif lock_now is None and old.is_locked is not None:
            # Source failed — treat as no change to avoid noise
            pass

        # Idle state — transition only, not continuous
        is_now_idle = new.idle_seconds >= self._idle_threshold
        if is_now_idle != self._was_idle:
            events.append(IdleStateChanged(
                snapshot=snapshot_dict,
                is_idle=is_now_idle,
                idle_seconds=new.idle_seconds,
            ))
            self._was_idle = is_now_idle

        return events

    # ------------------------------------------------------------------ #
    # Publishing                                                           #
    # ------------------------------------------------------------------ #

    def _publish(self, event: ObserverEvent, runtime: Runtime) -> None:
        """Publish an event on the Runtime's EventBus."""
        runtime.event_bus.publish(event)
        logger.debug("observer: published %s", event.type)
