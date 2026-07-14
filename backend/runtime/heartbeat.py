"""
Heartbeat for the JARVIS 3.0 Runtime.

Responsibilities:
    - Drive the Scheduler once per second (when the Runtime is alive).
    - Emit a HeartbeatTick event on the EventBus so future services can
      react to the pulse without polling.
    - Stay cheap. While idle, the heartbeat must not pin a CPU core.
      We sleep via `Event.wait(timeout=...)` so the wake-up is interruptible
      and the thread can be joined cleanly on shutdown.

What the heartbeat does NOT do today:
    - No screen polling.
    - No window enumeration.
    - No memory I/O.
    - No LLM calls.
    - No skill execution.

It only logs at DEBUG and lets the Scheduler fan out to whatever tasks
have been registered. Currently the Scheduler is empty, so in practice
this thread sleeps 99.99% of the time.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger("jarvis.runtime.heartbeat")


class Heartbeat:
    """1 Hz pulse for the Runtime.

    The pulse interval is configurable for tests; production code uses
    `interval_s=1.0`. Sleep is interruptible so stop() returns within
    a fraction of the interval instead of waiting up to a full second.
    """

    def __init__(
        self,
        on_tick: Callable[[float], None],
        interval_s: float = 1.0,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self._on_tick = on_tick
        self._interval = float(interval_s)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._tick_count = 0

    def start(self) -> None:
        """Spawn the heartbeat thread. Idempotent."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="JarvisRuntimeHeartbeat",
            daemon=True,
        )
        self._thread.start()
        logger.debug("heartbeat: started (interval=%.3fs)", self._interval)

    def stop(self, timeout: float | None = None) -> None:
        """Signal the heartbeat to exit and join the thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.debug("heartbeat: stopped (ticks=%d)", self._tick_count)

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def _run(self) -> None:
        # First tick fires after one full interval — gives services
        # registered in `start()` a chance to attach before the first
        # callback runs.
        while not self._stop_event.wait(self._interval):
            self._tick_count += 1
            try:
                self._on_tick(time.time())
            except Exception as exc:  # pragma: no cover — defensive
                # The Runtime is responsible for handling tick errors
                # itself; we just don't want to die silently here.
                logger.exception("heartbeat tick raised: %s", exc)
