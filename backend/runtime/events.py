"""
Event bus for the JARVIS 3.0 Runtime.

The bus is intentionally tiny. Future Operating Intelligence services
(Observer, Context Engine, Planner, Suggestion Engine) will subscribe to
these events to coordinate without touching each other directly.

Design rules:
    - Events are plain dataclasses, not strings. New subscribers should
      `isinstance` them, not string-match.
    - The bus is in-process and synchronous from the caller's perspective.
      Subscriber callbacks run on the thread that called `publish()`.
      Asynchrony belongs in the services themselves.
    - Thread-safe: a lock protects all mutations of the subscriber list.
      Any number of threads may call `publish()` concurrently.
    - Subscribers must not raise; exceptions are swallowed and logged so
      a single misbehaving subscriber can't crash the kernel.
    - Publish is fire-and-forget. There is no replay, no persistence.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable
from weakref import WeakSet

logger = logging.getLogger("jarvis.runtime.events")


@dataclass(frozen=True)
class RuntimeEvent:
    """Base class for all runtime events.

    Concrete events are dataclasses that subclass `RuntimeEvent` and add
    their own fields. Keeping the base class empty (apart from `type`)
    means subscribers can switch on type() without surprises.
    """

    type: str = "base"


@dataclass(frozen=True)
class StateChange(RuntimeEvent):
    """Emitted whenever RuntimeState transitions.

    `previous` is None for the very first transition out of BOOTING.
    """

    type: str = "state_change"
    previous: str | None = None
    current: str = "IDLE"
    timestamp: float = 0.0


@dataclass(frozen=True)
class HeartbeatTick(RuntimeEvent):
    """Emitted once per heartbeat cycle while the Runtime is alive.

    Carries no payload by default — subscribers are free to inspect
    Runtime.state if they need it. Kept lean so the bus doesn't allocate
    a heavy object every second.
    """

    type: str = "heartbeat"
    timestamp: float = 0.0


@dataclass(frozen=True)
class ServiceRegistered(RuntimeEvent):
    """Emitted when a service is added via Runtime.register_service."""

    type: str = "service_registered"
    service_name: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True)
class ServiceUnregistered(RuntimeEvent):
    """Emitted when a service is removed via Runtime.unregister_service."""

    type: str = "service_unregistered"
    service_name: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True)
class ServiceFailed(RuntimeEvent):
    """Emitted when a registered service raises during start() or stop()."""

    type: str = "service_failed"
    service_name: str = ""
    phase: str = ""  # "start" or "stop"
    error: str = ""
    timestamp: float = 0.0


Subscriber = Callable[[RuntimeEvent], None]


class EventBus:
    """Synchronous in-process pub/sub for RuntimeEvents.

    Thread-safe: all mutations of the subscriber list are guarded by a
    lock. Any number of threads may call `publish()` concurrently.

    Callbacks run on the thread that called `publish()`. They must be
    fast; a slow subscriber blocks that thread. Asynchrony is the
    subscriber's responsibility, not the bus's.
    """

    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []
        self._closed = False
        self._lock = threading.Lock()

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        """Register a callback. Returns an unsubscribe function."""
        with self._lock:
            if self._closed:
                raise RuntimeError("EventBus is closed; cannot subscribe")
            self._subscribers.append(callback)

        def _unsub() -> None:
            with self._lock:
                try:
                    self._subscribers.remove(callback)
                except ValueError:
                    pass

        return _unsub

    def publish(self, event: RuntimeEvent) -> None:
        """Fan an event out to all subscribers.

        The subscriber list is snapshotted under the lock, then callbacks
        run outside it so a slow subscriber doesn't block new
        subscriptions. Exceptions in any subscriber are caught and logged.
        """
        with self._lock:
            if self._closed:
                return
            subscribers = list(self._subscribers)

        for cb in subscribers:
            try:
                cb(event)
            except Exception as exc:  # pragma: no cover — defensive
                logger.exception(
                    "Event subscriber raised on %s: %s", event.type, exc
                )

    def close(self) -> None:
        """Drop all subscribers. Idempotent."""
        with self._lock:
            self._closed = True
            self._subscribers.clear()


# A module-level weak registry of "interested" observers — currently
# unused, but kept as a future hook for OS-level event sinks that want
# to attach without an explicit subscription (e.g. a debug HUD).
_observers: "WeakSet[Any]" = WeakSet()


def register_observer(observer: Any) -> None:
    """Optional: register a passive observer for future telemetry."""
    _observers.add(observer)


def observers() -> Iterable[Any]:
    """Iterate over currently registered observers (weakly held)."""
    return tuple(_observers)

