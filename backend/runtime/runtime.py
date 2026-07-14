"""
JARVIS 3.0 — Runtime kernel.

The Runtime is the outermost lifecycle owner of the Operating Intelligence.
It boots with the backend process, runs a 1 Hz heartbeat, holds the
authoritative RuntimeState, and owns the registry of background services.

The Runtime is deliberately inert in this phase. It does not:
    - Observe windows
    - Read the screen
    - Call the LLM
    - Execute skills
    - Access memory
    - Make suggestions

Those will be added in subsequent phases, each as a service registered
with this Runtime. The Reactive Engine (voice, router, intent, TTS,
FastAPI endpoints) is unaware of the Runtime and continues to work as
before.

Lifecycle:
    Runtime()          -> constructed (state=BOOTING, not running)
    .start()           -> heartbeat begins, state->IDLE, events flow
    .register_service  -> a future Observer / Context / Planner plugs in
    .set_state(...)    -> a service announces it's working (OBSERVING etc.)
    .stop()            -> state->SHUTTING_DOWN, services drained, joined
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Iterable, Protocol, runtime_checkable

from .events import (
    EventBus,
    HeartbeatTick,
    ServiceFailed,
    ServiceRegistered,
    ServiceUnregistered,
    StateChange,
)
from .heartbeat import Heartbeat
from .scheduler import Scheduler
from .state import RuntimeState

logger = logging.getLogger("jarvis.runtime")


@runtime_checkable
class RuntimeService(Protocol):
    """Protocol for services managed by the Runtime.

    Services in later phases (Observer, Context Engine, Planner, Suggestion
    Engine) will implement this. `start` and `stop` are called by the
    Runtime in registration order, reverse-registration order respectively.
    A service may optionally expose a `name` attribute; if not, the
    Runtime uses the class name.
    """

    name: str

    def start(self) -> None: ...
    def stop(self) -> None: ...


# A module-level singleton accessor. The lifespan in main.py instantiates
# exactly one Runtime and stores it here. Tests can override by passing
# their own instance to `set_runtime`.
_runtime_lock = threading.Lock()
_runtime: "Runtime | None" = None


class Runtime:
    """The JARVIS 3.0 Runtime kernel."""

    def __init__(self, heartbeat_interval_s: float = 1.0) -> None:
        self._state: RuntimeState = RuntimeState.BOOTING
        self._state_lock = threading.Lock()
        self._services: dict[str, RuntimeService] = {}
        self._services_lock = threading.Lock()
        self._start_time: float = 0.0
        self._stopped: bool = False
        self._event_bus = EventBus()
        self._scheduler = Scheduler()
        self._heartbeat = Heartbeat(
            on_tick=self._on_heartbeat,
            interval_s=max(0.01, heartbeat_interval_s),
        )

    # ------------------------------------------------------------------ #
    # Public properties                                                    #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    @property
    def uptime_s(self) -> float:
        if not self._start_time:
            return 0.0
        return time.time() - self._start_time

    @property
    def is_running(self) -> bool:
        return not self._stopped and self._state is not RuntimeState.SHUTTING_DOWN

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Begin the heartbeat and transition to IDLE.

        Safe to call once. Subsequent calls are no-ops so a careless
        double-start from a misbehaving boot path can't wedge the kernel.
        """
        if self._stopped:
            raise RuntimeError("Runtime has been stopped; cannot restart")
        if self._start_time:
            logger.debug("runtime.start() called twice — ignoring")
            return

        logger.info("JARVIS 3.0 Runtime: starting")
        self._start_time = time.time()
        # Start the heartbeat first so services that depend on it can
        # immediately schedule work.
        self._heartbeat.start()
        self._set_state(RuntimeState.IDLE)
        logger.info(
            "JARVIS 3.0 Runtime: online (state=%s, uptime=%.1fs)",
            self._state.value, self.uptime_s,
        )

    def stop(self, timeout: float | None = 2.0) -> None:
        """Tear the kernel down.

        Stops the heartbeat, drains registered services in reverse
        registration order, closes the event bus, and transitions to
        SHUTTING_DOWN. Idempotent.
        """
        if self._stopped:
            return

        logger.info("JARVIS 3.0 Runtime: shutting down")
        self._set_state(RuntimeState.SHUTTING_DOWN)
        self._heartbeat.stop(timeout=timeout)

        # Drain services in reverse-registration order. Errors in one
        # service must not stop the rest from being stopped.
        with self._services_lock:
            service_items = list(reversed(list(self._services.items())))
            self._services.clear()

        for name, svc in service_items:
            try:
                svc.stop()
            except Exception as exc:
                logger.exception("service %s failed during stop: %s", name, exc)
                self._event_bus.publish(ServiceFailed(
                    service_name=name, phase="stop",
                    error=str(exc), timestamp=time.time(),
                ))

        self._scheduler.clear()
        self._event_bus.close()
        self._stopped = True
        logger.info("JARVIS 3.0 Runtime: stopped")

    # ------------------------------------------------------------------ #
    # Heartbeat                                                            #
    # ------------------------------------------------------------------ #

    def heartbeat(self) -> None:
        """Run one heartbeat cycle on demand.

        The background Heartbeat thread calls this once per second.
        Tests and admin endpoints can call it directly. Cheap: it just
        drives the scheduler and publishes a tick event.
        """
        self._on_heartbeat(time.time())

    def _on_heartbeat(self, now: float) -> None:
        if self._stopped:
            return
        # Drive registered runtime services' scheduled tasks first.
        self._scheduler.tick(now)
        # Then publish to anyone listening on the bus.
        self._event_bus.publish(HeartbeatTick(timestamp=now))
        # DEBUG-only noise so the heartbeat doesn't spam INFO logs once
        # a second. Enable with `logging.getLogger("jarvis.runtime").setLevel(logging.DEBUG)`.
        logger.debug(
            "heartbeat: state=%s uptime=%.1fs ticks=%d services=%d",
            self._state.value, self.uptime_s,
            self._heartbeat.tick_count, len(self._services),
        )

    # ------------------------------------------------------------------ #
    # State transitions                                                    #
    # ------------------------------------------------------------------ #

    def set_state(self, new_state: RuntimeState) -> None:
        """Public state-transition entry point for services."""
        self._set_state(new_state)

    def _set_state(self, new_state: RuntimeState) -> None:
        with self._state_lock:
            if new_state is self._state:
                return
            previous = self._state
            self._state = new_state
        self._event_bus.publish(StateChange(
            previous=previous.value,
            current=new_state.value,
            timestamp=time.time(),
        ))
        logger.debug("runtime: %s -> %s", previous.value, new_state.value)

    # ------------------------------------------------------------------ #
    # Service registry                                                     #
    # ------------------------------------------------------------------ #

    def register_service(self, service: RuntimeService) -> None:
        """Attach a background service to the Runtime.

        If the Runtime is already running, the service is started
        immediately so it can begin work without waiting for a restart.
        """
        name = self._service_name(service)
        with self._services_lock:
            if name in self._services:
                raise ValueError(f"service already registered: {name}")
            self._services[name] = service

        logger.info("runtime: registered service %s", name)
        self._event_bus.publish(ServiceRegistered(
            service_name=name, timestamp=time.time(),
        ))

        # If we've already booted, start the service immediately so the
        # caller doesn't have to know whether start() has been called.
        if self.is_running:
            try:
                service.start()
            except Exception as exc:
                logger.exception("service %s failed during start: %s", name, exc)
                self._event_bus.publish(ServiceFailed(
                    service_name=name, phase="start",
                    error=str(exc), timestamp=time.time(),
                ))
                # Roll back the registration — a failed-to-start service
                # shouldn't sit in the registry pretending to be alive.
                with self._services_lock:
                    self._services.pop(name, None)

    def unregister_service(self, name: str) -> bool:
        """Detach a service and stop it. Returns True if it existed."""
        with self._services_lock:
            svc = self._services.pop(name, None)
        if svc is None:
            return False
        try:
            svc.stop()
        except Exception as exc:
            logger.exception("service %s failed during unregister: %s", name, exc)
            self._event_bus.publish(ServiceFailed(
                service_name=name, phase="stop",
                error=str(exc), timestamp=time.time(),
            ))
        self._event_bus.publish(ServiceUnregistered(
            service_name=name, timestamp=time.time(),
        ))
        logger.info("runtime: unregistered service %s", name)
        return True

    def service_names(self) -> list[str]:
        with self._services_lock:
            return list(self._services.keys())

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _service_name(service: RuntimeService) -> str:
        name = getattr(service, "name", None)
        if isinstance(name, str) and name:
            return name
        return type(service).__name__


# ---------------------------------------------------------------------------
# Process-wide accessor
# ---------------------------------------------------------------------------

def get_runtime() -> Runtime:
    """Return the singleton Runtime, creating one if necessary.

    The lifespan in main.py is responsible for calling `start()` and
    `stop()` on it. Calling `get_runtime()` from other modules must not
    have side effects beyond lazy creation.
    """
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = Runtime()
    return _runtime


def set_runtime(runtime: Runtime | None) -> None:
    """Install or clear the singleton. Used by tests and lifespan teardown."""
    global _runtime
    with _runtime_lock:
        _runtime = runtime
