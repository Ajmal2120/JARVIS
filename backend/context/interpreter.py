"""
JARVIS 3.0 — Context Interpreter.

The Context Interpreter derives *generic behavioral signals* from the
Observer event stream. It produces a ContextInterpretation that
downstream layers (Intent, Planner) consume.

What it answers:
    - "What level of activity is happening?"
    - "How stable is the current context?"
    - "How often is the user switching?"
    - "How long has the current focus persisted?"

What it does NOT answer:
    - "What is the user trying to do?"
    - "Is the user productive / distracted / browsing?"
    - "Are they in VS Code / Chrome / Photoshop?"
    - Anything app-specific, role-specific, or intent-specific.

Every signal here is a deterministic function of:
    - The current Context (a snapshot)
    - The stream of Observer events over time

No hardcoded application names. No classification. No LLM.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Deque

from runtime.runtime import Runtime, get_runtime
from runtime.events import RuntimeEvent

from observer.events import (
    ApplicationChanged,
    WindowChanged,
)
from .context import Context

logger = logging.getLogger("jarvis.context.interpreter")


class InteractionLevel(str, Enum):
    """How active the user currently is.

    Generic levels — they describe intensity, not productivity or
    activity type. All thresholds are deterministic.
    """

    IDLE = "idle"          # Locked or beyond the configured idle threshold
    LOW = "low"            # > 15s since last input but not yet idle
    MEDIUM = "medium"      # 2–15s since last input
    HIGH = "high"          # < 2s since last input


class StabilityLevel(str, Enum):
    """Coarse label for how long the current context has persisted."""

    FLUCTUATING = "fluctuating"  # < 10s since last change
    RECENT = "recent"            # 10–60s since last change
    STEADY = "steady"            # > 60s since last change


@dataclass(frozen=True)
class ContextInterpretation:
    """Immutable interpretation of the current context.

    All numeric fields use a common unit (seconds) where applicable.
    Designed to be the input for downstream layers (Intent, Planner).
    """

    interaction_level: InteractionLevel
    session_duration_s: float
    context_stability_s: float
    context_stability: StabilityLevel
    application_switch_frequency: float   # switches per minute
    current_focus_duration_s: float
    confidence: float
    interpreted_at: datetime


class ContextInterpreter:
    """Subscribes to Observer events; produces generic behavioral signals.

    The Interpreter is owned by the Context Engine. It runs in its own
    thread-safe state, listens directly to the Runtime EventBus, and
    exposes `get_current_interpretation()` for callers.

    Thread safety: all internal state is locked. `get_current_interpretation()`
    returns a fresh, immutable snapshot.
    """

    # Idle bucketing thresholds (seconds since last input)
    _IDLE_THRESHOLD_S = 60.0   # see also Observer's idle_threshold_s
    _LOW_THRESHOLD_S = 15.0
    _MEDIUM_THRESHOLD_S = 2.0

    # Stability labels
    _STEADY_S = 60.0
    _RECENT_S = 10.0

    # Application switch rate window
    _SWITCH_WINDOW_S = 300.0  # last 5 minutes

    def __init__(self, lookup_context: Callable[[], Context]) -> None:
        """Create the interpreter.

        `lookup_context` is a callable the Context Engine provides so the
        Interpreter can read the current Context snapshot. The Interpreter
        does NOT store the Context Engine reference — it gets fresh
        snapshots on demand.
        """
        self._lookup_context = lookup_context

        # --- Event stream state (the only "history" we keep) ---
        self._lock = threading.Lock()
        self._window_changes: Deque[float] = deque()  # timestamps
        self._app_changes: Deque[float] = deque()
        self._last_window_change: float = 0.0
        self._last_app_change: float = 0.0
        self._session_start_wall_time: datetime = datetime.now()
        self._session_start_uptime_s: float | None = None
        self._unsubscribe: Callable[[], None] | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        runtime = get_runtime()
        self._unsubscribe = runtime.event_bus.subscribe(self._on_event)

        # Seed the session-start uptime. If the system has been up, the
        # session started no earlier than now.
        ctx = self._lookup_context()
        self._session_start_uptime_s = ctx.system_uptime_s
        logger.info("interpreter: started")

    def stop(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        logger.info("interpreter: stopped")

    # ------------------------------------------------------------------ #
    # Event handling                                                       #
    # ------------------------------------------------------------------ #

    def _on_event(self, event: RuntimeEvent) -> None:
        # Only Observer events matter. Anything else ignored.
        snapshot = getattr(event, "snapshot", None)
        if not isinstance(snapshot, dict):
            return

        now = time.time()
        with self._lock:
            if isinstance(event, (WindowChanged, ApplicationChanged)):
                self._window_changes.append(now)
                if isinstance(event, ApplicationChanged):
                    self._app_changes.append(now)
                    self._last_app_change = now
                self._last_window_change = now

            self._trim_window(now)

    def _trim_window(self, now: float) -> None:
        """Drop timestamps older than the switch-window."""
        cutoff = now - self._SWITCH_WINDOW_S
        while self._window_changes and self._window_changes[0] < cutoff:
            self._window_changes.popleft()
        while self._app_changes and self._app_changes[0] < cutoff:
            self._app_changes.popleft()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_current_interpretation(self) -> ContextInterpretation:
        """Compute a fresh interpretation from current Context + event state."""
        ctx = self._lookup_context()
        now = time.time()

        with self._lock:
            # --- Session duration ---
            # If system uptime dropped, the system rebooted — restart the
            # session anchor so the count doesn't go negative.
            current_uptime = ctx.system_uptime_s
            if (
                self._session_start_uptime_s is not None
                and current_uptime < self._session_start_uptime_s
            ):
                self._session_start_uptime_s = current_uptime
            session_duration_s = max(
                0.0, current_uptime - (self._session_start_uptime_s or current_uptime)
            )

            # --- Interaction level ---
            idle_seconds = float(ctx.idle_seconds)
            is_locked = bool(ctx.is_locked)
            if is_locked or idle_seconds >= self._IDLE_THRESHOLD_S:
                level = InteractionLevel.IDLE
            elif idle_seconds >= self._LOW_THRESHOLD_S:
                level = InteractionLevel.LOW
            elif idle_seconds >= self._MEDIUM_THRESHOLD_S:
                level = InteractionLevel.MEDIUM
            else:
                level = InteractionLevel.HIGH

            # --- Context stability ---
            # Stability = time since the last WindowChanged/ApplicationChanged.
            # If we've never seen a change, treat as "just started now".
            last_change = max(self._last_window_change, self._last_app_change)
            if last_change == 0.0:
                context_stability_s = 0.0
            else:
                context_stability_s = max(0.0, now - last_change)

            if context_stability_s >= self._STEADY_S:
                stability = StabilityLevel.STEADY
            elif context_stability_s >= self._RECENT_S:
                stability = StabilityLevel.RECENT
            else:
                stability = StabilityLevel.FLUCTUATING

            # --- Application switch frequency (per minute) ---
            self._trim_window(now)
            # Switches per minute = (count / observed_seconds) * 60
            if self._app_changes and len(self._app_changes) >= 2:
                observed_s = max(1.0, self._app_changes[-1] - self._app_changes[0])
                switches_per_minute = (
                    (len(self._app_changes) - 1) / observed_s * 60.0
                )
            else:
                switches_per_minute = 0.0

            # --- Current focus duration (per process, not per window) ---
            if self._last_app_change == 0.0:
                current_focus_duration_s = 0.0
            else:
                current_focus_duration_s = max(0.0, now - self._last_app_change)

            confidence = float(ctx.confidence)

            interpreted_at = datetime.now()

        return ContextInterpretation(
            interaction_level=level,
            session_duration_s=session_duration_s,
            context_stability_s=context_stability_s,
            context_stability=stability,
            application_switch_frequency=switches_per_minute,
            current_focus_duration_s=current_focus_duration_s,
            confidence=confidence,
            interpreted_at=interpreted_at,
        )
