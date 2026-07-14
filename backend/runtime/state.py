"""
Runtime lifecycle states for JARVIS 3.0.

The enum is the single source of truth for what the kernel is doing right
now. State transitions are explicit (see `runtime.Runtime.set_state`) so
the Operating Intelligence services can subscribe and react.

States, in lifecycle order:
    BOOTING         — process is coming up, services not yet registered
    IDLE            — alive, no work in flight
    OBSERVING       — an Observer service is actively gathering context
    PROCESSING      — a Planner / Suggestion service is reasoning
    SPEAKING        — TTS is playing back to the user
    SHUTTING_DOWN   — stop() has been called, services are being drained

These are coarse-grained kernel states. Individual services are free to
have their own internal substates — they do not belong here.
"""
from __future__ import annotations

from enum import Enum


class RuntimeState(str, Enum):
    """High-level lifecycle states of the JARVIS 3.0 Runtime."""

    BOOTING = "BOOTING"
    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    PROCESSING = "PROCESSING"
    SPEAKING = "SPEAKING"
    SHUTTING_DOWN = "SHUTTING_DOWN"

    @classmethod
    def terminal(cls) -> set["RuntimeState"]:
        """States from which the kernel does not transition out automatically."""
        return {cls.SHUTTING_DOWN}

    @classmethod
    def active(cls) -> set["RuntimeState"]:
        """States that indicate the Runtime is doing real work (not idle)."""
        return {cls.OBSERVING, cls.PROCESSING, cls.SPEAKING}
