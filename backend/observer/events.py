"""
Observer event types.

All events extend `RuntimeEvent` (from `runtime.events`) so existing
Runtime subscribers can handle them alongside `StateChange`, `HeartbeatTick`,
etc. Subscribers should dispatch with `isinstance`, never by string.

The `snapshot` field carries the full current state on every event so a
subscriber that receives only one event type still has access to the
complete picture. This avoids the Observer having to know which fields
each subscriber cares about.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from runtime.events import RuntimeEvent


@dataclass(frozen=True)
class ObserverEvent:
    """Base class for all Observer events.

    Every concrete event includes the full system snapshot at the moment
    the change was detected. This is cheap — the snapshot is a shallow
    dict of strings and ints, rebuilt only when something actually changes.
    """

    type: str = "observer"
    snapshot: dict = field(default_factory=dict)


@dataclass(frozen=True)
class WindowChanged(ObserverEvent):
    """Active window title changed."""

    type: str = "window_changed"
    window_title: str = ""


@dataclass(frozen=True)
class ApplicationChanged(ObserverEvent):
    """Foreground process name changed."""

    type: str = "application_changed"
    process_name: str = ""


@dataclass(frozen=True)
class IdleStateChanged(ObserverEvent):
    """User transitioned between idle and active.

    `is_idle` is True when the user has been inactive beyond the idle
    threshold.
    """

    type: str = "idle_state_changed"
    is_idle: bool = False
    idle_seconds: float = 0.0


@dataclass(frozen=True)
class LockStateChanged(ObserverEvent):
    """Screen was locked or unlocked."""

    type: str = "lock_state_changed"
    is_locked: bool = False
