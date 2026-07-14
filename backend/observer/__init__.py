"""
JARVIS 3.0 — Observer package.

The Observer is the first Runtime service. It watches the operating system
and publishes raw observations. It never interprets, classifies, or acts
on what it sees.

Public surface:
    Observer              — the RuntimeService implementation
    SystemSnapshot        — current state snapshot (one per system, not per event)
    WindowChanged         — active window title changed
    ApplicationChanged    — foreground process changed
    IdleStateChanged      — user became idle or active
    LockStateChanged      — screen was locked or unlocked

Usage:
    from observer import Observer, WindowChanged
    runtime.register_service(Observer())
"""
from __future__ import annotations

from .events import (
    ApplicationChanged,
    IdleStateChanged,
    LockStateChanged,
    ObserverEvent,
    WindowChanged,
)
from .models import SystemSnapshot
from .observer import Observer

__all__ = [
    "Observer",
    "SystemSnapshot",
    "ObserverEvent",
    "WindowChanged",
    "ApplicationChanged",
    "IdleStateChanged",
    "LockStateChanged",
]
