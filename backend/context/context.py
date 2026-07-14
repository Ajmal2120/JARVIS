"""
Live context snapshot for JARVIS 3.0.

`Context` is a single point-in-time snapshot of the user's current operating
environment. It contains only raw data — no inference, no classification,
no history.

Every field maps directly to an Observer event field or to a field in the
Observer's `snapshot` dict. The Context Engine maintains one `Context`
instance and overwrites it on every relevant event.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Context:
    """Current live context of the user.

    All fields default to unknown/empty. The Context Engine updates them
    from Observer events. A field stays at its default until an event
    provides a real value.
    """

    active_window: str = ""
    active_application: str = ""
    last_activity_timestamp: Optional[datetime] = None
    is_idle: bool = False
    idle_seconds: float = 0.0
    system_uptime_s: float = 0.0
    is_locked: bool = False
    confidence: float = 1.0
    last_updated: datetime = field(default_factory=datetime.now)
