"""
Observation data models.

`SystemSnapshot` is a flat dict of the current system state. It is
rebuilt on every polling cycle and attached to every event as the
`snapshot` field. There is no history, no inference, no classification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Self


@dataclass
class SystemSnapshot:
    """Current point-in-time state of the operating system.

    Every field is a raw observation. None means the source could not
    be read (e.g., Win32 API not available). The Observer treats None
    as "unchanged" so transient read failures don't generate events.
    """

    window_title: str = ""
    process_name: str = ""
    idle_seconds: float = 0.0
    is_locked: bool = False
    system_uptime_s: float = 0.0
    observed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "window_title": self.window_title,
            "process_name": self.process_name,
            "idle_seconds": self.idle_seconds,
            "is_locked": self.is_locked,
            "system_uptime_s": self.system_uptime_s,
            "observed_at": self.observed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(
            window_title=d.get("window_title", ""),
            process_name=d.get("process_name", ""),
            idle_seconds=d.get("idle_seconds", 0.0),
            is_locked=d.get("is_locked", False),
            system_uptime_s=d.get("system_uptime_s", 0.0),
            observed_at=datetime.fromisoformat(d["observed_at"])
            if "observed_at" in d
            else datetime.now(),
        )
