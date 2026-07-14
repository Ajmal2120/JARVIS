"""
Supporting dataclasses for the Context Engine.

Currently unused — `Context` in `context.py` is the only model needed.
This file exists to satisfy the planned file structure and provides a
home for future context-adjacent models (e.g. typed confidence tiers,
context deltas) without growing `context.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .context import Context


@dataclass(frozen=True)
class ContextDelta:
    """Records what changed between two Context snapshots.

    Not stored — emitted only for debugging or future subscribers that
    want change notifications rather than full snapshots.
    """

    timestamp: datetime
    changed_fields: list[str]
    previous: Context
    current: Context
