"""
JARVIS 3.0 — Context Engine package.

The Context Engine is the second Runtime service. It subscribes to Observer
events and maintains the authoritative live context snapshot: "What is
happening right now?"

It also owns a ContextInterpreter that derives generic behavioral signals
from the same event stream.

Public surface:
    ContextEngine              — RuntimeService implementation
    Context                    — live context snapshot dataclass
    ContextInterpreter         — derives generic signals from Observer events
    ContextInterpretation      — immutable interpretation result
    InteractionLevel           — enum (idle / low / medium / high)
    StabilityLevel             — enum (fluctuating / recent / steady)
"""
from __future__ import annotations

from .context import Context
from .context_engine import ContextEngine
from .interpreter import (
    ContextInterpretation,
    ContextInterpreter,
    InteractionLevel,
    StabilityLevel,
)

__all__ = [
    "Context",
    "ContextEngine",
    "ContextInterpreter",
    "ContextInterpretation",
    "InteractionLevel",
    "StabilityLevel",
]
