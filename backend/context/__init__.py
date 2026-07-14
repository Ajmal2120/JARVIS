"""
JARVIS 3.0 — Context Engine package.

The Context Engine is the second Runtime service. It subscribes to Observer
events and maintains the authoritative live context snapshot: "What is
happening right now?"

It never reads the OS, never calls the LLM, never writes to Memory, never
makes decisions. It only transforms Observer events into a structured context.

Public surface:
    ContextEngine          — RuntimeService implementation
    Context                — live context snapshot dataclass
    get_current_context()  — module-level accessor (returns a copy)
"""
from __future__ import annotations

from .context import Context
from .context_engine import ContextEngine

__all__ = [
    "Context",
    "ContextEngine",
]
