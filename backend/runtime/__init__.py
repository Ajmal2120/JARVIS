"""
JARVIS 3.0 — Operating Intelligence Runtime (kernel).

The Runtime is the lifecycle owner of JARVIS 3.0. It boots with the backend,
runs a low-frequency heartbeat, and registers background services that will
be added in subsequent phases (Observer, Context Engine, Planner, etc.).

It is intentionally inert today: it does not observe, reason, call the LLM,
or execute skills. It only manages lifecycle and emits events.

Public surface:
    Runtime             — main kernel class
    RuntimeState        — lifecycle state enum
    RuntimeService      — protocol for future services
    get_runtime()       — process-wide accessor

The package is additive. Nothing in the existing Reactive Engine (voice,
router, intent, LLM, TTS, FastAPI endpoints) is modified by importing this
package.
"""
from __future__ import annotations

from .events import EventBus, RuntimeEvent
from .runtime import Runtime, get_runtime
from .scheduler import Scheduler
from .state import RuntimeState

__all__ = [
    "Runtime",
    "RuntimeState",
    "RuntimeEvent",
    "EventBus",
    "Scheduler",
    "get_runtime",
]
