"""
JARVIS 3.0 — Planner package.

The Planner decides WHETHER JARVIS should act.  It never decides WHAT to
do or HOW — that belongs to the future Suggestion Engine.

Pipeline position:

    Context → Inference → Planner → (future) Suggestion

Only deterministic rules. No LLM. No memory. No skill execution.
"""
from .models import Plan, Decision, DEFAULT_CONFIDENCE_THRESHOLD
from .rules import PlanningRule, PlanningVerdict
from .planner import Planner

__all__ = [
    "Plan",
    "Decision",
    "PlanningRule",
    "PlanningVerdict",
    "Planner",
    "DEFAULT_CONFIDENCE_THRESHOLD",
]
