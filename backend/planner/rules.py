"""
JARVIS 3.0 — Planner rule interface.

A PlanningRule is a pluggable unit of decision logic.  It receives the
current ranked hypotheses plus the live context + interpretation, and
returns either:

    - A Decision (the rule's recommendation, with a reason and confidence), or
    - None (the rule has no opinion — let other rules speak).

The Planner is the *only* consumer of PlanningRule outputs.  It merges
the recommendations into a single Plan.

Rules:
    - Must NOT call the LLM.
    - Must NOT access Memory.
    - Must NOT publish Runtime events.
    - Must NOT hold references to Planner, Suggestion Engine, or Skills.
    - Must be deterministic — no random, no wall-clock reliance other
      than what's passed in.
    - Must be fast — execution happens on the EventBus thread, blocking
      other subscribers if slow.

Adding a new rule:
    1. Drop a new file in backend/planner/rules/.
    2. Subclass PlanningRule.
    3. Add the class to Planner.DEFAULT_RULES (one tuple entry).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from context import Context, ContextInterpretation
from inference.models import Hypothesis

from .models import Decision


@dataclass(frozen=True)
class PlanningVerdict:
    """Single rule's contribution to the Plan.

    Attributes:
        decision:   The rule's recommended Decision.
        reason:     Human-readable explanation (concise).
        confidence: 0.0–1.0 confidence in the recommendation.
        weight:     Relative weight of this verdict when several rules
                    compete (default 1.0).  The Planner uses the
                    highest-weighted verdict as the Plan.
    """

    decision: Decision
    reason: str
    confidence: float = 1.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.decision, Decision):
            raise ValueError(f"decision must be a Decision, got {type(self.decision).__name__}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not (0.0 < self.weight <= 10.0):
            raise ValueError(f"weight must be in (0, 10], got {self.weight}")


class PlanningRule(ABC):
    """Base class for all planning rules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable unique identifier for this rule."""

    @abstractmethod
    def evaluate(
        self,
        context: Context,
        interpretation: ContextInterpretation,
        hypotheses: List[Hypothesis],
    ) -> Optional[PlanningVerdict]:
        """Return a verdict, or None if this rule has no opinion.

        ``hypotheses`` is the ranked list produced by the Inference
        Engine.  An empty list means "no signal".
        """
