"""
JARVIS 3.0 — Planner models.

The Planner emits a Plan after each evaluation cycle.  A Plan answers
exactly one question:

    "Given the current hypotheses, should the system act?"

Three possible decisions:

    ASK_PERMISSION  — there is meaningful evidence; a future Suggestion
                      Engine may earn the right to propose something.
    WAIT           — there is signal but not yet enough.  Stay quiet
                      and re-evaluate next cycle.
    IGNORE         — there is nothing actionable (no hypotheses, user
                      idle, user locked, app-switching wildly, etc.).

The Planner never specifies WHAT should be suggested.  That is the
Suggestion Engine's responsibility in a future phase.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Tuple

from inference.models import Hypothesis


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

class Decision(str, Enum):
    """Whether the system should consider acting.

    Stays a string enum so JSON-friendly consumers (Suggestion Engine,
    CognitiveLogger, REST API) can serialise it without ceremony.
    """

    ASK_PERMISSION = "ASK_PERMISSION"
    WAIT = "WAIT"
    IGNORE = "IGNORE"


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Below this, no hypothesis is strong enough to ASK_PERMISSION.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.7


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    """Output of one Planner evaluation cycle.

    A Plan is a frozen, point-in-time decision.  It is NOT a TODO list.
    The downstream Suggestion Engine (later phase) consumes Plans and
    decides whether to generate a proposal; the Planner never knows what
    the proposal would be.

    Attributes:
        plan_id:               Stable identifier for this Plan instance.
        timestamp:             When this Plan was produced.
        decision:              One of `Decision`.
        reason:                Human-readable explanation of the choice.
        confidence:            0.0–1.0 confidence in the decision itself.
        supporting_hypotheses: Tuple of Hypothesis IDs that influenced
                               the choice.  Empty for IGNORE.
        supporting_evidence:   Evidence strings aggregated from the
                               supporting hypotheses.  Mirrors what the
                               rules evaluated.
    """

    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    decision: Decision = Decision.IGNORE
    reason: str = ""
    confidence: float = 0.0
    supporting_hypotheses: Tuple[str, ...] = ()
    supporting_evidence: Tuple[str, ...] = ()
    # Stable ID of the Inference cycle this Plan corresponds to.  Lets
    # the future Suggestion Engine correlate Plans with their triggers.
    inference_cycle_id: str = ""

    def __post_init__(self) -> None:
        # ``frozen=True`` rejects assignment, so we validate at
        # construction time only.
        if not isinstance(self.decision, Decision):
            raise ValueError(f"decision must be a Decision, got {type(self.decision).__name__}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    # ------------------------------------------------------------------ #
    # Convenience                                                         #
    # ------------------------------------------------------------------ #

    @property
    def is_actionable(self) -> bool:
        """True when a downstream component is allowed to do anything."""
        return self.decision == Decision.ASK_PERMISSION

    def with_hypotheses(self, hypotheses: List[Hypothesis]) -> "Plan":
        """Return a new Plan carrying the supporting hypothesis IDs.

        The runtime never modifies a frozen Plan in place.  We rebuild
        one with the additional information.
        """
        return Plan(
            plan_id=self.plan_id,
            timestamp=self.timestamp,
            decision=self.decision,
            reason=self.reason,
            confidence=self.confidence,
            supporting_hypotheses=tuple(h.id for h in hypotheses),
            supporting_evidence=tuple(
                ev for h in hypotheses for ev in h.evidence
            ),
            inference_cycle_id=self.inference_cycle_id,
        )
