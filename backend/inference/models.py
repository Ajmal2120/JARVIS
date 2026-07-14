"""
JARVIS 3.0 — Inference Engine models.

The Inference Engine produces a ranked list of Hypotheses about what
the user is trying to accomplish. Each Hypothesis carries a confidence
score and an evidence trail.

No LLM. No memory. No suggestions. Pure deterministic inference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass(frozen=True)
class Hypothesis:
    """One possible user goal, with a confidence score and evidence.

    Attributes:
        id:            Stable slug — e.g. "software_development".
        confidence:    0.0–1.0.  Higher means stronger evidence.
        evidence:      Human-readable strings describing what
                       triggered this hypothesis.  Used for debugging
                       and future explanation surfaces.
        inferred_at:   When this hypothesis was produced.
    """

    id: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    inferred_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Hypothesis.id must be a non-empty string")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Hypothesis.confidence must be in [0, 1], got {self.confidence}"
            )
