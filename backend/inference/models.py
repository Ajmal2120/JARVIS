"""
JARVIS 3.0 — Inference Engine models.

The Inference Engine produces a ranked list of Hypotheses about what
the user is trying to accomplish. Each Hypothesis carries a confidence
score and an evidence trail.

No LLM. No memory. No suggestions. Pure deterministic inference.

Persistence (added Phase 5.5):

Hypotheses are no longer regenerated from scratch every cycle. They
are *updated*: confidence is averaged with the new reading, evidence
is de-duplicated and re-ordered, and metadata (age, update_count) is
carried forward. Public API (``get_current_hypotheses``) is unchanged.

Each Hypothesis now also tracks:
    created_at:     First time this hypothesis was produced.
    last_updated:   Most recent update time.
    update_count:   How many times this hypothesis has been refreshed.
    cycle_id:       The cycle ID that last produced this hypothesis.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Hypothesis:
    """One possible user goal, with a confidence score and evidence.

    Attributes:
        id:            Stable slug — e.g. "software_development".
        confidence:    0.0–1.0.  Higher means stronger evidence.
        evidence:      Human-readable strings describing what
                       triggered this hypothesis.  Used for debugging
                       and future explanation surfaces.
        inferred_at:   When this hypothesis was produced (last update
                       time, kept for backward compatibility).
        created_at:    When this hypothesis was first created (set on
                       initial creation, never changes on updates).
        last_updated:  When this hypothesis was last updated (== created_at
                       on initial creation, refreshed every update).
        update_count:  Number of times this hypothesis has been refreshed
                       since it was first created.
        cycle_id:      The reasoning cycle ID that last produced this
                       hypothesis (used for log correlation).
    """

    id: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    inferred_at: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    update_count: int = 0
    cycle_id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Hypothesis.id must be a non-empty string")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Hypothesis.confidence must be in [0, 1], got {self.confidence}"
            )
        if self.update_count < 0:
            raise ValueError(
                f"Hypothesis.update_count must be >= 0, got {self.update_count}"
            )

    # ------------------------------------------------------------------ #
    # Convenience                                                         #
    # ------------------------------------------------------------------ #

    @property
    def age_seconds(self) -> float:
        """Age of this hypothesis in seconds since ``created_at``."""
        return (datetime.now() - self.created_at).total_seconds()

    @property
    def is_stale(self, max_age_s: float = 300.0) -> bool:
        """True when this hypothesis hasn't been refreshed in ``max_age_s``.

        Default 5 minutes. Used by the CognitiveLogger and future
        Suggestion Engine to decide when to expire entries.
        """
        return (datetime.now() - self.last_updated).total_seconds() > max_age_s

    # ------------------------------------------------------------------ #
    # Update / merge                                                      #
    # ------------------------------------------------------------------ #

    def with_update(
        self,
        *,
        confidence: Optional[float] = None,
        evidence: Optional[Iterable[str]] = None,
        cycle_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> "Hypothesis":
        """Return a new Hypothesis reflecting an updated reading.

        ``confidence`` is averaged with the previous reading so a single
        outlier cycle cannot spike the score. ``evidence`` is de-duplicated
        while preserving original order. ``cycle_id`` is updated only if
        explicitly provided.

        ``update_count`` is incremented by one.
        """
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {confidence}"
            )
        now = now or datetime.now()

        new_conf = (
            confidence if confidence is not None else self.confidence
        )
        # Average the readings (weighted 50/50) for stability.
        merged_conf = (self.confidence + new_conf) / 2.0

        if evidence is None:
            new_evidence = list(self.evidence)
        else:
            merged: list[str] = []
            seen: set[str] = set()
            for ev in list(self.evidence) + list(evidence):
                key = ev.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    merged.append(ev)
            new_evidence = merged

        return replace(
            self,
            confidence=merged_conf,
            evidence=new_evidence,
            inferred_at=now,
            last_updated=now,
            update_count=self.update_count + 1,
            cycle_id=(cycle_id if cycle_id is not None else self.cycle_id),
        )

    @classmethod
    def create(
        cls,
        *,
        id: str,
        confidence: float,
        evidence: Optional[Iterable[str]] = None,
        cycle_id: str = "",
        now: Optional[datetime] = None,
    ) -> "Hypothesis":
        """Create a fresh hypothesis. ``update_count`` starts at 0."""
        now = now or datetime.now()
        return cls(
            id=id,
            confidence=confidence,
            evidence=list(evidence or []),
            inferred_at=now,
            created_at=now,
            last_updated=now,
            update_count=0,
            cycle_id=cycle_id,
        )
