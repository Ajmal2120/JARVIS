"""
JARVIS 3.0 — Inference rule interface.

An InferenceRule is a pluggable unit of inference.  It receives the
current Context and ContextInterpretation and may return a Hypothesis
if it has enough evidence.

Rules:
    - Must NOT call the LLM.
    - Must NOT access Memory.
    - Must NOT publish Runtime events.
    - Must NOT hold references to the Engine, Observer, or Context Engine.
    - May hold internal state (e.g. accumulating evidence over cycles).
    - Must return None when evidence is insufficient.

Adding a new rule:
    1. Create a new file in backend/inference/rules/.
    2. Implement the evaluate() method.
    3. Add the class to InferenceEngine.DEFAULT_RULES (one tuple entry).
    That's it — no other files change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from context import Context, ContextInterpretation
from .models import Hypothesis

if TYPE_CHECKING:
    pass


class InferenceRule(ABC):
    """Base class for all inference rules.

    Lifecycle: instantiated by InferenceEngine at startup.
    evaluate() is called whenever the Context Engine publishes a
    new interpretation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this rule."""

    @abstractmethod
    def evaluate(
        self, context: Context, interpretation: ContextInterpretation
    ) -> Hypothesis | None:
        """Return a Hypothesis if evidence supports it, else None.

        Deterministic only.  No external calls.
        """
