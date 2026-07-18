"""
JARVIS 3.0 — Suggestion Messenger.

Builds contextual, calm suggestion messages and reasons from the
highest-confidence hypothesis and context.

This module is a pure function layer — no side effects, no I/O,
no LLM calls.  It reads hypothesis metadata and context signals
to produce a human-readable message and explanation.

Design:
    - message: What JARVIS says to the user.
    - reason: Short explanation of *why* this suggestion exists.

Both are derived from existing context and hypotheses only.
No information is invented.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("jarvis.suggestion")


# ---------------------------------------------------------------------------
# Contextual message templates
# ---------------------------------------------------------------------------

#: Mapping of hypothesis id -> (message template, reason template)
#: Templates use f-string style formatting with named placeholders.
#: Placeholders are resolved at call time from context/hypothesis.
CONTEXTUAL_TEMPLATES: dict[str, dict[str, str]] = {
    "software_development": {
        "message": (
            "Sir, I noticed something while you were {focus_topic} in "
            "{focus_app}."
        ),
        "reason": (
            "You've been focused on {focus_topic} for "
            "{focus_duration_min:.0f} minutes with {stability} context."
        ),
    },
    "research": {
        "message": (
            "Sir, I found something that may help your research on "
            "{focus_topic}."
        ),
        "reason": (
            "You've been reading {focus_topic} for "
            "{focus_duration_min:.0f} minutes."
        ),
    },
    "writing": {
        "message": (
            "Sir, I have an idea that may improve this document "
            "about {focus_topic}."
        ),
        "reason": (
            "You've been editing {focus_topic} for "
            "{focus_duration_min:.0f} minutes."
        ),
    },
    "browsing": {
        "message": (
            "Sir, while you were browsing, I noticed something that "
            "might be relevant to {focus_topic}."
        ),
        "reason": (
            "You've been on {focus_topic} for "
            "{focus_duration_min:.0f} minutes."
        ),
    },
    "communication": {
        "message": (
            "Sir, I noticed something while you were handling "
            "communication in {focus_app}."
        ),
        "reason": (
            "You've been in {focus_app} for "
            "{focus_duration_min:.0f} minutes."
        ),
    },
}

#: Fallback for unknown hypothesis ids.
DEFAULT_TEMPLATES: dict[str, str] = {
    "message": "Sir, I think I noticed something that might help.",
    "reason": "Your current focus has been stable for a while.",
}


# ---------------------------------------------------------------------------
# Helper: extract context values
# ---------------------------------------------------------------------------

def _get_context_value(context: Any, attr: str, default: str = "") -> str:
    """Safely extract a string attribute from a context object."""
    if context is None:
        return default
    val = getattr(context, attr, None) or default
    if not isinstance(val, str):
        val = str(val)
    return val


def _get_float_value(context: Any, attr: str, default: float = 0.0) -> float:
    """Safely extract a float attribute from a context object."""
    if context is None:
        return default
    val = getattr(context, attr, None)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _get_stability_label(stability: Any) -> str:
    """Convert a stability enum/value to a human label."""
    if stability is None:
        return "stable"
    val = str(stability).lower()
    if "steady" in val or "stable" in val:
        return "stable"
    if "fluctuat" in val or "unstable" in val:
        return "somewhat unstable"
    return "stable"


def _format_minutes(seconds: float) -> float:
    """Convert seconds to minutes (rounded)."""
    return max(1.0, round(seconds / 60.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_message(
    hypothesis_id: str,
    context: Any,
    focus_topic_override: Optional[str] = None,
) -> str:
    """Build a contextual suggestion message.

    Args:
        hypothesis_id: The dominant hypothesis id (e.g. "software_development").
        context: The current Context object.
        focus_topic_override: Optional override for the focus topic.

    Returns:
        A human-readable message string.
    """
    templates = CONTEXTUAL_TEMPLATES.get(hypothesis_id, {})
    template = templates.get("message", DEFAULT_TEMPLATES["message"])

    # Resolve placeholders
    focus_topic = (
        focus_topic_override
        or _get_context_value(context, "active_window")
        or "your work"
    )
    focus_app = (
        _get_context_value(context, "active_application")
        or "your application"
    )

    try:
        return template.format(
            focus_topic=focus_topic,
            focus_app=focus_app,
        )
    except (KeyError, ValueError) as exc:
        logger.debug("messenger: format failed for message template: %s", exc)
        return DEFAULT_TEMPLATES["message"]


def build_reason(
    hypothesis_id: str,
    context: Any,
    interpretation: Any,
) -> str:
    """Build a short explanation for why this suggestion exists.

    The reason is derived from:
    - Focus duration (how long the user has been on the current task)
    - Context stability (steady / fluctuating)
    - Active application / window

    No information is invented.  If data is missing, a generic
    reason is returned.

    Args:
        hypothesis_id: The dominant hypothesis id.
        context: The current Context object.
        interpretation: The current ContextInterpretation object.

    Returns:
        A short explanation string.
    """
    templates = CONTEXTUAL_TEMPLATES.get(hypothesis_id, {})
    template = templates.get("reason", DEFAULT_TEMPLATES["reason"])

    focus_topic = (
        _get_context_value(context, "active_window")
        or "your current task"
    )
    focus_duration_s = _get_float_value(
        interpretation or context, "current_focus_duration_s", 120.0
    )
    focus_duration_min = _format_minutes(focus_duration_s)

    stability = _get_stability_label(
        getattr(interpretation, "context_stability", None) if interpretation else None
    )

    try:
        return template.format(
            focus_topic=focus_topic,
            focus_duration_min=focus_duration_min,
            stability=stability,
            focus_app=_get_context_value(context, "active_application", "this app"),
        )
    except (KeyError, ValueError) as exc:
        logger.debug("messenger: format failed for reason template: %s", exc)
        return DEFAULT_TEMPLATES["reason"]


def build_suggestion_text(
    hypothesis_id: str,
    context: Any,
    interpretation: Any,
    focus_topic_override: Optional[str] = None,
) -> tuple[str, str]:
    """Build both message and reason in one call.

    Returns:
        (message, reason) tuple.
    """
    message = build_message(hypothesis_id, context, focus_topic_override)
    reason = build_reason(hypothesis_id, context, interpretation)
    return message, reason
