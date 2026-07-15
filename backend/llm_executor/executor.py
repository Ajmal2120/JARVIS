"""
JARVIS 3.0 — LLM Executor.

Converts a PreparedRequest into a normalized LLMResponse using the
currently configured LLM provider.

It does NOT observe, infer, plan, suggest, or speak.

Usage::

    from llm_executor import LLMExecutor
    executor = LLMExecutor()
    response = await executor.execute(prepared_request)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from llm import get_provider
from llm.provider import ChatResponse

from .models import LLMResponse


logger = logging.getLogger("jarvis.llm_executor")


_SYSTEM_PROMPT = """You are JARVIS, an intelligent assistant observing the user's work.

Context from the Operating Intelligence layer:
{interpretation_block}

What the system thinks the user is doing:
{hypotheses_block}

The user's current focus:
{focus_block}

The Planner has determined that it is appropriate to offer help.

TASK:
Generate a short, helpful, proactive message (one or two sentences).
- Be polite but concise.
- Address what the user is working on.
- Offer specific, actionable help.
- Do NOT make assumptions beyond the provided context.
- Do NOT produce markdown formatting.
"""


def _interpretation_block(interp: Any) -> str:
    """Format the interpretation into a readable block."""
    if interp is None:
        return "(no interpretation available)"
    parts = []
    il = getattr(interp, "interaction_level", None)
    cs = getattr(interp, "context_stability", None)
    af = getattr(interp, "application_switch_frequency", 0.0)
    fd = getattr(interp, "current_focus_duration_s", 0.0)
    if il is not None:
        il_str = il.value if hasattr(il, "value") else str(il)
        parts.append(f"- Interaction level: {il_str}")
    if cs is not None:
        cs_str = cs.value if hasattr(cs, "value") else str(cs)
        parts.append(f"- Context stability: {cs_str}")
    parts.append(f"- App switches: {af:.1f}/min")
    parts.append(f"- Focus duration: {fd:.0f}s")
    return "\n".join(parts) if parts else "(no interpretation details)"


def _hypotheses_block(hypotheses: Any) -> str:
    """Format hypotheses into a readable block."""
    if not hypotheses:
        return "(no hypotheses)"
    lines = []
    for h in list(hypotheses)[:5]:
        conf = getattr(h, "confidence", 0.0)
        hid = getattr(h, "id", "?")
        lines.append(f"- {hid} (confidence: {conf:.0%})")
    return "\n".join(lines)


def _focus_block(ctx: Any, meta: Dict[str, Any]) -> str:
    """Format the focus context."""
    app = meta.get("focus_app", "") or getattr(ctx, "active_application", "") or "(unknown)"
    win = meta.get("focus_topic", "") or getattr(ctx, "active_window", "") or "(unknown)"
    return f"- Application: {app}\n- Window/topic: {win}"


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class LLMExecutor:
    """Execute a PreparedRequest against the configured LLM provider.

    Stateless. Each execute() call is independent.
    """

    name: str = "llm_executor"

    def __init__(self) -> None:
        self._provider_cache: Any = None

    async def execute(self, prepared_request: Any) -> LLMResponse:
        """Execute a PreparedRequest and return the response.

        Args:
            prepared_request: A PreparedRequest from the Suggestion Engine.

        Returns:
            LLMResponse with success=True on normal completion.
        """
        t0 = time.time()
        try:
            messages = self._build_messages(prepared_request)
        except Exception as exc:
            logger.exception("llm_executor: message construction failed: %s", exc)
            return LLMResponse(success=False, error=f"Message construction failed: {exc}")

        provider_name = ""
        model_name = ""

        try:
            provider = self._get_provider()
            provider_name = type(provider).__name__
            model_name = getattr(provider, "model", "") or ""

            response: ChatResponse = await provider.chat(messages=messages)
            content = (response.content or "").strip()

            elapsed = time.time() - t0
            logger.info(
                "llm_executor: completed in %.2fs (%s / %s)",
                elapsed, provider_name, model_name,
            )
            return LLMResponse(
                content=content,
                provider=provider_name,
                model=model_name,
                usage=getattr(response, "usage", None),
                raw=response,
                success=True,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            logger.error(
                "llm_executor: provider %s failed after %.2fs: %s",
                provider_name or "unknown", elapsed, exc,
            )
            return LLMResponse(
                provider=provider_name or "unknown",
                model=model_name,
                success=False,
                error=str(exc),
            )

    def _get_provider(self) -> Any:
        if self._provider_cache is None:
            self._provider_cache = get_provider()
        return self._provider_cache

    def _build_messages(self, pr: Any) -> List[Dict[str, Any]]:
        """Convert a PreparedRequest into a list of chat messages."""
        interpretation = getattr(pr, "interpretation", None)
        hypotheses = list(getattr(pr, "hypotheses", []) or [])
        context = getattr(pr, "context", None)
        meta: Dict[str, Any] = dict(getattr(pr, "provider_metadata", {}) or {})

        system_prompt = _SYSTEM_PROMPT.format(
            interpretation_block=_interpretation_block(interpretation),
            hypotheses_block=_hypotheses_block(hypotheses),
            focus_block=_focus_block(context, meta),
        )

        suggested_topics = meta.get("suggested_topics", [])
        topics_str = ", ".join(suggested_topics) if suggested_topics else "general"

        user_message = (
            "The user is currently working on '{focus}'. "
            "Provide a brief, helpful proactive message. "
            "Consider: {topics}."
        ).format(
            focus=meta.get("focus_topic", "their work"),
            topics=topics_str,
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
