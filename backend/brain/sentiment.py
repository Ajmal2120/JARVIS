"""
Text sentiment / mood inference for J.A.R.V.I.S.

Approach: send the last N user messages to Claude with a tiny prompt asking
for a one-word mood classification. Cheap, accurate, no extra dependencies.

Valid moods: relaxed, focused, tired, frustrated, playful, sad, neutral.
"""
from __future__ import annotations

import time
from typing import Optional

from config import settings
from memory import long_term
from memory.short_term import buffer


VALID_MOODS = {
    "relaxed", "focused", "tired", "frustrated",
    "playful", "sad", "neutral",
}


# Cache: avoid hammering Claude every chat
_CACHE: dict[str, tuple[float, str]] = {}
_CACHE_TTL_SECONDS = 600  # 10 minutes


def _cached(user_id: str) -> Optional[str]:
    entry = _CACHE.get(user_id)
    if not entry:
        return None
    ts, mood = entry
    if time.time() - ts > _CACHE_TTL_SECONDS:
        _CACHE.pop(user_id, None)
        return None
    return mood


def _set_cache(user_id: str, mood: str) -> None:
    _CACHE[user_id] = (time.time(), mood)


def _heuristic_mood(messages: list[str]) -> Optional[str]:
    """
    Cheap fallback that runs before we hit Claude. Returns a guess based on
    simple lexical signals. Never returns anything if messages are empty.
    """
    if not messages:
        return None
    text = " ".join(messages).lower()
    n = len(text.split())

    # Very short / terse + late hours = tired
    if n < 6 and any(t in text for t in ("tired", "sleepy", "exhausted", "zzz", "long day")):
        return "tired"

    # Explicit frustration cues
    if any(t in text for t in ("angry", "annoyed", "frustrated", "ugh", "hate", "broken")):
        return "frustrated"

    # Sad cues
    if any(t in text for t in ("sad", "down", "lonely", "miss", "lost", "depressed")):
        return "sad"

    # Playful cues
    if any(t in text for t in ("lol", "haha", ":)", "😂", "lmao", "hehe", "bro", "kidding")):
        return "playful"

    return None



async def infer_mood(user_id: str = "default",
                     recent_user_messages: Optional[list[str]] = None) -> str:
    """
    Infer the user's current mood. Returns one of VALID_MOODS.

    Order of resolution:
      1. Cache (10 min TTL)
      2. Lexical heuristic (free, no API call)
      3. Claude API call
      4. Fallback: 'neutral'
    """
    cached = _cached(user_id)
    if cached:
        return cached

    # Gather messages
    if recent_user_messages is None:
        recent_buf = buffer.recent(user_id, limit=10)
        recent_user_messages = [
            t["content"] for t in recent_buf if t["role"] == "user"
        ][-5:]
    recent_user_messages = [m for m in recent_user_messages if m and m.strip()]
    if not recent_user_messages:
        return "neutral"

    # Try heuristic first
    guess = _heuristic_mood(recent_user_messages)
    if guess:
        _set_cache(user_id, guess)
        await long_term.add_mood(guess, source="heuristic")
        return guess

    # Otherwise ask the LLM
    try:
        from llm import get_provider

        provider = get_provider()
        joined = "\n".join(f"- {m}" for m in recent_user_messages)
        prompt = (
            "You classify the user's emotional state from their recent messages. "
            "Reply with exactly ONE word from this set: "
            "relaxed, focused, tired, frustrated, playful, sad, neutral.\n\n"
            f"Recent user messages:\n{joined}\n\n"
            "Mood:"
        )
        resp = await provider.chat(
            messages=[{"role": "user", "content": prompt}],
            model=settings.CLAUDE_MODEL,
            max_tokens=8,
            system=None,
            tools=[],
        )
        text = resp.text.lower()
        # Extract the first valid mood word
        for word in text.replace(",", " ").split():
            w = word.strip(".,!?;:").lower()
            if w in VALID_MOODS:
                _set_cache(user_id, w)
                await long_term.add_mood(w, source="claude")
                return w
    except Exception:
        # Silent fallback — sentiment is non-critical
        pass

    return "neutral"


def invalidate_cache(user_id: str = "default") -> None:
    """Clear the cached mood for a user (e.g. after a long pause)."""
    _CACHE.pop(user_id, None)
