"""
JARVIS Streaming Patch — jarvis_core.py modifications.

Patches jarvis_core.chat() with sentence-level streaming:
  - First complete sentence is spoken immediately via Kokoro TTS
  - Remaining sentences are buffered and form the final reply text
  - Tool-use path stays non-streaming (needs full response for inputs)

Activate by importing this module at app startup:
    from streaming_patch import apply_streaming_patch
    await apply_streaming_patch()
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping punctuation attached."""
    # Sentence boundaries: . ! ? followed by space or end
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]


async def apply_streaming_patch() -> None:
    """Monkey-patch jarvis_core.chat() with streaming support."""
    import jarvis_core as jc
    import asyncio

    original_chat = jc.chat

    async def _retry_with_backoff(coro_func, max_retries: int = 5, base_delay: float = 2.0, **kwargs):
        """Retry with exponential backoff for 429 rate limits."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await coro_func(**kwargs)
            except Exception as e:
                last_error = e
                is_rate_limit = "429" in str(e) or (hasattr(e, 'status_code') and e.status_code == 429)
                if is_rate_limit and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit (stream), retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    raise
        raise last_error

    async def streaming_chat(user_id: str, message: str) -> dict:
        # --- same pre-flight as original ---
        from brain.context import build_context
        from brain.sentiment import infer_mood
        from brain.system_prompt import build_system_prompt
        from config import settings
        from memory.short_term import buffer
        from datetime import datetime, timezone
        from zoneinfo import ZoneInfo

        ctx = await build_context(user_id, message)
        mood = ctx.get("mood") or "neutral"
        tz = ZoneInfo(settings.TIMEZONE)
        now = datetime.now(tz)
        time_str = now.strftime("%I:%M %p, %A %d %B %Y (%Z)")
        prompt = build_system_prompt(ctx)
        prompt = f"Current date and time: {time_str}\n\n" + prompt

        history = await jc._history_messages(user_id, limit=20)
        provider = await jc._get_chat_provider()
        client = provider._get_client()
        messages = history + [{"role": "user", "content": message}]
        tools_used: list[str] = []

        # --- Check for tool calls first ---
        try:
            test_response = await provider.chat(
                messages=messages,
                model=settings.CLAUDE_MODEL,
                max_tokens=10,  # Just enough to detect if it's a tool call
                system=prompt,
                tools=[],  # Don't enable tools for test
            )

            # If tool calls are detected, fall back to original chat
            if test_response.tool_calls:
                logger.info("Complex request with tools detected, falling back to original chat")
                return await original_chat(user_id, message)

            # No tool calls, proceed with streaming
            response = test_response

        except Exception as e:
            logger.warning("Failed to check for tool calls, falling back: %s", e)
            return await original_chat(user_id, message)

        # --- STREAMING PATH (no tool calls) ---
        full_text = ""
        sentence_buffer = ""
        first_sentence_spoken = False
        spoke_first = {"done": False}

        async def _speak_first(sentence: str) -> None:
            """Speak the first complete sentence immediately via the TTS queue."""
            if not spoke_first["done"]:
                spoke_first["done"] = True
                try:
                    from tts.speaker import speak_async as _qspeak
                    _qspeak(sentence)
                    logger.info("[stream] queued first sentence: %.60s…", sentence)
                except Exception as e:
                    logger.warning("[stream] first-sentence speak failed: %s", e)

        try:
            async for chunk in provider.stream_chat(
                messages=messages,
                model=settings.CLAUDE_MODEL,
                max_tokens=1024,
                system=prompt,
                tools=[],
            ):
                if chunk.get("type") == "content":
                    text_delta = chunk.get("content", "")
                    full_text += text_delta
                    sentence_buffer += text_delta
                    # Check for sentence boundary
                    if re.search(r'[.!?]\s*$', sentence_buffer.strip()):
                        sentence = sentence_buffer.strip()
                        sentence_buffer = ""
                        if not first_sentence_spoken:
                            await _speak_first(sentence)
                            first_sentence_spoken = True
                        # continue buffering remaining sentences into full_text
                elif chunk.get("type") == "done":
                    # Stream finished, no more content
                    break
        except Exception as e:
            logger.warning("Streaming failed, falling back to non-stream: %s", e)
            full_text = response.content

        reply_text = full_text.strip()

        # Same cleanup as original chat()
        reply_text = re.sub(r'^```json\s*', '', reply_text)
        reply_text = re.sub(r'^```\s*', '', reply_text)
        reply_text = re.sub(r'\s*```$', '', reply_text).strip()

        if reply_text.startswith('{'):
            try:
                import json
                parsed = json.loads(reply_text)
                if isinstance(parsed, dict):
                    mood = parsed.get("mood", mood)
                    reply_text = parsed.get("reply", reply_text)
            except json.JSONDecodeError:
                pass

        mood_match = re.search(r'"mood"\s*:\s*"(\w+)"', reply_text)
        if mood_match:
            mood = mood_match.group(1)
            reply_text = re.sub(r'\{[^}]*"mood"[^}]*\}', '', reply_text).strip()

        # Clean up any remaining tool call artifacts
        reply_text = re.sub(r'\[TOOL_CALL\].*?\[/TOOL_CALL\]', '', reply_text, flags=re.DOTALL).strip()
        reply_text = re.sub(r'``.*?``', '', reply_text, flags=re.DOTALL).strip()
        reply_text = re.sub(r'<function>.*?</function>', '', reply_text, flags=re.DOTALL).strip()
        reply_text = re.sub(r'```.*?```', '', reply_text, flags=re.DOTALL).strip()
        reply_text = re.sub(r'\{[^}]*"tool"[^}]*\}', '', reply_text, flags=re.DOTALL).strip()
        reply_text = reply_text.strip()

        if not reply_text:
            reply_text = "I'm at a loss for words, Sir. A rare occurrence."

        await jc._remember_turn(user_id, message, reply_text, mood)
        return {
            "reply": reply_text,
            "mood": mood,
            "tools_used": tools_used,
            "audio_b64": None,
        }

    # Monkey-patch
    jc.chat = streaming_chat
    logger.info("[streaming] jarvis_core.chat() patched — sentence streaming active")