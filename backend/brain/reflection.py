"""
Reflection engine for J.A.R.V.I.S.

Periodically reviews recent conversations and asks Claude to extract:
  - new facts worth remembering
  - behavioral patterns noticed
  - existing facts that look wrong

Writes findings back to long-term memory.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any


from config import settings
from memory import long_term


logger = logging.getLogger("jarvis.reflection")


REFLECTION_PROMPT = """\
You are J.A.R.V.I.S. performing a periodic self-review.

Given the recent conversation excerpts below, your current memory facts, and \
recent mood readings, identify:

1. NEW FACTS — stable preferences, names, dates, life details the user has \
   shared that should be remembered across sessions.
2. NEW PATTERNS — behavioral patterns you notice (e.g. "asks about weather \
   each morning", "more focused on Mondays", "prefers dry humor in the evening").
3. CORRECTIONS — existing facts that look wrong or outdated and should be \
   removed.

Output STRICT JSON of the form:
{{
  "new_facts": [
    {{"key": "short_key", "value": "...", "category": "...", "confidence": 0.9}}
  ],
  "new_patterns": [
    {{"description": "...", "confidence": 0.7, "evidence": "short quote or paraphrase"}}
  ],
  "forget_keys": ["key_a", "key_b"]
}}

Rules:
- Only include items with high confidence. If unsure, omit.
- Do NOT include transient facts (what the user said today that isn't a stable trait).
- Patterns must be supported by AT LEAST TWO messages in the excerpts.
- If there is nothing new, return an empty structure. Do not invent.

Output only the JSON object, nothing else.
"""


async def run_reflection_pass(user_id: str = "default",
                              lookback_messages: int = 50) -> dict:
    """
    Review recent activity and update memory. Returns a summary of what changed.
    """
    if not settings.ANTHROPIC_API_KEY and not settings.GEMINI_API_KEY:
        return {"status": "skipped", "reason": "No LLM API key set"}

    # 1. Gather data
    convos = await long_term.recent_conversations(limit=lookback_messages)
    if len(convos) < 6:
        return {"status": "skipped", "reason": "not enough conversation history"}

    existing_facts = await long_term.list_facts(limit=100)
    recent_moods = await long_term.recent_moods(limit=10)
    existing_patterns = await long_term.list_patterns(min_confidence=0.0, limit=50)

    convo_text = "\n".join(
        f"[{c['timestamp']}] {c['role'].upper()}: {c['content']}"
        for c in convos[-lookback_messages:]
    )
    facts_text = "\n".join(f"- {f['key']}: {f['value']}" for f in existing_facts) or "(none)"
    patterns_text = "\n".join(f"- {p['description']}" for p in existing_patterns) or "(none)"
    moods_text = "\n".join(f"- {m['timestamp']}: {m['reading']}" for m in recent_moods) or "(none)"

    user_msg = (
        f"=== CURRENT FACTS ===\n{facts_text}\n\n"
        f"=== EXISTING PATTERNS ===\n{patterns_text}\n\n"
        f"=== RECENT MOODS ===\n{moods_text}\n\n"
        f"=== RECENT CONVERSATIONS ({len(convos)} entries) ===\n{convo_text}\n\n"
        "Reflect and return JSON only."
    )

    # 2. Call LLM
    try:
        from llm import get_provider

        provider = get_provider()
        resp = await provider.chat(
            messages=[{"role": "user", "content": user_msg}],
            model=settings.CLAUDE_MODEL,
            max_tokens=2000,
            system=REFLECTION_PROMPT,
            tools=[],
        )
        text = resp.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
    except Exception as e:
        logger.exception("Reflection pass failed")
        return {"status": "error", "error": str(e)}

    # 3. Apply changes
    added_facts = 0
    added_patterns = 0
    forgot = 0

    for f in data.get("new_facts", []) or []:
        key = f.get("key")
        val = f.get("value")
        if not key or not val:
            continue
        await long_term.set_fact(
            key, val,
            f.get("category", "general"),
            float(f.get("confidence", 0.9)),
        )
        added_facts += 1

    for p in data.get("new_patterns", []) or []:
        desc = p.get("description")
        if not desc:
            continue
        await long_term.add_pattern(
            desc,
            evidence=p.get("evidence"),
            confidence=float(p.get("confidence", 0.6)),
        )
        added_patterns += 1

    for k in data.get("forget_keys", []) or []:
        if await long_term.delete_fact(k):
            forgot += 1

    summary = {
        "status": "ok",
        "added_facts": added_facts,
        "added_patterns": added_patterns,
        "forgot_keys": forgot,
        "considered_messages": len(convos),
    }
    logger.info("Reflection pass: %s", summary)
    return summary