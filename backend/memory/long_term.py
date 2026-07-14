"""
Long-term memory for J.A.R.V.I.S.

Backed by SQLite via aiosqlite. Five tables:
  - facts:         stable preferences, names, dates
  - patterns:      observed behavioral patterns (e.g. "asks weather at 8 AM")
  - moods:         recent emotional readings
  - rules:         proactivity rules (editable without code)
  - conversations: history (also for reflection)

Connection model: a single shared aiosqlite connection per process. This
avoids the Python 3.14 + aiosqlite issue where each connection spawns its
own background Thread that cannot be restarted. Writes are serialised by an
asyncio lock; SQLite itself handles concurrency at the file level.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from config import settings


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    category TEXT,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT NOT NULL,
    evidence TEXT,
    confidence REAL DEFAULT 0.5,
    last_observed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS moods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reading TEXT NOT NULL,
    source TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    condition_json TEXT NOT NULL,
    action TEXT NOT NULL,
    action_payload TEXT,
    enabled INTEGER DEFAULT 1,
    cooldown_hours REAL DEFAULT 12,
    last_fired TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    mood TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    target TEXT,
    status TEXT NOT NULL,
    detail TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# -----------------------------------------------------------------------------
# Connection helper
# -----------------------------------------------------------------------------

_db_conn: Optional[aiosqlite.Connection] = None
_db_lock = asyncio.Lock()
_db_ready = False


async def _get_db() -> aiosqlite.Connection:
    """Return the shared connection, opening it on first call."""
    global _db_conn, _db_ready
    async with _db_lock:
        if _db_conn is None:
            settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            _db_conn = await aiosqlite.connect(settings.DB_PATH)
            _db_conn.row_factory = aiosqlite.Row
        if not _db_ready:
            await _db_conn.executescript(SCHEMA)
            await _db_conn.commit()
            _db_ready = True
    return _db_conn


async def init_db() -> None:
    """Initialize database tables. Idempotent."""
    db = await _get_db()
    # Schema was already applied by _get_db; nothing else to do.
    _ = db


# -----------------------------------------------------------------------------
# CONVERSATION SUMMARIES
# -----------------------------------------------------------------------------

async def generate_conversation_summary(user_id: str, max_turns: int = 10) -> str:
    """Generate a summary of the last N conversation turns."""
    db = await _get_db()

    # Get recent conversation turns
    cursor = await db.execute("""
        SELECT role, content, mood, timestamp
        FROM conversations
        WHERE timestamp > datetime('now', '-24 hours')
        ORDER BY timestamp DESC
        LIMIT ?
    """, (max_turns,))

    rows = await cursor.fetchall()
    turns = [dict(row) for row in rows]

    if not turns:
        return "No recent conversations to summarize."

    # Format for summary
    context = []
    for turn in reversed(turns):  # Chronological order
        role = turn['role'].capitalize()
        content = turn['content']
        mood = turn.get('mood', 'neutral')
        timestamp = turn['timestamp']

        # Truncate long messages
        if len(content) > 100:
            content = content[:100] + "..."

        context.append(f"[{timestamp}] {role} ({mood}): {content}")

    return "\n".join(context)


async def save_conversation_summary(user_id: str, summary: str) -> None:
    """Save a conversation summary to the database."""
    db = await _get_db()

    # Store in conversations table as a system message
    await db.execute(
        "INSERT INTO conversations (role, content, mood) VALUES (?, ?, ?)",
        ("system", f"[Conversation summary]: {summary}", "summary")
    )

    # Also store as a rule for future reference
    await db.execute("""
        INSERT INTO rules (name, condition_json, action, action_payload, cooldown_hours)
        VALUES (?, ?, ?, ?, 0)
    """, (
        f"conversation_summary_{user_id}",
        json.dumps({"user_id": user_id}),
        "show_summary",
        json.dumps({"summary": summary, "timestamp": datetime.now().isoformat()}),
        0
    ))

    await db.commit()


# -----------------------------------------------------------------------------
# FACTS
# -----------------------------------------------------------------------------

async def get_fact(key: str) -> Optional[str]:
    db = await _get_db()
    cursor = await db.execute("SELECT value FROM facts WHERE key = ?", (key,))
    row = await cursor.fetchone()
    return row["value"] if row else None


async def set_fact(key: str, value: str, category: str = "general",
                   confidence: float = 1.0) -> None:
    db = await _get_db()
    await db.execute(
        """
        INSERT INTO facts (key, value, category, confidence, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            category = excluded.category,
            confidence = excluded.confidence,
            updated_at = CURRENT_TIMESTAMP
        """,
        (key, value, category, confidence),
    )
    await db.commit()


async def delete_fact(key: str) -> bool:
    db = await _get_db()
    cursor = await db.execute("DELETE FROM facts WHERE key = ?", (key,))
    await db.commit()
    return cursor.rowcount > 0


async def list_facts(category: Optional[str] = None,
                     limit: int = 200) -> list[dict[str, Any]]:
    sql = "SELECT * FROM facts"
    args: tuple = ()
    if category:
        sql += " WHERE category = ?"
        args = (category,)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    args = args + (limit,)
    db = await _get_db()
    cursor = await db.execute(sql, args)
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# -----------------------------------------------------------------------------
# PATTERNS
# -----------------------------------------------------------------------------

async def add_pattern(description: str, evidence: Any = None,
                      confidence: float = 0.5) -> int:
    ev = json.dumps(evidence) if evidence is not None else None
    db = await _get_db()
    cursor = await db.execute(
        """
        INSERT INTO patterns (description, evidence, confidence, last_observed)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (description, ev, confidence),
    )
    await db.commit()
    return cursor.lastrowid or 0


async def list_patterns(min_confidence: float = 0.0,
                        limit: int = 50) -> list[dict[str, Any]]:
    db = await _get_db()
    cursor = await db.execute(
        """
        SELECT * FROM patterns
        WHERE confidence >= ?
        ORDER BY confidence DESC, last_observed DESC
        LIMIT ?
        """,
        (min_confidence, limit),
    )
    rows = await cursor.fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("evidence"):
            try:
                d["evidence"] = json.loads(d["evidence"])
            except Exception:
                pass
        out.append(d)
    return out


async def clear_patterns() -> None:
    db = await _get_db()
    await db.execute("DELETE FROM patterns")
    await db.commit()


# -----------------------------------------------------------------------------
# MOODS
# -----------------------------------------------------------------------------

async def add_mood(reading: str, source: str = "sentiment") -> None:
    db = await _get_db()
    await db.execute(
        "INSERT INTO moods (reading, source) VALUES (?, ?)",
        (reading, source),
    )
    await db.commit()


async def recent_moods(limit: int = 10) -> list[dict[str, Any]]:
    db = await _get_db()
    cursor = await db.execute(
        "SELECT * FROM moods ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def latest_mood() -> Optional[dict[str, Any]]:
    rows = await recent_moods(limit=1)
    return rows[0] if rows else None


# -----------------------------------------------------------------------------
# RULES
# -----------------------------------------------------------------------------

DEFAULT_RULES: list[dict[str, Any]] = [
    {
        "name": "morning_briefing",
        "condition": {"hour": 8, "no_chat_today": True},
        "action": "morning_briefing",
        "action_payload": None,
        "cooldown_hours": 12,
    },
    {
        "name": "late_night_nudge",
        "condition": {"hour_min": 23, "chatting_over_hours": 2},
        "action": "gentle_nudge",
        "action_payload": None,
        "cooldown_hours": 4,
    },
    {
        "name": "silence_checkin",
        "condition": {"days_since_chat_gte": 1},
        "action": "check_in",
        "action_payload": None,
        "cooldown_hours": 48,
    },
]


async def list_rules(enabled_only: bool = False) -> list[dict[str, Any]]:
    sql = "SELECT * FROM rules"
    args: tuple = ()
    if enabled_only:
        sql += " WHERE enabled = 1"
    sql += " ORDER BY id"
    db = await _get_db()
    cursor = await db.execute(sql, args)
    rows = await cursor.fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["condition"] = json.loads(d.pop("condition_json"))
        except Exception:
            d["condition"] = {}
        if d.get("action_payload"):
            try:
                d["action_payload"] = json.loads(d["action_payload"])
            except Exception:
                pass
        out.append(d)
    return out


async def add_rule(name: str, condition: dict, action: str,
                   action_payload: Optional[dict] = None,
                   cooldown_hours: float = 12.0) -> int:
    cond = json.dumps(condition)
    payload = json.dumps(action_payload) if action_payload else None
    db = await _get_db()
    cursor = await db.execute(
        """
        INSERT OR REPLACE INTO rules
            (name, condition_json, action, action_payload, cooldown_hours)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, cond, action, payload, cooldown_hours),
    )
    await db.commit()
    return cursor.lastrowid or 0


async def set_rule_enabled(rule_id: int, enabled: bool) -> None:
    db = await _get_db()
    await db.execute(
        "UPDATE rules SET enabled = ? WHERE id = ?",
        (1 if enabled else 0, rule_id),
    )
    await db.commit()


async def delete_rule(rule_id: int) -> None:
    db = await _get_db()
    await db.execute("DELETE FROM rules WHERE id = ?", (rule_id,))
    await db.commit()


async def mark_rule_fired(rule_id: int) -> None:
    db = await _get_db()
    await db.execute(
        "UPDATE rules SET last_fired = CURRENT_TIMESTAMP WHERE id = ?",
        (rule_id,),
    )
    await db.commit()


async def seed_default_rules() -> None:
    """Seed default rules on first run, if not already present."""
    existing = {r["name"] for r in await list_rules()}
    for rule in DEFAULT_RULES:
        if rule["name"] not in existing:
            await add_rule(
                name=rule["name"],
                condition=rule["condition"],
                action=rule["action"],
                action_payload=rule.get("action_payload"),
                cooldown_hours=rule["cooldown_hours"],
            )


# -----------------------------------------------------------------------------
# CONVERSATIONS
# -----------------------------------------------------------------------------

async def add_conversation(role: str, content: str, mood: Optional[str] = None) -> int:
    db = await _get_db()
    cursor = await db.execute(
        "INSERT INTO conversations (role, content, mood) VALUES (?, ?, ?)",
        (role, content, mood),
    )
    await db.commit()
    return cursor.lastrowid or 0


async def recent_conversations(limit: int = 50,
                               role: Optional[str] = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM conversations"
    args: tuple = ()
    if role:
        sql += " WHERE role = ?"
        args = (role,)
    sql += " ORDER BY id DESC LIMIT ?"
    args = args + (limit,)
    db = await _get_db()
    cursor = await db.execute(sql, args)
    rows = await cursor.fetchall()
    return [dict(r) for r in rows][::-1]  # chronological


async def last_interaction_at() -> Optional[datetime]:
    """Return the timestamp of the last conversation as a UTC-aware datetime."""
    db = await _get_db()
    cursor = await db.execute(
        "SELECT timestamp FROM conversations ORDER BY id DESC LIMIT 1"
    )
    row = await cursor.fetchone()
    if not row:
        return None
    # SQLite CURRENT_TIMESTAMP returns 'YYYY-MM-DD HH:MM:SS' in UTC.
    # Return as offset-aware UTC so callers can compare safely.
    from datetime import timezone
    try:
        naive = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
        return naive.replace(tzinfo=timezone.utc)
    except Exception:
        return None


async def clear_conversations() -> None:
    db = await _get_db()
    await db.execute("DELETE FROM conversations")
    await db.commit()


# -----------------------------------------------------------------------------
# Convenience: simple keyword-based fact retrieval for the prompt
# -----------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "my", "your", "his", "their", "our", "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "so", "as", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "about", "what", "when", "where",
    "why", "how", "do", "does", "did", "have", "has", "had", "will",
    "would", "can", "could", "should", "may", "might", "just", "very",
    "really", "ok", "okay", "yeah", "yes", "no", "not", "sir", "please",
    "thanks", "thank",
}


def _keywords(text: str) -> set[str]:
    out = set()
    for w in text.lower().split():
        w = "".join(c for c in w if c.isalnum())
        if w and w not in _STOPWORDS and len(w) > 2:
            out.add(w)
    return out


async def relevant_facts(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Return facts whose key or value shares a word with the query.

    Falls back to most-recently-updated if nothing matches.
    """
    qkeys = _keywords(query)
    if not qkeys:
        return await list_facts(limit=limit)
    all_facts = await list_facts(limit=500)
    scored: list[tuple[int, dict]] = []
    for f in all_facts:
        text = f"{f['key']} {f['value']}".lower()
        score = sum(1 for w in qkeys if w in text)
        if score > 0:
            scored.append((score, f))
    if not scored:
        return all_facts[:limit]
    scored.sort(key=lambda x: -x[0])
    return [f for _, f in scored[:limit]]


async def relevant_patterns(query: str, limit: int = 5) -> list[dict[str, Any]]:
    qkeys = _keywords(query)
    all_p = await list_patterns(min_confidence=0.3, limit=200)
    if not qkeys:
        return all_p[:limit]
    scored: list[tuple[int, dict]] = []
    for p in all_p:
        text = p["description"].lower()
        score = sum(1 for w in qkeys if w in text)
        if score > 0:
            scored.append((score, p))
    if not scored:
        return all_p[:limit]
    scored.sort(key=lambda x: -x[0])


# -----------------------------------------------------------------------------
# Activity log (PC-control audit trail)
# -----------------------------------------------------------------------------

async def log_activity(kind: str, target: str, status: str, detail: str = "") -> int:
    """Record one row. Returns the new row id."""
    db = await _get_db()
    cur = await db.execute(
        "INSERT INTO activity_log (kind, target, status, detail) VALUES (?, ?, ?, ?)",
        (kind, target[:500], status, detail[:1000]),
    )
    await db.commit()
    return cur.lastrowid or 0


async def recent_activity(limit: int = 50, kind: Optional[str] = None) -> list[dict[str, Any]]:
    """Return the most recent activity rows (newest first)."""
    db = await _get_db()
    if kind:
        cur = await db.execute(
            "SELECT id, kind, target, status, detail, timestamp "
            "FROM activity_log WHERE kind = ? ORDER BY id DESC LIMIT ?",
            (kind, limit),
        )
    else:
        cur = await db.execute(
            "SELECT id, kind, target, status, detail, timestamp "
            "FROM activity_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
    rows = await cur.fetchall()
    return [
        {"id": r[0], "kind": r[1], "target": r[2], "status": r[3],
         "detail": r[4], "timestamp": r[5]}
        for r in rows
    ]
    return [p for _, p in scored[:limit]]
