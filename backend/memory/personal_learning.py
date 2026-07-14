"""
Personal Learning System for JARVIS.

Persists learned information about the user:
- Pronunciation variations
- Application names
- Preferred phrasing
- Contact names
- Repeated corrections
- User habits
"""

import asyncio
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "personal_learning.db"


@dataclass
class LearnedItem:
    """A learned piece of information."""
    id: Optional[int]
    category: str  # 'pronunciation', 'vocabulary', 'phrasing', 'contact', 'correction', 'habit'
    original: str  # What the user said / typed
    learned: str   # What it means / how JARVIS should interpret it
    confidence: float  # 0.0 - 1.0
    use_count: int
    created_at: str
    updated_at: str


class PersonalLearningDB:
    """SQLite database for personal learning."""

    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._conn = None
            self._initialized = True

    async def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._ensure_tables()
        return self._conn

    def _ensure_tables(self):
        """Create tables if they don't exist."""
        cursor = self._conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learned_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                original TEXT NOT NULL,
                learned TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                use_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON learned_items(category)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_original ON learned_items(original)
        """)

        self._conn.commit()

    async def learn(self, category: str, original: str, learned: str, confidence: float = 0.5) -> int:
        """Learn a new association."""
        conn = await self._get_conn()
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        # Check if this already exists
        cursor.execute(
            "SELECT id, use_count FROM learned_items WHERE original = ? AND category = ?",
            (original.lower(), category)
        )
        existing = cursor.fetchone()

        if existing:
            # Update existing entry
            cursor.execute("""
                UPDATE learned_items
                SET learned = ?, confidence = ?, use_count = ?, updated_at = ?
                WHERE id = ?
            """, (learned, confidence, existing[1] + 1, now, existing[0]))
            item_id = existing[0]
        else:
            # Insert new entry
            cursor.execute("""
                INSERT INTO learned_items (category, original, learned, confidence, use_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (category, original.lower(), learned, confidence, 1, now, now))
            item_id = cursor.lastrowid

        conn.commit()
        logger.debug(f"Learned: [{category}] '{original}' → '{learned}'")
        return item_id

    async def lookup(self, category: str, original: str) -> Optional[str]:
        """Look up a learned interpretation."""
        conn = await self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT learned, confidence FROM learned_items
            WHERE original = ? AND category = ?
            ORDER BY confidence DESC, use_count DESC
            LIMIT 1
        """, (original.lower(), category))

        row = cursor.fetchone()
        if row:
            # Increment use count
            cursor.execute("""
                UPDATE learned_items SET use_count = use_count + 1 WHERE original = ? AND category = ?
            """, (original.lower(), category))
            conn.commit()
            return row[0]
        return None

    async def get_all_for_category(self, category: str) -> list[LearnedItem]:
        """Get all learned items for a category."""
        conn = await self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM learned_items
            WHERE category = ?
            ORDER BY confidence DESC, use_count DESC
        """, (category,))

        return [LearnedItem(**dict(row)) for row in cursor.fetchall()]

    async def get_recent_corrections(self, limit: int = 10) -> list[LearnedItem]:
        """Get recent corrections made by the user."""
        return await self.get_recent_by_category('correction', limit)

    async def get_recent_by_category(self, category: str, limit: int = 10) -> list[LearnedItem]:
        """Get recent learned items for a category."""
        conn = await self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM learned_items
            WHERE category = ?
            ORDER BY updated_at DESC
            LIMIT ?
        """, (category, limit))

        return [LearnedItem(**dict(row)) for row in cursor.fetchall()]

    async def get_vocabulary(self) -> dict[str, str]:
        """Get all vocabulary mappings."""
        conn = await self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT original, learned FROM learned_items
            WHERE category = 'vocabulary' OR category = 'pronunciation'
            ORDER BY use_count DESC
        """)

        return {row[0]: row[1] for row in cursor.fetchall()}

    async def record_correction(self, wrong: str, correct: str) -> int:
        """Record a user correction - important for learning."""
        return await self.learn('correction', wrong, correct, confidence=0.9)

    async def learn_pronunciation(self, heard: str, actual: str) -> int:
        """Learn a pronunciation variation."""
        return await self.learn('pronunciation', heard, actual, confidence=0.8)

    async def learn_vocabulary(self, phrase: str, meaning: str) -> int:
        """Learn a user's phrase meaning."""
        return await self.learn('vocabulary', phrase, meaning, confidence=0.7)

    async def learn_contact(self, nickname: str, actual_name: str) -> int:
        """Learn a contact name mapping."""
        return await self.learn('contact', nickname, actual_name, confidence=0.9)

    async def learn_app_name(self, variation: str, actual_name: str) -> int:
        """Learn an application name variation."""
        return await self.learn('app_name', variation, actual_name, confidence=0.9)


# Convenience functions
_learner = PersonalLearningDB()


async def learn(category: str, original: str, learned: str, confidence: float = 0.5) -> int:
    """Learn a new association."""
    return await _learner.learn(category, original, learned, confidence)


async def lookup(category: str, original: str) -> Optional[str]:
    """Look up a learned interpretation."""
    return await _learner.lookup(category, original)


async def apply_learning(text: str) -> str:
    """
    Apply learned interpretations to user input.
    This is called before intent detection to normalize user input.
    """
    # Split into words/phrases
    words = text.lower().split()
    result = text

    # Check full phrase first
    full_lookup = await lookup('vocabulary', text)
    if full_lookup:
        result = full_lookup

    # Check each word
    for word in words:
        word_lookup = await lookup('vocabulary', word)
        if word_lookup:
            result = result.replace(word, word_lookup)

    # Check pronunciation variations
    for word in words:
        pron_lookup = await lookup('pronunciation', word)
        if pron_lookup:
            result = result.replace(word, pron_lookup)

    return result


async def get_user_vocabulary() -> dict[str, str]:
    """Get all user's learned vocabulary."""
    return await _learner.get_vocabulary()


async def record_correction(wrong: str, correct: str) -> int:
    """Record a user correction."""
    return await _learner.record_correction(wrong, correct)


async def learn_from_correction(wrong: str, correct: str):
    """
    Learn from a user correction.
    This teaches JARVIS to not make the same mistake again.
    """
    # If JARVIS misunderstood, record the correction
    await record_correction(wrong, correct)
    logger.info(f"Learned from correction: '{wrong}' → '{correct}'")