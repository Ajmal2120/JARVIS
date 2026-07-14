"""
JARVIS passphrase / PIN authentication (Phase 1).

Stores a hashed passphrase in the SQLite `auth` table. The frontend must
authenticate over the WebSocket before JARVIS will engage. Wrong passphrase
=> connection is closed with a polite JARVIS refusal.

Phase 2 will add voice biometrics on top of this — same flow, more layers.

Uses the same shared connection as `memory.long_term` to avoid the
Python 3.14 + aiosqlite thread-restart issue.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
from typing import Optional

import aiosqlite

from config import settings


# -----------------------------------------------------------------------------
# Hashing
# -----------------------------------------------------------------------------

def _hash(passphrase: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, 200_000)


def _new_salt() -> bytes:
    return secrets.token_bytes(16)


# -----------------------------------------------------------------------------
# Shared connection (delegates to long_term's connection)
# -----------------------------------------------------------------------------

_auth_lock = asyncio.Lock()
_auth_ready = False


async def _get_db() -> aiosqlite.Connection:
    """Get the shared DB connection, ensuring the auth table exists."""
    global _auth_ready
    # Import here to avoid circular imports
    from memory.long_term import _get_db as _lt_get_db
    db = await _lt_get_db()
    async with _auth_lock:
        if not _auth_ready:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS auth (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            await db.commit()
            _auth_ready = True
    return db


async def is_passphrase_set() -> bool:
    """Returns True if a passphrase has been configured."""
    db = await _get_db()
    cursor = await db.execute(
        "SELECT value FROM auth WHERE key = 'passphrase_hash'"
    )
    row = await cursor.fetchone()
    return row is not None


async def set_passphrase(passphrase: str) -> None:
    """
    Set or replace the passphrase.
    A reasonable passphrase: 4-6+ words OR a 6+ digit PIN.
    """
    if not passphrase or not passphrase.strip():
        raise ValueError("Passphrase cannot be empty.")
    db = await _get_db()
    salt = _new_salt()
    digest = _hash(passphrase, salt)
    stored = f"{salt.hex()}${digest.hex()}"
    async with _auth_lock:
        await db.execute(
            "INSERT OR REPLACE INTO auth (key, value) VALUES ('passphrase_hash', ?)",
            (stored,),
        )
        await db.commit()


async def verify_passphrase(passphrase: str) -> bool:
    """Constant-time check. Returns True on match."""
    if not passphrase:
        return False
    db = await _get_db()
    cursor = await db.execute(
        "SELECT value FROM auth WHERE key = 'passphrase_hash'"
    )
    row = await cursor.fetchone()
    if not row:
        # No passphrase set => first run. Treat as "open" until user sets one.
        return True
    stored = row[0]
    try:
        salt_hex, digest_hex = stored.split("$", 1)
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(digest_hex)
    actual = _hash(passphrase, salt)
    return hmac.compare_digest(expected, actual)


async def clear_passphrase() -> None:
    """Remove the passphrase (re-enables open access — for testing only)."""
    db = await _get_db()
    async with _auth_lock:
        await db.execute("DELETE FROM auth WHERE key = 'passphrase_hash'")
        await db.commit()


# -----------------------------------------------------------------------------
# Convenience for tests / CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def _demo():
        if not await is_passphrase_set():
            print("No passphrase set. Setting demo passphrase: 'open sesame'")
            await set_passphrase("open sesame")
        ok = await verify_passphrase("open sesame")
        print("verify('open sesame') ->", ok)
        bad = await verify_passphrase("wrong")
        print("verify('wrong') ->", bad)

    asyncio.run(_demo())
