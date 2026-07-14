"""
Text-to-Speech for JARVIS.

Uses tts/speaker.py which provides automatic fallback:
1. Kokoro TTS (if model files present)
2. SAPI TTS (Windows built-in, no downloads)
3. PowerShell TTS (last resort)

Public API:
    synthesize_bytes(text, voice=None) -> bytes
    speak_async(text, voice=None) -> None
"""

from __future__ import annotations

import logging
from typing import Optional

from tts.speaker import speak_async as _speak_async_impl, synthesize_bytes as _synthesize

logger = logging.getLogger(__name__)


def speak_async(text: str, voice: Optional[str] = None) -> None:
    """
    Speak text asynchronously (non-blocking).
    Uses the best available TTS engine.
    """
    if not text or not text.strip():
        return
    _speak_async_impl(text, voice)


def synthesize_bytes(text: str, voice: Optional[str] = None) -> bytes:
    """
    Generate audio bytes from text.
    Returns raw audio data for streaming/encoding.
    """
    if not text or not text.strip():
        return b""
    return _synthesize(text, voice)
