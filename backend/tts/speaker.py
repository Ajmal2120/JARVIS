"""
Text-to-Speech for J.A.R.V.I.S.

Priority:
1. Kokoro TTS (local, fast, multilingual) - if model files present
2. SAPI TTS (Windows built-in, no downloads needed)
3. PowerShell fallback (last resort)

Public API:
    speak(text, voice=None) -> bool
    speak_async(text, voice=None) -> None
    synthesize_bytes(text, voice=None) -> bytes
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TTS Engine Selection
# ---------------------------------------------------------------------------

class TTSEngine:
    """Abstraction for different TTS backends."""

    def speak(self, text: str) -> bool:
        raise NotImplementedError

    def speak_async(self, text: str):
        raise NotImplementedError

    def synthesize(self, text: str) -> bytes:
        raise NotImplementedError


class KokoroTTS(TTSEngine):
    """Kokoro ONNX TTS - fastest and best quality."""

    # Voice presets for different languages
    ENGLISH_VOICE = "bf_vale"   # British male voice (male!)
    CHINESE_VOICE = "zf_001"    # Chinese female voice (if model supports it)
    HINDI_VOICE = "bf_vale"     # Use British voice as fallback for Hindi

    def __init__(self):
        self._engine = None
        self._available_voices = []
        self._model_path = None
        self._voices_path = None
        self._sapi_fallback = None  # Cached SAPI fallback engine

    def _get_sapi_fallback(self):
        """Lazily create a cached SAPI fallback engine."""
        if self._sapi_fallback is None:
            try:
                self._sapi_fallback = SAPITTS()
            except Exception as e:
                logger.warning("[TTS] Could not create SAPI fallback: %s", e)
        return self._sapi_fallback

    def _load(self):
        if self._engine is not None:
            return True

        try:
            from kokoro_onnx import Kokoro
        except ImportError:
            logger.warning("kokoro-onnx not installed")
            return False

        # Get model directory relative to this file
        model_dir = Path(__file__).parent.parent / "kokoro"

        # Look for model files
        onnx_path = model_dir / "kokoro-v1.1-zh.onnx"
        voices_path = model_dir / "voices-v1.1-zh.bin"

        if not onnx_path.exists():
            # Try alternate names
            for name in ["kokoro.onnx", "kokoro-v1.0.onnx", "kokoro-v1.1.onnx"]:
                alt = model_dir / name
                if alt.exists():
                    onnx_path = alt
                    break

        if not voices_path.exists():
            for name in ["voices.bin", "voices.json"]:
                alt = model_dir / name
                if alt.exists():
                    voices_path = alt
                    break

        if not onnx_path.exists():
            logger.warning("Kokoro model not found at %s", onnx_path)
            return False

        try:
            logger.info("[TTS] Loading Kokoro model from %s", onnx_path)
            self._engine = Kokoro(str(onnx_path), str(voices_path))
            self._available_voices = self._engine.get_voices()
            self._model_path = onnx_path
            self._voices_path = voices_path
            logger.info("[TTS] Kokoro loaded with %d voices", len(self._available_voices))
            logger.info("[TTS] Available voices: %s", ", ".join(self._available_voices[:5]))
            return True
        except Exception as e:
            logger.error("[TTS] Failed to load Kokoro: %s", e)
            self._engine = None
            return False

    def _detect_lang(self, text: str) -> str:
        """Detect language from text."""
        # Check for Hindi/Devanagari script
        for ch in text:
            if 0x0900 <= ord(ch) <= 0x097F:
                return "hi"
        # Check for Chinese characters
        for ch in text:
            if 0x4E00 <= ord(ch) <= 0x9FFF:
                return "zh"
        # Check for Hindi words
        hi_words = {"namaste", "kaise", "kya", "hai", "nahi", "haan", "thik", "aap", "main", "batao", "dekho", "suno"}
        tokens = text.lower().split()
        return "hi" if sum(1 for t in tokens if t.strip(".,!?") in hi_words) >= 2 else "en"

    def _get_voice(self, text: str) -> tuple[str, str]:
        """Get appropriate voice and language. Returns (voice, lang_code)."""
        lang = self._detect_lang(text)
        if lang == "zh" and "zf_001" in self._available_voices:
            return self.CHINESE_VOICE, "zh"
        elif lang == "hi":
            return self.HINDI_VOICE, "en-us"  # British voice works for Hindi
        return self.ENGLISH_VOICE, "en-us"

    def _speak_via_sapi(self, text: str) -> bool:
        """Fallback: speak via SAPI when Kokoro fails."""
        logger.info("[TTS] [Fallback] SAPI: \"%s\"", text[:50])
        try:
            sapi = self._get_sapi_fallback()
            if sapi is None:
                logger.error("[TTS] [Fallback] No SAPI engine available")
                return False
            result = sapi.speak(text)
            if result:
                logger.info("[TTS] [Fallback] SAPI playback complete")
            else:
                logger.error("[TTS] [Fallback] SAPI returned False")
            return result
        except Exception as e:
            logger.error("[TTS] [Fallback] SAPI failed: %s", e)
            return False

    def speak(self, text: str) -> bool:
        if not text.strip():
            logger.warning("[TTS] Empty text, skipping")
            return False

        logger.info("[TTS] Kokoro speak requested: \"%s\"", text[:50])

        if not self._load():
            logger.error("[TTS] Kokoro engine not available — trying SAPI fallback")
            return self._speak_via_sapi(text)

        try:
            voice, lang = self._get_voice(text)
            logger.info("[TTS] Voice: %s, Language: %s", voice, lang)

            logger.info("[TTS] Synthesizing speech...")
            samples, sr = self._engine.create(text.strip(), voice=voice, lang=lang, speed=1.0)

            if samples is None or len(samples) == 0:
                logger.error("[TTS] Kokoro returned empty audio — trying SAPI fallback")
                return self._speak_via_sapi(text)

            samples = np.asarray(samples).flatten().astype(np.float32)
            logger.info("[TTS] Generated %d samples at %dHz (%.2fs)",
                        len(samples), sr, len(samples) / sr)

            # Check for interruption before playing
            if _speech_aborted.is_set():
                logger.info("[TTS] Speech aborted before playback")
                _speech_aborted.clear()
                return False

            # Select audio output device
            device = _get_output_device()
            logger.info("[TTS] Playing audio on device %s", device)

            logger.info("[TTS] Playback started")
            sd.play(samples, samplerate=sr, blocking=True, device=device)

            # Check for interruption after playing
            if _speech_aborted.is_set():
                logger.info("[TTS] Speech interrupted during playback")
                _speech_aborted.clear()
                sd.stop()
                return False

            logger.info("[TTS] Playback complete")
            return True

        except Exception as e:
            error_msg = str(e).lower()
            if "abort" in error_msg:
                logger.info("[TTS] Speech aborted")
                return False
            logger.warning("[TTS] Kokoro playback failed: %s — trying SAPI fallback", e)
            return self._speak_via_sapi(text)

    def speak_async(self, text: str):
        """Run blocking speak() in a background thread with health monitoring."""
        global _speaking_thread, _speech_aborted

        _speech_aborted.clear()

        # Create thread with error reporting
        def _speak_wrapper():
            try:
                success = self.speak(text)
                if not success:
                    logger.error("[TTS] Async speech failed for: \"%s\"", text[:50])
            except Exception as e:
                logger.error("[TTS] Async speech exception: %s", e, exc_info=True)

        _speaking_thread = threading.Thread(target=_speak_wrapper, daemon=True)
        _speaking_thread.start()
        logger.info("[TTS] Started background thread for: \"%s\"", text[:50])

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""

        if not self._load():
            logger.error("[TTS] Cannot synthesize - Kokoro not available")
            return b""

        try:
            voice, lang = self._get_voice(text)
            samples, sr = self._engine.create(text.strip(), voice=voice, lang=lang, speed=1.0)

            if samples is None or len(samples) == 0:
                logger.error("[TTS] Kokoro returned empty audio for synthesis")
                return b""

            # Convert float32 samples to WAV bytes
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sr)
                audio_int16 = (np.asarray(samples).flatten() * 32767).astype(np.int16)
                wav.writeframes(audio_int16.tobytes())

            logger.info("[TTS] Synthesized %d bytes of audio", len(buf.getvalue()))
            return buf.getvalue()

        except Exception as e:
            logger.error("[TTS] Kokoro synthesize failed: %s", e, exc_info=True)
            return b""


class SAPITTS(TTSEngine):
    """Windows SAPI TTS - works without any downloads."""

    def speak(self, text: str) -> bool:
        if not text.strip():
            return False

        logger.info("[TTS] SAPI speak requested: \"%s\"", text[:50])

        try:
            import pyttsx3

            # Create fresh engine per call to avoid threading issues
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)

            voices = engine.getProperty('voices')
            if voices:
                male_voice = None
                for voice in voices:
                    name = voice.name.lower()
                    if 'male' in name or 'david' in name or 'guy' in name:
                        male_voice = voice
                        break
                if male_voice is None:
                    male_voice = voices[0]
                engine.setProperty('voice', male_voice.id)
                logger.info("[TTS] SAPI voice: %s", male_voice.name)

            engine.say(text)
            engine.runAndWait()
            logger.info("[TTS] SAPI playback complete")
            return True

        except Exception as e:
            logger.error("[TTS] SAPI speak failed: %s", e, exc_info=True)
            return False

    def speak_async(self, text: str):
        """Run blocking speak() in a background thread with error reporting."""
        global _speaking_thread, _speech_aborted

        _speech_aborted.clear()

        def _speak_wrapper():
            try:
                success = self.speak(text)
                if not success:
                    logger.error("[TTS] SAPI async speech failed for: \"%s\"", text[:50])
            except Exception as e:
                logger.error("[TTS] SAPI async speech exception: %s", e, exc_info=True)

        _speaking_thread = threading.Thread(target=_speak_wrapper, daemon=True)
        _speaking_thread.start()
        logger.info("[TTS] Started SAPI background thread for: \"%s\"", text[:50])

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to WAV bytes using a fresh SAPI engine."""
        if not text.strip():
            return b""

        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[0].id)
            engine.save_to_file(text, '_sapi_temp.wav')
            engine.runAndWait()
            try:
                with open('_sapi_temp.wav', 'rb') as f:
                    data = f.read()
            finally:
                Path('_sapi_temp.wav').unlink(missing_ok=True)
            logger.info("[TTS] SAPI synthesized %d bytes", len(data))
            return data
        except Exception as e:
            logger.error("[TTS] SAPI synthesize failed: %s", e, exc_info=True)
            return b""


class PowerShellTTS(TTSEngine):
    """PowerShell System.Speech fallback - last resort."""

    def speak(self, text: str) -> bool:
        if not text.strip():
            return False

        logger.info("[TTS] PowerShell speak requested: \"%s\"", text[:50])

        try:
            # Use PowerShell to call .NET Speech Synthesis
            # Escape double quotes for PowerShell
            escaped = text.replace('"', '`"')
            ps_script = f'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Speak("{escaped}")
'''
            result = subprocess.run(
                ['powershell', '-NoProfile', '-Command', ps_script],
                capture_output=True,
                timeout=30
            )

            if result.returncode == 0:
                logger.info("[TTS] PowerShell playback complete")
            else:
                logger.error("[TTS] PowerShell failed: %s", result.stderr.decode())

            return result.returncode == 0

        except Exception as e:
            logger.error("[TTS] PowerShell TTS failed: %s", e, exc_info=True)
            return False

    def speak_async(self, text: str):
        """Run blocking speak() in a background thread with error reporting."""
        global _speaking_thread, _speech_aborted

        _speech_aborted.clear()

        def _speak_wrapper():
            try:
                success = self.speak(text)
                if not success:
                    logger.error("[TTS] PowerShell async speech failed for: \"%s\"", text[:50])
            except Exception as e:
                logger.error("[TTS] PowerShell async speech exception: %s", e, exc_info=True)

        _speaking_thread = threading.Thread(target=_speak_wrapper, daemon=True)
        _speaking_thread.start()
        logger.info("[TTS] Started PowerShell background thread for: \"%s\"", text[:50])

    def synthesize(self, text: str) -> bytes:
        """Not implemented for PowerShell fallback."""
        return b""


# ---------------------------------------------------------------------------
# Audio output device selection
# ---------------------------------------------------------------------------

def _get_output_device() -> Optional[str | int]:
    """Get the default audio output device."""
    try:
        # Use default output device
        return None  # None = system default
    except Exception as e:
        logger.warning("[TTS] Could not get output device: %s", e)
        return None


# ---------------------------------------------------------------------------
# Thread health monitoring
# ---------------------------------------------------------------------------

_speaking_thread: Optional[threading.Thread] = None
_speech_aborted = threading.Event()
_speak_lock = threading.Lock()
_last_speak_success: bool = False
_last_speak_error: Optional[str] = None


def interrupt_speech() -> bool:
    """
    Interrupt any currently playing speech.

    Returns True if speech was interrupted, False if nothing was playing.
    """
    global _speaking_thread, _speech_aborted

    if _speaking_thread and _speaking_thread.is_alive():
        _speech_aborted.set()
        logger.info("[TTS] Speech interrupted by user")
        # Give thread a moment to stop
        _speaking_thread.join(timeout=0.5)
        return True

    # Also try to stop sounddevice
    try:
        sd.stop()
    except Exception:
        pass

    return False


def is_speaking() -> bool:
    """Check if JARVIS is currently speaking."""
    global _speaking_thread

    # Check thread health
    if _speaking_thread and _speaking_thread.is_alive():
        return True

    # Thread died silently
    if _speaking_thread and not _speaking_thread.is_alive():
        logger.warning("[TTS] Speech thread died silently")
        _speaking_thread = None

    return False


def get_last_speak_status() -> tuple[bool, Optional[str]]:
    """Return (success, error_message) from the last speak call."""
    return _last_speak_success, _last_speak_error


# ---------------------------------------------------------------------------
# Global TTS Engine (Kokoro -> SAPI -> PowerShell)
# ---------------------------------------------------------------------------

_tts_engine: Optional[TTSEngine] = None
_tts_engine_name: str = "none"


def _get_engine() -> TTSEngine:
    """Get the best available TTS engine - Kokoro first, then SAPI."""
    global _tts_engine, _tts_engine_name

    if _tts_engine is not None:
        return _tts_engine

    # Try Kokoro first
    engine = KokoroTTS()
    if engine._load():
        _tts_engine = engine
        _tts_engine_name = "Kokoro"
        logger.info("[TTS] Engine selected: Kokoro")
        return engine

    # Fall back to SAPI
    logger.warning("[TTS] Kokoro not available, falling back to SAPI")
    engine = SAPITTS()
    if engine.speak(""):  # Test if SAPI works
        _tts_engine = engine
        _tts_engine_name = "SAPI"
        logger.info("[TTS] Engine selected: SAPI (fallback)")
        return engine

    # PowerShell fallback
    logger.warning("[TTS] SAPI not available, falling back to PowerShell")
    _tts_engine = PowerShellTTS()
    _tts_engine_name = "PowerShell"
    logger.info("[TTS] Engine selected: PowerShell (last resort)")
    return _tts_engine


def get_engine_name() -> str:
    """Return the name of the currently active TTS engine."""
    return _tts_engine_name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def speak(text: str, voice: Optional[str] = None) -> bool:
    """Speak text synchronously (blocking)."""
    global _last_speak_success, _last_speak_error

    if not text or not text.strip():
        logger.warning("[TTS] speak() called with empty text")
        return False

    logger.info("[TTS] speak() called: \"%s\"", text[:50])
    engine = _get_engine()

    try:
        with _speak_lock:
            success = engine.speak(text)
            _last_speak_success = success
            _last_speak_error = None if success else f"{type(engine).__name__} returned False"

            if not success:
                logger.error("[TTS] speak() failed with %s", type(engine).__name__)

            return success
    except Exception as e:
        logger.error("[TTS] speak() exception: %s", e, exc_info=True)
        _last_speak_success = False
        _last_speak_error = str(e)
        return False


def speak_async(text: str, voice: Optional[str] = None):
    """Speak text asynchronously (non-blocking)."""
    if not text or not text.strip():
        logger.warning("[TTS] speak_async() called with empty text")
        return

    engine = _get_engine()
    engine.speak_async(text)


def interrupt() -> bool:
    """Interrupt current speech. Returns True if speech was interrupted."""
    return interrupt_speech()


def speaking() -> bool:
    """Check if currently speaking."""
    return is_speaking()


def synthesize_bytes(text: str, voice: Optional[str] = None) -> bytes:
    """Generate audio bytes. Returns empty bytes if not supported."""
    if not text or not text.strip():
        return b""
    return _get_engine().synthesize(text)


# Legacy API compat
VOICES = {
    "af_heart": "Default (English)",
    "af_sky": "Alternate (English)",
    "hi_0": "Hindi",
}

MODULATION = {}


def synthesize(text: str, voice: Optional[str] = None, rate: Optional[str] = None, pitch: Optional[str] = None) -> Path:
    """Legacy API - returns a dummy path since we don't cache."""
    return settings.TTS_CACHE_DIR / "nocache"


async def synthesize_async(text: str, voice: Optional[str] = None) -> bytes:
    """Async wrapper for synthesize_bytes."""
    return synthesize_bytes(text, voice)


async def warm_cache(phrases: list[str]) -> int:
    """No-op - TTS is fast enough on-demand."""
    return 0


def list_voices() -> dict:
    return dict(VOICES)


def list_modulations() -> dict:
    return dict(MODULATION)
