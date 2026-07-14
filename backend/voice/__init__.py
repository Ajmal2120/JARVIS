"""
JARVIS Voice Module

Unified voice system providing:
- Speech-to-Text (STT) using faster-whisper medium + silero VAD (offline)
- Text-to-Speech (TTS) using Kokoro TTS (local, multilingual)
- Wake word detection
- Native audio capture (no browser required)
- Intent detection for zero-latency simple commands
"""

# Lazy imports — heavy models load on first use, not at import time

def is_listening():
    from .stt import is_listening as _is_listening
    return _is_listening()
