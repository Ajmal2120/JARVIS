"""
Speech-to-Text functionality for JARVIS.

Uses faster-whisper with custom VAD for offline speech recognition.
"""

import logging
import json
import queue
import threading
import time
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import sounddevice as sd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 8000  # 0.5s at 16kHz

# ---------------------------------------------------------------------------
# Speech detection tuning
# ---------------------------------------------------------------------------

# Minimum speech duration before transcribing (prevents false positives)
MIN_SPEECH_SAMPLES = int(1.0 * SAMPLE_RATE)  # 1.0s - lowered for faster response

# Silence duration that ends an utterance (reduced for quicker response)
MAX_SILENCE_SAMPLES = int(0.8 * SAMPLE_RATE)  # 0.8s - lowered from 1.2s

# RMS threshold below which audio is considered silence.
# Tuned for typical laptop microphone. Music/songs below this are ignored.
SILENCE_THRESHOLD = 0.015

# RMS threshold above which audio is considered loud (music/tv/etc)
# Audio above this threshold clears the buffer to prevent false triggers.
LOUD_THRESHOLD = 0.2

# ---------------------------------------------------------------------------
# Whisper transcription settings
# ---------------------------------------------------------------------------

# Beam size for decoding.
# 1 = fastest, identical accuracy for clear speech.
# Higher values add latency with diminishing returns on CPU.
BEAM_SIZE = 1

# Language for transcription.
# Set to "en" for English-only, None for auto-detect.
VOICE_LANGUAGE = "en"

# ---------------------------------------------------------------------------
# Microphone device persistence
# ---------------------------------------------------------------------------

def _config_path() -> Path:
    """Path to persistent voice config (mic selection, etc.)."""
    return Path(__file__).parent.parent / "memory" / "voice_config.json"


def _load_voice_config() -> dict:
    """Load saved voice configuration."""
    path = _config_path()
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_voice_config(cfg: dict) -> None:
    """Save voice configuration to disk."""
    path = _config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not save voice config: %s", e)


def _persist_mic_device(device_id: int) -> None:
    """Remember the user's microphone selection."""
    cfg = _load_voice_config()
    cfg["mic_device_id"] = device_id
    _save_voice_config(cfg)


def get_saved_mic_device() -> Optional[int]:
    """Return the previously saved microphone device id, or None."""
    cfg = _load_voice_config()
    return cfg.get("mic_device_id")


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_listening = False
_stop_event = threading.Event()
_audio_stream = None
_transcribe_thread = None
_model = None
_device = None
_language = None  # Set from VOICE_LANGUAGE at load time

# Queues
_audio_queue: queue.Queue = queue.Queue(maxsize=50)  # Bounded: ~6s of audio at 0.5s/chunk
_result_queue: queue.Queue = queue.Queue(maxsize=20)  # Bounded: results
_audio_dropped = 0  # Counter for overflow diagnostics


# ---------------------------------------------------------------------------
# Device / model helpers
# ---------------------------------------------------------------------------

def _get_compute_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    try:
        import torch  # noqa: F401
        # Only use CUDA if we actually have a GPU
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _get_model_size() -> str:
    """Return configured Whisper model size."""
    try:
        from config import settings
        size = getattr(settings, "STT_MODEL", None)
        if size:
            return size
    except Exception:
        pass
    return "small"


def _load_model():
    """Load faster-whisper model (lazy, one-time initialization)."""
    global _model, _device, _language

    if _model is not None:
        return

    _device = _get_compute_device()
    model_size = _get_model_size()
    _language = VOICE_LANGUAGE or None  # None triggers auto-detect

    logger.info("[STT] Loading faster-whisper model (%s) on %s...", model_size, _device)

    from faster_whisper import WhisperModel
    _model = WhisperModel(
        model_size,
        device=_device,
        compute_type="int8" if _device == "cpu" else "float16",
        num_workers=1,  # Single thread = lower latency
    )
    lang_info = _language or "auto-detect"
    logger.info("[STT] Model loaded: %s (%s, language=%s)", model_size, _device, lang_info)


# ---------------------------------------------------------------------------
# Audio callbacks
# ---------------------------------------------------------------------------

def _push_audio(chunk) -> bool:
    """Push audio chunk into the processing queue with overflow protection.

    Returns True if the chunk was accepted, False if it was dropped.
    """
    global _audio_dropped
    try:
        _audio_queue.put_nowait(chunk)
        return True
    except queue.Full:
        _audio_dropped += 1
        try:
            # Drop oldest, make room for newest
            _audio_queue.get_nowait()
            _audio_queue.put_nowait(chunk)
            return True
        except queue.Empty:
            return False


def _audio_callback(indata, frames, time_info, status):
    """Callback for sounddevice — pushes audio chunks into the processing queue."""
    if _listening and not _stop_event.is_set():
        chunk = indata.flatten().copy()
        _push_audio(chunk)


# ---------------------------------------------------------------------------
# Core transcription worker
# ---------------------------------------------------------------------------

def _transcribe_worker():
    """Background thread: accumulate speech, transcribe complete utterances."""
    logger.info("[STT] Transcription thread started")

    speech_buffer = []
    silence_samples = 0

    while not _stop_event.is_set():
        try:
            chunk = _audio_queue.get(timeout=0.2)

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            is_silent = rms < SILENCE_THRESHOLD
            is_loud = rms > LOUD_THRESHOLD

            if is_loud:
                # Very loud audio (music, TV, clapping) — discard buffer
                # This prevents false wake/command triggers from non-speech audio
                logger.debug("[STT] Loud audio detected (RMS=%.6f) — clearing buffer", rms)
                speech_buffer.clear()
                silence_samples = 0
                continue

            if not is_silent:
                # Speech detected
                speech_buffer.extend(chunk)
                silence_samples = 0
                logger.debug("[STT] Speech detected (RMS=%.6f, buffer=%.2fs)",
                             rms, len(speech_buffer) / SAMPLE_RATE)
            elif speech_buffer:
                # Silence after speech — count down
                silence_samples += len(chunk)

                if silence_samples >= MAX_SILENCE_SAMPLES:
                    # Utterance complete — transcribe
                    _process_utterance(speech_buffer)
                    speech_buffer = []
                    silence_samples = 0

        except queue.Empty:
            # Timeout — flush any buffered speech
            if speech_buffer and len(speech_buffer) >= MIN_SPEECH_SAMPLES:
                logger.debug("[STT] Timeout flush (buffer=%.2fs)",
                             len(speech_buffer) / SAMPLE_RATE)
                _process_utterance(speech_buffer)
                speech_buffer = []
                silence_samples = 0
            continue

        except Exception as e:
            logger.error("[STT] Worker error: %s", e)

    # Drain remaining speech on shutdown
    if speech_buffer and len(speech_buffer) >= MIN_SPEECH_SAMPLES:
        try:
            _process_utterance(speech_buffer)
        except Exception:
            pass

    logger.info("[STT] Transcription thread stopped")


def _process_utterance(speech_buffer: list):
    """Transcribe a buffered utterance and publish the result."""
    audio = np.array(speech_buffer, dtype=np.float32)
    duration = len(audio) / SAMPLE_RATE
    logger.info("[STT] Speech ended (%.2fs) — transcribing...", duration)

    try:
        _transcribe_audio(audio)
    except Exception as e:
        logger.error("[STT] Transcription failed: %s", e)


def _transcribe_audio(audio_array: np.ndarray):
    """Run faster-whisper on the given audio array."""
    if len(audio_array) < int(0.5 * SAMPLE_RATE):
        logger.debug("[STT] Audio too short (%.2fs), skipping", len(audio_array) / SAMPLE_RATE)
        return

    try:
        segments, info = _model.transcribe(
            audio_array,
            language=_language,
            beam_size=BEAM_SIZE,
            vad_filter=False,  # Custom RMS detector used instead
            without_timestamps=True,
        )

        text_parts = []
        for segment in segments:
            if segment.no_speech_prob > 0.8:
                continue
            text = segment.text.strip()
            if text:
                text_parts.append(text)

        result_text = " ".join(text_parts).strip()
        if result_text:
            logger.info("[STT] Recognized: \"%s\"", result_text)
            try:
                _result_queue.put_nowait(result_text)
            except queue.Full:
                global _audio_dropped
                _audio_dropped += 1
                logger.warning("[STT] Recognition queue full — dropping result")
        else:
            logger.debug("[STT] No speech detected in transcription")

    except Exception as e:
        logger.error("[STT] Transcription error: %s", e)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_listening() -> bool:
    """Start continuous speech listening on the configured microphone."""
    global _listening, _audio_stream, _transcribe_thread, _stop_event

    if _listening:
        logger.warning("[STT] Already listening")
        return False

    try:
        _load_model()
    except Exception as e:
        logger.error("[STT] Failed to load model: %s", e)
        return False

    _stop_event.clear()
    _listening = True

    # Select microphone
    _select_microphone()

    # Start audio capture
    try:
        _audio_stream = sd.InputStream(
            callback=_audio_callback,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            dtype="float32",
        )
        _audio_stream.start()
    except Exception as e:
        logger.error("[STT] Failed to start audio stream: %s", e)
        _listening = False
        return False

    # Start transcription thread
    _transcribe_thread = threading.Thread(target=_transcribe_worker, daemon=True)
    _transcribe_thread.start()

    global _audio_dropped
    logger.info("[STT] Speech recognition started (model=%s, device=%s)", _get_model_size(), _device)
    return True


def restart_listening() -> bool:
    """Force-stop and restart STT. Used by the health watchdog for recovery."""
    global _listening, _audio_stream, _transcribe_thread, _stop_event

    logger.warning("[STT] Restarting speech recognition...")

    # Force-stop everything
    _listening = False
    _stop_event.set()

    if _audio_stream:
        try:
            _audio_stream.stop()
            _audio_stream.close()
        except Exception:
            pass
        _audio_stream = None

    if _transcribe_thread and _transcribe_thread.is_alive():
        _transcribe_thread.join(timeout=3.0)
        if _transcribe_thread.is_alive():
            logger.warning("[STT] Transcription thread refused to stop — proceeding anyway")
    _transcribe_thread = None

    # Small delay to let resources settle
    time.sleep(0.5)

    # Drain queues
    global _audio_dropped
    while not _audio_queue.empty():
        try:
            _audio_queue.get_nowait()
        except queue.Empty:
            break
    _audio_dropped = 0
    _stop_event.clear()

    # Restart
    return start_listening()


def get_audio_drop_count() -> int:
    """Return how many audio chunks were dropped due to queue overflow."""
    global _audio_dropped
    return _audio_dropped


def _select_microphone():
    """Select the best available microphone, preferring saved preference."""
    try:
        devices = get_audio_devices()
        if not devices:
            logger.warning("[STT] No audio input devices found")
            return

        saved_id = get_saved_mic_device()
        selected_device = None
        selected_name = None

        # 1. Try saved preference first
        if saved_id is not None:
            for d in devices:
                if d["index"] == saved_id:
                    selected_device = d["index"]
                    selected_name = d["name"]
                    break

        # 2. Fall back to first available device
        if selected_device is None:
            selected_device = devices[0]["index"]
            selected_name = devices[0]["name"]

        set_audio_device(selected_device)
        logger.info("[STT] Microphone: %s (id=%d)", selected_name, selected_device)

    except Exception as e:
        logger.warning("[STT] Could not select microphone: %s", e)


def stop_listening() -> None:
    """Stop continuous speech listening."""
    global _listening, _audio_stream, _transcribe_thread, _stop_event

    if not _listening:
        return

    logger.info("[STT] Stopping speech recognition...")
    _listening = False
    _stop_event.set()

    # Stop audio stream
    if _audio_stream:
        try:
            _audio_stream.stop()
            _audio_stream.close()
        except Exception:
            pass
        _audio_stream = None

    # Wait for thread to finish (force kill after timeout)
    if _transcribe_thread and _transcribe_thread.is_alive():
        _transcribe_thread.join(timeout=5.0)
        if _transcribe_thread.is_alive():
            logger.error("[STT] Transcription thread did not stop cleanly — resource leak possible")

    _transcribe_thread = None
    logger.info("[STT] Speech recognition stopped")


def is_listening() -> bool:
    """Check if speech recognition is active."""
    return _listening


def is_transcribe_thread_healthy() -> bool:
    """Check if the transcription thread is alive and listening."""
    if not _listening:
        return False
    if _transcribe_thread is None:
        return False
    return _transcribe_thread.is_alive()


def listen(timeout: float = 5.0) -> Optional[str]:
    """Listen for speech and return the transcript."""
    if not _listening:
        if not start_listening():
            return None

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            text = _result_queue.get(timeout=0.2)
            return text
        except queue.Empty:
            continue
    return None


def listen_continuous() -> Generator[str, None, None]:
    """Yield speech results continuously (complete utterances only)."""
    if not _listening:
        if not start_listening():
            return

    try:
        while _listening:
            try:
                text = _result_queue.get(timeout=0.5)
                if text:
                    yield text
            except queue.Empty:
                continue
    finally:
        pass  # coordinator manages lifecycle


def get_audio_devices() -> list:
    """Get available audio input devices with their indices."""
    try:
        devices = sd.query_devices()
        input_devices = []
        for i, d in enumerate(devices):
            if hasattr(d, "max_input_channels") and d.max_input_channels > 0:
                input_devices.append({
                    "index": i,
                    "name": d["name"],
                    "max_input_channels": d["max_input_channels"],
                })
        return input_devices
    except Exception as e:
        logger.error("[STT] Failed to query audio devices: %s", e)
        return []


def set_audio_device(device_id: int) -> bool:
    """Set the audio input device and persist the choice."""
    try:
        sd.default.device[0] = device_id
        _persist_mic_device(device_id)
        return True
    except Exception as e:
        logger.error("[STT] Failed to set audio device: %s", e)
        return False
