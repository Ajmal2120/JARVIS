"""
Health monitoring and recovery for the voice pipeline.

Provides:
- Health checks for each voice pipeline component
- Automatic recovery of failed threads
- Pipeline status reporting
- Watchdog timer for overall system health
"""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# How often the watchdog checks component health
WATCHDOG_INTERVAL_SEC = 30


def check_stt_health() -> dict:
    """Check STT health. Returns dict with status details."""
    try:
        from voice.stt import is_listening, is_transcribe_thread_healthy, _stop_event, _transcribe_thread

        listening = is_listening()
        thread_ok = is_transcribe_thread_healthy() if listening else "not_listening"
        stopped = _stop_event.is_set() if listening else False

        healthy = listening and thread_ok is True and not stopped

        return {
            "healthy": healthy,
            "listening": listening,
            "thread_healthy": thread_ok,
            "stopped": stopped,
        }
    except Exception as e:
        logger.error("[Health] STT check failed: %s", e)
        return {"healthy": False, "error": str(e)}


def check_tts_health() -> dict:
    """Check TTS health. Returns dict with status details."""
    try:
        from tts.speaker import (
            is_speaking, get_engine_name,
            _last_speak_success, _last_speak_error,
            _speaking_thread,
        )

        engine = get_engine_name()
        speaking = is_speaking()
        last_ok = _last_speak_success
        last_err = _last_speak_error

        return {
            "healthy": True,  # TTS is healthy if it can report status
            "engine": engine,
            "speaking": speaking,
            "last_speak_success": last_ok,
            "last_speak_error": last_err,
        }
    except Exception as e:
        logger.error("[Health] TTS check failed: %s", e)
        return {"healthy": False, "error": str(e)}


def check_wake_word_health() -> dict:
    """Check wake word detector health."""
    try:
        from voice.coordinator import _voice_coordinator

        if _voice_coordinator is None:
            return {"healthy": False, "error": "No coordinator"}

        return {
            "healthy": True,
            "active": _voice_coordinator.active,
            "running": _voice_coordinator.running,
        }
    except Exception as e:
        logger.error("[Health] Wake word check failed: %s", e)
        return {"healthy": False, "error": str(e)}


def get_pipeline_health() -> dict:
    """Get complete voice pipeline health report."""
    return {
        "stt": check_stt_health(),
        "tts": check_tts_health(),
        "wake_word": check_wake_word_health(),
        "timestamp": time.time(),
    }


def is_pipeline_healthy() -> bool:
    """Quick check: is the entire voice pipeline healthy?"""
    report = get_pipeline_health()
    return (
        report["stt"].get("healthy", False)
        and report["tts"].get("healthy", False)
        and report["wake_word"].get("healthy", False)
    )


def attempt_stt_recovery() -> bool:
    """Attempt to recover STT if it has stopped."""
    try:
        from voice.stt import is_listening, restart_listening

        if is_listening():
            logger.info("[Health] STT is already listening — no recovery needed")
            return True

        logger.warning("[Health] STT is not listening — attempting recovery")

        result = restart_listening()
        if result:
            logger.info("[Health] STT recovery successful")
        else:
            logger.error("[Health] STT recovery failed")

        return result

    except Exception as e:
        logger.error("[Health] STT recovery exception: %s", e)
        return False


def print_health_report() -> None:
    """Print a human-readable health report to logs."""
    report = get_pipeline_health()

    logger.info("=" * 50)
    logger.info("VOICE PIPELINE HEALTH REPORT")
    logger.info("=" * 50)

    for component, details in report.items():
        if component == "timestamp":
            continue

        if isinstance(details, dict):
            status = "OK" if details.get("healthy", False) else "DEGRADED"
            logger.info("  [%s] %s", component.upper(), status)
            for key, val in details.items():
                if key != "healthy":
                    logger.info("    %s: %s", key, val)

    overall = "HEALTHY" if is_pipeline_healthy() else "DEGRADED"
    logger.info("=" * 50)
    logger.info("  OVERALL: %s", overall)
    logger.info("=" * 50)


# ---------------------------------------------------------------------------
# Watchdog — background thread that periodically checks and recovers
# ---------------------------------------------------------------------------

class VoiceWatchdog:
    """Background watchdog that monitors and auto-recovers the voice pipeline."""

    def __init__(self, interval_sec: int = WATCHDOG_INTERVAL_SEC):
        self._interval = interval_sec
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = False

    def start(self) -> bool:
        if self._running:
            return False
        self._stop.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[Health] Watchdog started (interval=%ds)", self._interval)
        return True

    def stop(self) -> None:
        self._stop.set()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("[Health] Watchdog stopped")

    @property
    def running(self) -> bool:
        return self._running

    def _run(self) -> None:
        while not self._stop.is_set():
            self._check_and_recover()
            self._stop.wait(timeout=self._interval)

    def _check_and_recover(self) -> None:
        try:
            stt_health = check_stt_health()

            # Auto-recover STT if it died
            if stt_health.get("listening") and not stt_health.get("thread_healthy"):
                logger.warning("[Health] STT thread dead — auto-recovering")
                attempt_stt_recovery()

            # Check TTS last-known state
            tts_health = check_tts_health()
            if tts_health.get("last_speak_success") is False:
                logger.warning("[Health] Last TTS attempt failed (engine=%s, error=%s)",
                              tts_health.get("engine", "?"),
                              tts_health.get("last_speak_error", "unknown"))

            # Detect and recover dead TTS thread
            self._check_tts_thread(tts_health)

        except Exception as e:
            logger.error("[Health] Watchdog check failed: %s", e)

    def _check_tts_thread(self, tts_health: dict) -> None:
        """Check if the TTS speaking thread is alive; attempt recovery if dead."""
        try:
            from tts.speaker import (
                is_speaking, get_engine_name, _speaking_thread,
                _last_speak_success, _last_speak_error,
            )

            speaking = is_speaking()
            if not speaking and tts_health.get("speaking"):
                # Was speaking but thread died — attempt recovery by clearing state
                logger.warning("[Health] TTS speaking thread died — clearing dead state")
                _last_speak_success = False
                _last_speak_error = "Watchdog detected dead TTS thread"
        except Exception as e:
            logger.debug("[Health] TTS thread check skipped: %s", e)


# Global watchdog instance
_watchdog: Optional[VoiceWatchdog] = None


def get_watchdog() -> VoiceWatchdog:
    """Get the global watchdog instance, creating it if needed."""
    global _watchdog
    if _watchdog is None:
        _watchdog = VoiceWatchdog()
    return _watchdog
