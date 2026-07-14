"""
Future architecture preparation hooks for JARVIS voice pipeline.

This module provides extension points for future enhancements without breaking
the existing architecture:

1. VAD provider abstraction (Silero, Whisper VAD, custom VAD)
2. Streaming response handling (progressive TTS)
3. Conversation state management
4. Confidence scoring framework
5. Barge-in (interrupt) detection
6. Echo suppression
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VAD Provider Interface (for future Silero integration)
# ---------------------------------------------------------------------------

class VADProvider(ABC):
    """Abstract base class for Voice Activity Detection providers."""

    @abstractmethod
    async def detect_speech(
        self,
        audio_chunk: bytes,
        sample_rate: int
    ) -> Dict[str, Union[bool, float, List[float]]]:
        """Detect speech in audio chunk.
        Returns: {"has_speech": bool, "confidence": float, "speech_timestamps": List[float]}
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if VAD is available/installed."""
        pass


class WhisperVADProvider(VADProvider):
    """Built-in Whisper VAD provider."""

    def __init__(self):
        self._model = None

    def is_available(self) -> bool:
        try:
            import faster_whisper
            return True
        except ImportError:
            return False

    async def detect_speech(self, audio_chunk: bytes, sample_rate: int) -> Dict:
        # Would use faster-whisper's VAD
        # Placeholder for future implementation
        return {"has_speech": True, "confidence": 0.5, "speech_timestamps": []}


class SileroVADProvider(VADProvider):
    """Future Silero VAD provider."""

    def __init__(self):
        self._model = None

    def is_available(self) -> bool:
        try:
            from silero_vad import get_speech_timestamps
            return True
        except ImportError:
            return False

    async def detect_speech(self, audio_chunk: bytes, sample_rate: int) -> Dict:
        # Would use Silero VAD
        # Placeholder for future implementation
        return {"has_speech": True, "confidence": 0.5, "speech_timestamps": []}


# Get default VAD provider (current implementation)
def get_vad_provider() -> VADProvider:
    """Get the default VAD provider."""
    # Future could prioritize Silero, then Whisper, then fallback to custom
    provider = WhisperVADProvider()
    if provider.is_available():
        return provider
    return provider


# ---------------------------------------------------------------------------
# Streaming Response Interface
# ---------------------------------------------------------------------------

class TTSStreamingCallback(Protocol):
    """Callback for progressive TTS playback."""

    def __call__(self, audio_chunk: bytes, chunk_index: int, total_chunks: int) -> None:
        pass


class StreamingTTSEngine(ABC):
    """Abstract base for streaming TTS engines."""

    @abstractmethod
    async def stream_speak(
        self,
        text: str,
        callback: TTSStreamingCallback,
        chunk_size: int = 512
    ) -> bool:
        """Speak text in chunks via callback."""
        pass


# ---------------------------------------------------------------------------
# Conversation State Manager
# ---------------------------------------------------------------------------

@dataclass
class ConversationContext:
    """Context for understanding conversational state."""

    last_command: str = ""
    last_response: str = ""
    topic: str = ""
    confidence: float = 0.0
    user_name: str = ""
    location: Optional[str] = None

    def clear(self):
        """Clear conversation context."""
        self.__dict__.clear()
        self.last_command = ""
        self.last_response = ""
        self.topic = ""
        self.confidence = 0.0


class ConversationManager:
    """Manages conversation context for better understanding."""

    def __init__(self):
        self.context = ConversationContext()
        self._listeners: List[Callable[[ConversationContext], None]] = []

    def update(self, command: str, response: str):
        """Update context with new command/response."""
        self.context.last_command = command
        self.context.last_response = response
        self._notify_listeners()

    def add_listener(self, callback: Callable[[ConversationContext], None]):
        """Register callback for context changes."""
        self._listeners.append(callback)

    def _notify_listeners(self):
        """Notify all listeners of context update."""
        for listener in self._listeners:
            try:
                listener(self.context)
            except Exception as e:
                logger.error("Context listener error: %s", e)

    def clear_context(self):
        """Clear all conversation context."""
        self.context.clear()
        self._notify_listeners()


# Global conversation manager instance
_conversation_manager = ConversationManager()


def get_conversation_manager() -> ConversationManager:
    """Get the global conversation manager."""
    return _conversation_manager


# ---------------------------------------------------------------------------
# Confidence Scoring Framework
# ---------------------------------------------------------------------------

class ConfidenceScorer:
    """Scores speech recognition quality for quality monitoring."""

    def __init__(self):
        self._scores = []

    def add_score(self, text: str, confidence: float, audio_level: float):
        """Record a confidence score."""
        self._scores.append({
            "text": text,
            "confidence": confidence,
            "audio_level": audio_level,
            "timestamp": asyncio.get_event_loop().time()
        })

    def get_average_confidence(self, window: int = 10) -> float:
        """Get average confidence over recent window."""
        recent = self._scores[-window:]
        if not recent:
            return 0.0
        return sum(s["confidence"] for s in recent) / len(recent)

    def clear_history(self):
        """Clear confidence history."""
        self._scores.clear()


# Global confidence scorer instance
_confidence_scorer = ConfidenceScorer()


def get_confidence_scorer() -> ConfidenceScorer:
    """Get the global confidence scorer."""
    return _confidence_scorer


# ---------------------------------------------------------------------------
# Barge-In Detection (Future)
# ---------------------------------------------------------------------------

def initialize_barge_in_detection(
    microphone_callback: Callable,
    barge_in_threshold: float = 0.3
) -> Callable:
    """Initialize barge-in detection for interrupting JARVIS.

    Returns a wrapped callback that can detect user interruption.
    """
    logger.info("[Future] Barge-in detection initialized (threshold: %.2f)", barge_in_threshold)

    async def barge_in_monitor(audio_chunk):
        # Check audio chunk for interruption signal
        # Placeholder implementation
        return None

    return barge_in_monitor


# ---------------------------------------------------------------------------
# Echo Suppression (Future)
# ---------------------------------------------------------------------------

def enable_echo_suppression(output_device: Optional[int] = None):
    """Enable echo suppression to prevent JARVIS from hearing its own voice.

    Future implementation:
    1. Monitor TTS output
    2. Buffer and subtract from microphone input
    3. Use DSP techniques to reduce echo
    """
    logger.info("[Future] Echo suppression enabled (device: %s)", output_device)

    async def echo_suppressor(audio_chunk) -> bytes:
        # Future: Apply echo cancellation
        return audio_chunk

    return echo_suppressor


# ---------------------------------------------------------------------------
# Utility functions for easy integration
# ---------------------------------------------------------------------------

def register_conversation_listener(callback: Callable[[ConversationContext], None]):
    """Register a callback for conversation events."""
    get_conversation_manager().add_listener(callback)


async def test_future_components():
    """Test all future components are importable and available."""
    # Test VAD provider
    vad = get_vad_provider()
    logger.info("[Future] VAD provider: %s (%s)",
               type(vad).__name__, "available" if vad.is_available() else "unavailable")

    # Test conversation manager
    conv = get_conversation_manager()
    conv.update("test command", "test response")
    logger.info("[Future] Conversation context updated")

    # Test confidence scorer
    scorer = get_confidence_scorer()
    scorer.add_score("test", 0.8, 0.5)
    logger.info("[Future] Average confidence: %.2f", scorer.get_average_confidence())

    logger.info("[Future] All future components tested successfully")


# ---------------------------------------------------------------------------
# Plugin registration for future extensions
# ---------------------------------------------------------------------------

class VoicePlugin(ABC):
    """Base class for voice pipeline plugins."""

    @abstractmethod
    def initialize(self, config: Dict) -> bool:
        """Initialize plugin with config."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up plugin resources."""
        pass


_plugins: Dict[str, VoicePlugin] = {}


def register_plugin(name: str, plugin: VoicePlugin):
    """Register a voice pipeline plugin."""
    _plugins[name] = plugin
    logger.info("[Future] Plugin registered: %s", name)


def get_plugin(name: str) -> Optional[VoicePlugin]:
    """Get a registered plugin by name."""
    return _plugins.get(name)


# Initialize all future components
def initialize_future_components():
    """Initialize all future architecture components."""
    logger.info("[Future] Initializing future architecture components")

    # Set up default conversation manager listeners
    def log_context_change(context: ConversationContext):
        logger.debug("[Future] Context updated: last_command=\"%s\"",
                    context.last_command[:50])

    register_conversation_listener(log_context_change)

    logger.info("[Future] Future architecture initialized")


# Auto-initialize when module loads
initialize_future_components()
