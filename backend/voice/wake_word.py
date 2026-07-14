"""
Wake word detection for JARVIS.

Monitors speech recognition output for wake words like "JARVIS".
Includes fuzzy matching for variations like "jarvus", "jarvas", "garvis".
"""

import logging
import threading
from typing import Callable, Optional, Set

logger = logging.getLogger(__name__)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class WakeWordDetector:
    """Detects wake words in speech recognition output."""

    # Default wake words
    DEFAULT_WAKE_WORDS = {
        'jarvis',
        'hey jarvis',
        'ok jarvis',
        'yo jarvis',
        'wake up',
        'jarvis are you there',
        'jarvis wake up',
    }

    def __init__(
        self,
        wake_words: Optional[Set[str]] = None,
        on_wake: Optional[Callable] = None,
        on_sleep: Optional[Callable] = None,
    ):
        """Initialize wake word detector."""
        self.wake_words = wake_words or self.DEFAULT_WAKE_WORDS
        self.on_wake = on_wake
        self.on_sleep = on_sleep
        self.is_active = False
        self._lock = threading.Lock()

    def _fuzzy_match(self, text: str) -> bool:
        """Check if text contains a fuzzy match to 'jarvis' (edit distance <= 1)."""
        text_lower = text.lower()
        words = text_lower.split()

        for word in words:
            # Clean punctuation
            word = ''.join(c for c in word if c.isalnum())
            if not word:
                continue

            # Direct match
            if word == 'jarvis':
                return True

            # Skip short words
            if len(word) < 4:
                continue

            # Check edit distance (fuzzy matching for STT errors)
            # "jarvus", "jarvas", "garvis", "jarviss", etc.
            dist = _levenshtein_distance(word, 'jarvis')
            if dist <= 1:
                logger.info(f"Fuzzy wake word match: '{word}' (distance={dist})")
                return True

        return False

    def contains_wake_word(self, text: str) -> bool:
        """Check if text contains any wake word."""
        text_lower = text.lower()

        # Exact match check
        if any(word in text_lower for word in self.wake_words):
            return True

        # Fuzzy match for jarvis variations
        if self._fuzzy_match(text_lower):
            return True

        return False

    def extract_command(self, text: str) -> str:
        """Extract command text after wake word."""
        text_lower = text.lower()
        for word in self.wake_words:
            if word in text_lower:
                # Remove wake word and clean up
                command = text_lower.replace(word, '', 1)
                command = command.strip(' ,.!?').strip()
                # Preserve original casing
                original_idx = text_lower.find(word)
                if original_idx >= 0:
                    before = text[:original_idx]
                    after = text[original_idx + len(word):].strip(' ,.!?').strip()
                    return after
                return command
        return text.strip()

    def process(self, text: str) -> Optional[str]:
        """Process speech text and return command if wake word detected, None otherwise."""
        if not self.contains_wake_word(text):
            return None

        # First wake - activate
        if not self.is_active:
            with self._lock:
                self.is_active = True
            if self.on_wake:
                self.on_wake()
            return self.extract_command(text)

        # Already active - check for sleep
        sleep_words = {'sleep', 'goodnight', 'stop listening', 'go to sleep', 'sleep mode'}
        text_lower = text.lower()
        if any(word in text_lower for word in sleep_words):
            with self._lock:
                self.is_active = False
            if self.on_sleep:
                self.on_sleep()
            return None

        # Active and no sleep word - return command
        return self.extract_command(text)

    def deactivate(self):
        """Manually deactivate wake word detection."""
        with self._lock:
            self.is_active = False
        if self.on_sleep:
            self.on_sleep()

    def activate(self):
        """Manually activate wake word detection."""
        with self._lock:
            self.is_active = True
        if self.on_wake:
            self.on_wake()

    @property
    def active(self) -> bool:
        """Check if wake word detector is active."""
        return self.is_active


def detect_wake_word(text: str, wake_words: Optional[Set[str]] = None) -> bool:
    """Quick check if text contains wake word."""
    detector = WakeWordDetector(wake_words=wake_words)
    return detector.contains_wake_word(text)


def extract_command(text: str, wake_words: Optional[Set[str]] = None) -> str:
    """Extract command after wake word."""
    detector = WakeWordDetector(wake_words=wake_words)
    return detector.extract_command(text)
