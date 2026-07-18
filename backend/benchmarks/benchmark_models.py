"""
JARVIS 3.0 — Benchmark data models.

Defines the dataclasses used to record and report STT benchmark results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class BenchmarkResult:
    """Result of a single STT benchmark utterance."""

    # Identity
    utterance_id: str = ""
    category: str = ""  # short_command | long_command | noisy | quiet | fast | slow | accent | long_pause

    # Ground truth
    expected: str = ""
    recognized: str = ""

    # Audio metrics
    audio_duration_s: float = 0.0
    audio_file: str = ""

    # Latency breakdown (milliseconds)
    wake_to_listen_ms: float = 0.0
    stt_inference_ms: float = 0.0
    end_to_end_ms: float = 0.0

    # Quality metrics
    wer: float = 0.0
    cer: float = 0.0
    confidence: float = 0.0

    # VAD / acceptance
    vad_decision: str = ""  # accepted | rejected
    silence_detected_ms: float = 0.0

    # Outcome
    accepted: bool = False
    error: str = ""

    @property
    def correct(self) -> bool:
        """True when the recognized text matches the expected text."""
        return self.wer == 0.0 and self.accepted


@dataclass
class BenchmarkSummary:
    """Aggregated statistics across a benchmark run."""

    total_utterances: int = 0
    accepted: int = 0
    rejected: int = 0
    average_latency_ms: float = 0.0
    average_wer: float = 0.0
    average_cer: float = 0.0
    average_confidence: float = 0.0
    average_stt_inference_ms: float = 0.0
    rejection_rate: float = 0.0
    by_category: dict[str, list[BenchmarkResult]] = field(default_factory=dict)

    def compute(self, results: list[BenchmarkResult]) -> None:
        """Compute aggregates from a list of results."""
        self.total_utterances = len(results)
        if not results:
            return

        accepted_results = [r for r in results if r.accepted]
        self.accepted = len(accepted_results)
        self.rejected = self.total_utterances - self.accepted
        self.rejection_rate = self.rejected / self.total_utterances if self.total_utterances else 0.0

        if accepted_results:
            self.average_latency_ms = sum(r.end_to_end_ms for r in accepted_results) / len(accepted_results)
            self.average_stt_inference_ms = sum(r.stt_inference_ms for r in accepted_results) / len(accepted_results)
            self.average_wer = sum(r.wer for r in accepted_results) / len(accepted_results)
            self.average_cer = sum(r.cer for r in accepted_results) / len(accepted_results)
            confs = [r.confidence for r in accepted_results if r.confidence > 0]
            self.average_confidence = sum(confs) / len(confs) if confs else 0.0

        # Group by category
        self.by_category = {}
        for r in results:
            self.by_category.setdefault(r.category, []).append(r)
