"""
JARVIS 3.0 — STT Benchmark Runner.

Benchmarks the faster-whisper STT pipeline against a dataset of
audio files with known transcripts.  Produces:

  - benchmark_results.csv  (per-utterance metrics)
  - benchmark_report.md    (human-readable summary)

This module is completely isolated from production code.  It imports
voice.stt only for the transcribe function, never mutates any
production state.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the backend package is importable when run directly
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from benchmarks.benchmark_models import BenchmarkResult, BenchmarkSummary

logger = logging.getLogger("jarvis.benchmarks.stt")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate using jiwer (or a simple fallback)."""
    try:
        import jiwer
        return jiwer.wer(reference.strip(), hypothesis.strip())
    except ImportError:
        ref = reference.strip().split()
        hyp = hypothesis.strip().split()
        if not ref:
            return 0.0 if not hyp else 1.0
        # Levenshtein at word level
        m, n = len(ref), len(hyp)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if ref[i-1] == hyp[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost,
                )
        return dp[m][n] / m


def _cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate using jiwer (or a simple fallback)."""
    try:
        import jiwer
        return jiwer.cer(reference.strip(), hypothesis.strip())
    except ImportError:
        ref = reference.strip()
        hyp = hypothesis.strip()
        if not ref:
            return 0.0 if not hyp else 1.0
        m, n = len(ref), len(hyp)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if ref[i-1] == hyp[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost,
                )
        return dp[m][n] / m


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkUtterance:
    """One entry in the benchmark dataset."""
    utterance_id: str
    category: str
    expected: str
    audio_file: str = ""  # path to WAV file, or empty for synthetic
    metadata: dict = None  # type: ignore


DEFAULT_DATASET: list[BenchmarkUtterance] = [
    # ── Short commands ──────────────────────────────────────────────────
    BenchmarkUtterance("s01", "short_command", "Open Chrome"),
    BenchmarkUtterance("s02", "short_command", "Close Chrome"),
    BenchmarkUtterance("s03", "short_command", "Mute"),
    BenchmarkUtterance("s04", "short_command", "Stop"),
    BenchmarkUtterance("s05", "short_command", "Volume up"),
    BenchmarkUtterance("s06", "short_command", "Volume down"),
    BenchmarkUtterance("s07", "short_command", "Analyze screen"),
    BenchmarkUtterance("s08", "short_command", "What time is it"),
    BenchmarkUtterance("s09", "short_command", "Take a screenshot"),
    BenchmarkUtterance("s10", "short_command", "Lock the screen"),
    # ── Long commands ───────────────────────────────────────────────────
    BenchmarkUtterance("l01", "long_command",
                       "Analyze my current screen and summarize what I'm working on"),
    BenchmarkUtterance("l02", "long_command",
                       "Can you explain the code currently open in Visual Studio Code"),
    BenchmarkUtterance("l03", "long_command",
                       "Search the web for the latest AI news"),
    BenchmarkUtterance("l04", "long_command",
                       "Open the file manager and navigate to the documents folder"),
    BenchmarkUtterance("l05", "long_command",
                       "Create a new folder on the desktop called project files"),
    # ── Variations ──────────────────────────────────────────────────────
    BenchmarkUtterance("v01", "quiet_speech", "Open Chrome"),
    BenchmarkUtterance("v02", "fast_speech", "Close Chrome"),
    BenchmarkUtterance("v03", "slow_speech", "Mute"),
    BenchmarkUtterance("v04", "accent", "Volume up"),
    BenchmarkUtterance("v05", "long_pause", "Analyze screen"),
]


# ---------------------------------------------------------------------------
# Audio synthesis (for when real recordings are unavailable)
# ---------------------------------------------------------------------------

def _synthesize_speech(text: str, output_path: str) -> bool:
    """Generate a WAV file from text using the system TTS engine.

    Returns True on success, False otherwise.  This is a best-effort
    helper so the benchmark can run without pre-recorded audio.
    """
    try:
        import wave
        import numpy as np
        import struct

        # Try using pyttsx3 / SAPI to generate speech
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.setProperty("volume", 1.0)

            # Save to file — pytts3 can write WAV on Windows
            temp_path = output_path + ".tmp"
            engine.save_to_file(text, temp_path)
            engine.runAndWait()

            # pyttsx3 saves as .wav on some platforms, .txt on others
            # If it succeeded, move it
            if os.path.exists(temp_path):
                size = os.path.getsize(temp_path)
                if size > 1000:  # real audio file
                    os.replace(temp_path, output_path)
                    return True
                os.remove(temp_path)
        except Exception:
            pass

        # Fallback: generate a silent/sine WAV (for latency testing only)
        sample_rate = 16000
        duration = max(0.5, len(text) * 0.08)  # ~80ms per word
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Mix of tones so the STT gets *something* (even if it produces
        # garbage — we can measure the pipeline latency regardless)
        tone = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 880 * t)
        audio = (tone * 32767).astype(np.int16)

        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())

        return True

    except Exception as exc:
        logger.debug("Speech synthesis failed: %s", exc)
        return False


def _synthesize_dataset(dataset_dir: Path) -> None:
    """Generate WAV files for any dataset entries that lack them."""
    for entry in DEFAULT_DATASET:
        wav_path = dataset_dir / f"{entry.utterance_id}.wav"
        if not wav_path.exists():
            logger.info("[Benchmark] Synthesizing %s (%s) -> %s",
                        entry.utterance_id, entry.category, wav_path)
            _synthesize_speech(entry.expected, str(wav_path))


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class STTBenchmark:
    """Runs the STT benchmark against the faster-whisper pipeline.

    Usage::

        bench = STTBenchmark()
        results = bench.run()
        bench.print_report(results)
        bench.export_csv(results, "results.csv")
        bench.export_markdown(results, "report.md")
    """

    def __init__(
        self,
        dataset: Optional[list[BenchmarkUtterance]] = None,
        model_size: str = "small",
        language: str = "en",
        output_dir: str = "benchmarks/output",
    ) -> None:
        self.dataset = dataset or DEFAULT_DATASET
        self.model_size = model_size
        self.language = language
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Derived directories
        self.dataset_dir = self.output_dir / "dataset"
        self.dataset_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(self) -> list[BenchmarkResult]:
        """Run the full benchmark and return per-utterance results."""
        logger.info("[Benchmark] Starting STT benchmark (%d utterances, model=%s)",
                    len(self.dataset), self.model_size)

        # Ensure audio files exist
        _synthesize_dataset(self.dataset_dir)

        # Lazy-load the model (mirrors voice.stt._load_model)
        model = self._load_model()

        results: list[BenchmarkResult] = []

        for entry in self.dataset:
            result = self._benchmark_utterance(model, entry)
            results.append(result)

        summary = self._summarize(results)

        # Print summary
        self._print_summary(summary)
        return results

    def print_report(self, results: list[BenchmarkResult]) -> None:
        """Print a human-readable report to stdout."""
        for r in results:
            print("-" * 55)
            print(f"Command:      {r.expected}")
            print(f"Recognized:   {r.recognized or '(nothing)'}")
            print(f"WER:          {r.wer:.1%}")
            print(f"CER:          {r.cer:.1%}")
            print(f"Latency:      {r.end_to_end_ms:.0f} ms")
            print(f"  Inference:  {r.stt_inference_ms:.0f} ms")
            print(f"Accepted:     {'YES' if r.accepted else 'NO'}")
            if r.error:
                print(f"Error:        {r.error}")

        summary = self._summarize(results)
        print()
        print("=" * 55)
        print("AGGREGATE STATISTICS")
        print("=" * 55)
        print(f"Total utterances:    {summary['total']}")
        print(f"Accepted:            {summary['accepted']}")
        print(f"Rejected:            {summary['rejected']}")
        print(f"Rejection rate:      {summary['rejection_rate']:.1%}")
        print(f"Avg latency:         {summary['avg_latency_ms']:.0f} ms")
        print(f"Avg STT inference:   {summary['avg_stt_inference_ms']:.0f} ms")
        print(f"Avg WER:             {summary['avg_wer']:.1%}")
        print(f"Avg CER:             {summary['avg_cer']:.1%}")
        print(f"Avg confidence:      {summary['avg_confidence']:.2f}")

    def export_csv(self, results: list[BenchmarkResult], filename: str = "benchmark_results.csv") -> Path:
        """Export per-utterance results to CSV."""
        csv_path = self.output_dir / filename
        fields = [
            "utterance_id", "category", "expected", "recognized",
            "audio_duration_s", "wake_to_listen_ms", "stt_inference_ms",
            "end_to_end_ms", "wer", "cer", "confidence",
            "vad_decision", "accepted", "error",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "utterance_id": r.utterance_id,
                    "category": r.category,
                    "expected": r.expected,
                    "recognized": r.recognized,
                    "audio_duration_s": f"{r.audio_duration_s:.2f}",
                    "wake_to_listen_ms": f"{r.wake_to_listen_ms:.0f}",
                    "stt_inference_ms": f"{r.stt_inference_ms:.0f}",
                    "end_to_end_ms": f"{r.end_to_end_ms:.0f}",
                    "wer": f"{r.wer:.4f}",
                    "cer": f"{r.cer:.4f}",
                    "confidence": f"{r.confidence:.2f}",
                    "vad_decision": r.vad_decision,
                    "accepted": "YES" if r.accepted else "NO",
                    "error": r.error,
                })
        logger.info("[Benchmark] CSV written to %s", csv_path)
        return csv_path

    def export_markdown(self, results: list[BenchmarkResult], filename: str = "benchmark_report.md") -> Path:
        """Export a human-readable Markdown report."""
        md_path = self.output_dir / filename
        summary = self._summarize(results)

        lines = [
            f"# JARVIS 3.0 — STT Benchmark Report",
            f"",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## Configuration",
            f"",
            f"| Parameter | Value |",
            f"|---|---|",
            f"| Model | {self.model_size} |",
            f"| Language | {self.language} |",
            f"| Dataset size | {len(self.dataset)} |",
            f"",
            f"## Per-Utterance Results",
            f"",
        ]

        for r in results:
            lines.append(f"### `{r.utterance_id}` — {r.category}")
            lines.append(f"")
            lines.append(f"| Metric | Value |")
            lines.append(f"|---|---|")
            lines.append(f"| Expected | {r.expected} |")
            lines.append(f"| Recognized | {r.recognized or '(nothing)'} |")
            lines.append(f"| WER | {r.wer:.1%} |")
            lines.append(f"| CER | {r.cer:.1%} |")
            lines.append(f"| End-to-end latency | {r.end_to_end_ms:.0f} ms |")
            lines.append(f"| STT inference | {r.stt_inference_ms:.0f} ms |")
            lines.append(f"| Accepted | {'YES' if r.accepted else 'NO'} |")
            if r.error:
                lines.append(f"| Error | {r.error} |")
            lines.append(f"")

        lines += [
            f"## Aggregate Statistics",
            f"",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Total utterances | {summary['total']} |",
            f"| Accepted | {summary['accepted']} |",
            f"| Rejected | {summary['rejected']} |",
            f"| Rejection rate | {summary['rejection_rate']:.1%} |",
            f"| Avg latency | {summary['avg_latency_ms']:.0f} ms |",
            f"| Avg STT inference | {summary['avg_stt_inference_ms']:.0f} ms |",
            f"| Avg WER | {summary['avg_wer']:.1%} |",
            f"| Avg CER | {summary['avg_cer']:.1%} |",
            f"| Avg confidence | {summary['avg_confidence']:.2f} |",
            f"",
            f"## By Category",
            f"",
        ]

        for cat, cat_results in summary["by_category"].items():
            cat_summary = self._summarize(cat_results)
            lines.append(f"### {cat}")
            lines.append(f"")
            lines.append(f"| Metric | Value |")
            lines.append(f"|---|---|")
            lines.append(f"| Count | {cat_summary['total']} |")
            lines.append(f"| Accepted | {cat_summary['accepted']} |")
            lines.append(f"| Avg WER | {cat_summary['avg_wer']:.1%} |")
            lines.append(f"| Avg latency | {cat_summary['avg_latency_ms']:.0f} ms |")
            lines.append(f"")

        md_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("[Benchmark] Markdown report written to %s", md_path)
        return md_path

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _load_model(self):
        """Lazy-load faster-whisper (mirrors voice.stt._load_model)."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError("faster-whisper is not installed")

        device = "cuda"
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        except ImportError:
            device = "cpu"

        logger.info("[Benchmark] Loading Whisper model (%s) on %s...", self.model_size, device)
        model = WhisperModel(
            self.model_size,
            device=device,
            compute_type="int8" if device == "cpu" else "float16",
            num_workers=1,
        )
        logger.info("[Benchmark] Model loaded")
        return model

    def _benchmark_utterance(
        self, model: Any, entry: BenchmarkUtterance
    ) -> BenchmarkResult:
        """Run STT on one audio file and measure everything."""
        wav_path = self.dataset_dir / f"{entry.utterance_id}.wav"
        audio_duration_s = 0.0
        recognized = ""
        stt_inference_ms = 0.0
        confidence = 0.0
        vad_decision = "rejected"
        accepted = False
        error = ""

        try:
            if wav_path.exists():
                audio_duration_s = self._audio_duration(wav_path)
                t0 = time.monotonic()
                recognized, confidence = self._transcribe(model, wav_path)
                stt_inference_ms = (time.monotonic() - t0) * 1000.0
                vad_decision = "accepted"
                accepted = bool(recognized.strip())
            else:
                error = f"audio file missing: {wav_path}"

        except Exception as exc:
            error = str(exc)
            logger.warning("[Benchmark] %s failed: %s", entry.utterance_id, exc)

        wer = _wer(entry.expected, recognized) if recognized else 1.0
        cer = _cer(entry.expected, recognized) if recognized else 1.0

        return BenchmarkResult(
            utterance_id=entry.utterance_id,
            category=entry.category,
            expected=entry.expected,
            recognized=recognized,
            audio_duration_s=audio_duration_s,
            audio_file=str(wav_path),
            stt_inference_ms=stt_inference_ms,
            end_to_end_ms=stt_inference_ms,  # no wake word in bench
            wer=wer,
            cer=cer,
            confidence=confidence,
            vad_decision=vad_decision,
            accepted=accepted,
            error=error,
        )

    def _transcribe(self, model: Any, wav_path: Path) -> tuple[str, float]:
        """Run whisper transcription and return (text, avg_confidence)."""
        try:
            import numpy as np
            import soundfile as sf

            audio, sr = sf.read(str(wav_path), dtype="float32")
            if sr != 16000:
                import scipy.signal
                ratio = 16000.0 / sr
                new_len = int(len(audio) * ratio)
                audio = scipy.signal.resample(audio, new_len).astype(np.float32)

            segments, info = model.transcribe(
                audio,
                language=self.language,
                beam_size=3,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
                without_timestamps=True,
            )

            text_parts = []
            confidences = []
            for seg in segments:
                if seg.no_speech_prob > 0.9:
                    continue
                text = seg.text.strip()
                if text:
                    text_parts.append(text)
                    confidences.append(1.0 - seg.no_speech_prob)

            text = " ".join(text_parts).strip()
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            return text, avg_conf

        except Exception as exc:
            logger.warning("[Benchmark] transcribe failed: %s", exc)
            return "", 0.0

    @staticmethod
    def _audio_duration(wav_path: Path) -> float:
        """Return audio duration in seconds."""
        try:
            import soundfile as sf
            info = sf.info(str(wav_path))
            return float(info.duration)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ #
    # Reporting helpers                                                   #
    # ------------------------------------------------------------------ #

    def _summarize(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """Return a plain-dict summary for reporting."""
        accepted = [r for r in results if r.accepted]
        rejected = [r for r in results if not r.accepted]
        total = len(results)

        def avg(lst, attr):
            vals = [getattr(r, attr) for r in lst]
            return sum(vals) / len(vals) if vals else 0.0

        by_cat: dict[str, list[BenchmarkResult]] = {}
        for r in results:
            by_cat.setdefault(r.category, []).append(r)

        return {
            "total": total,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "rejection_rate": len(rejected) / total if total else 0.0,
            "avg_latency_ms": avg(accepted, "end_to_end_ms"),
            "avg_stt_inference_ms": avg(accepted, "stt_inference_ms"),
            "avg_wer": avg(accepted, "wer"),
            "avg_cer": avg(accepted, "cer"),
            "confs": [r.confidence for r in accepted if r.confidence > 0],
            "avg_confidence": avg([r for r in accepted if r.confidence > 0], "confidence"),
            "by_category": by_cat,
        }

    def _print_summary(self, summary: dict) -> None:
        """Print aggregate stats to stdout."""
        print()
        print("=" * 55)
        print("AGGREGATE STATISTICS")
        print("=" * 55)
        print(f"Total utterances:    {summary['total']}")
        print(f"Accepted:            {summary['accepted']}")
        print(f"Rejected:            {summary['rejected']}")
        print(f"Rejection rate:      {summary['rejection_rate']:.1%}")
        print(f"Avg latency:         {summary['avg_latency_ms']:.0f} ms")
        print(f"Avg STT inference:   {summary['avg_stt_inference_ms']:.0f} ms")
        print(f"Avg WER:             {summary['avg_wer']:.1%}")
        print(f"Avg CER:             {summary['avg_cer']:.1%}")
        print(f"Avg confidence:      {summary['avg_confidence']:.2f}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="JARVIS 3.0 STT Benchmark")
    parser.add_argument("--model", default="small", help="Whisper model size")
    parser.add_argument("--language", default="en", help="Transcription language")
    parser.add_argument("--output-dir", default="benchmarks/output", help="Output directory")
    parser.add_argument("--csv", default="benchmark_results.csv", help="CSV filename")
    parser.add_argument("--md", default="benchmark_report.md", help="Markdown report filename")
    parser.add_argument("--no-regression", action="store_true",
                        help="Skip regression detection (just run the benchmark)")
    parser.add_argument("--regression-summary", default="benchmark_summary.json",
                        help="Filename for the persisted summary (enables comparison)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    bench = STTBenchmark(
        model_size=args.model,
        language=args.language,
        output_dir=args.output_dir,
    )
    results = bench.run()
    bench.print_report(results)
    csv_path = bench.export_csv(results, args.csv)
    md_path = bench.export_markdown(results, args.md)
    print(f"\nCSV:  {csv_path}")
    print(f"Report: {md_path}")

    # ------------------------------------------------------------------ #
    # Regression detection                                                #
    # ------------------------------------------------------------------ #
    from benchmarks.report import (
        save_summary, load_summary, compare_summaries, print_regression_report,
    )

    summary_path = bench.output_dir / "benchmark_summary.json"
    baseline_path = bench.output_dir / "baseline_summary.json"
    current_summary = bench._summarize(results)
    current_summary["timestamp"] = datetime.now().isoformat()

    # Always save the current run
    save_summary(current_summary, summary_path)

    # If no baseline exists, this is the first run
    baseline = load_summary(baseline_path)
    if baseline is None:
        # Promote this run to the baseline
        save_summary(current_summary, baseline_path)
        print(f"\nFirst run — baseline saved to {baseline_path}")
        print("Future runs will compare against this baseline.")
    else:
        report = compare_summaries(current_summary, baseline)
        print_regression_report(report)

        if report.verdict == "REGRESSION":
            print("\nExiting with code 2 due to regression(s).")
            raise SystemExit(2)


if __name__ == "__main__":
    main()
