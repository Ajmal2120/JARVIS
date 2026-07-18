"""
JARVIS 3.0 — Benchmark package.

This package is completely isolated from production code.  It contains
tools to measure and report on the STT pipeline's accuracy and latency.

Files:
    benchmark_models.py  — Result and Summary dataclasses
    stt_benchmark.py     — Runner that exercises faster-whisper
    report.py            — Report generation helpers (CSV, Markdown)
"""