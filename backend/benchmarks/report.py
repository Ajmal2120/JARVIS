"""
JARVIS 3.0 — Benchmark report generation.

Provides standalone functions to export benchmark results to CSV
and Markdown, plus regression detection against a previous run.

This module is completely isolated from production code.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.benchmark_models import BenchmarkResult

logger = logging.getLogger("jarvis.benchmarks.report")


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegressionConfig:
    """Thresholds that trigger a regression flag.

    All thresholds are percentages (positive numbers).  A metric
    "regresses" when the relative change exceeds the threshold in the
    negative direction.

    Attributes:
        latency_pct:     End-to-end latency regression threshold (%).
        wer_pct:         WER regression threshold (percentage points).
        cer_pct:         CER regression threshold (percentage points).
        rejection_pct:   Rejection rate increase threshold (percentage
                         points, e.g. 5 = +5% absolute).
        confidence_pct:  Confidence regression threshold (% relative).
    """

    latency_pct: float = 10.0
    wer_pct: float = 2.0
    cer_pct: float = 2.0
    rejection_pct: float = 5.0
    confidence_pct: float = 5.0


@dataclass(frozen=True)
class MetricChange:
    """Diff for a single metric between two runs."""
    current: float = 0.0
    previous: float = 0.0
    change_pct: float = 0.0        # relative change (current - previous) / previous
    change_abs: float = 0.0        # absolute change
    threshold: float = 0.0         # the configured threshold
    is_regression: bool = False    # True when exceeded in the bad direction
    is_improvement: bool = False   # True when exceeded in the good direction
    status: str = "no_change"      # regression | improvement | no_change


@dataclass(frozen=True)
class RegressionReport:
    """Full regression analysis between two benchmark runs."""

    timestamp: str = ""
    baseline_timestamp: str = ""
    changes: dict[str, MetricChange] = field(default_factory=dict)
    verdict: str = "UNKNOWN"   # IMPROVED | NO_CHANGE | REGRESSION | MIXED
    regression_count: int = 0
    improvement_count: int = 0
    no_change_count: int = 0


# Threshold configurations for each metric.
# Each tuple is (config_field, metric_key, lower_is_better).
_METRIC_THRESHOLDS: list[tuple[str, str, bool]] = [
    ("latency_pct", "avg_latency_ms", True),          # lower is better
    ("wer_pct", "avg_wer", True),                       # lower is better
    ("cer_pct", "avg_cer", True),                       # lower is better
    ("rejection_pct", "rejection_rate", True),          # lower is better (absolute)
    ("confidence_pct", "avg_confidence", False),        # higher is better
]


def _relative_change(current: float, previous: float) -> float:
    """Return relative change as a percentage.  Handles zero baseline."""
    if previous == 0:
        return 0.0 if current == 0 else 100.0
    return ((current - previous) / abs(previous)) * 100.0


def compare_summaries(
    current: dict[str, Any],
    previous: dict[str, Any],
    config: RegressionConfig | None = None,
) -> RegressionReport:
    """Compare two benchmark summaries and return a regression report.

    Args:
        current:   Summary dict from the current run (see ``_summarize``).
        previous:  Summary dict loaded from ``benchmark_summary.json``.
        config:    Optional ``RegressionConfig`` with custom thresholds.

    Returns:
        ``RegressionReport`` with per-metric changes and overall verdict.
    """
    if config is None:
        config = RegressionConfig()

    changes: dict[str, MetricChange] = {}
    regressions = 0
    improvements = 0
    no_changes = 0

    for field_name, metric_key, lower_is_better in _METRIC_THRESHOLDS:
        threshold = getattr(config, field_name)
        cur_val = float(current.get(metric_key, 0.0) or 0.0)
        prev_val = float(previous.get(metric_key, 0.0) or 0.0)
        rel = _relative_change(cur_val, prev_val)

        # For absolute-difference metrics (rejection_rate), use abs change
        if field_name == "rejection_pct":
            change_pct = cur_val - prev_val   # absolute pp change
            is_reg = (cur_val - prev_val) > (threshold / 100.0)
            is_imp = (prev_val - cur_val) > (threshold / 100.0)
        else:
            change_pct = rel
            if lower_is_better:
                is_reg = rel > threshold
                is_imp = rel < -threshold
            else:
                is_reg = rel < -threshold
                is_imp = rel > threshold

        if is_reg:
            status = "regression"
            regressions += 1
        elif is_imp:
            status = "improvement"
            improvements += 1
        else:
            status = "no_change"
            no_changes += 1

        changes[metric_key] = MetricChange(
            current=round(cur_val, 4),
            previous=round(prev_val, 4),
            change_pct=round(change_pct, 2),
            change_abs=round(cur_val - prev_val, 4),
            threshold=threshold,
            is_regression=is_reg,
            is_improvement=is_imp,
            status=status,
        )

    if regressions > 0 and improvements > 0:
        verdict = "MIXED"
    elif regressions > 0:
        verdict = "REGRESSION"
    elif improvements > 0:
        verdict = "IMPROVED"
    else:
        verdict = "NO_CHANGE"

    return RegressionReport(
        timestamp=current.get("timestamp", ""),
        baseline_timestamp=previous.get("timestamp", ""),
        changes=changes,
        verdict=verdict,
        regression_count=regressions,
        improvement_count=improvements,
        no_change_count=no_changes,
    )


def print_regression_report(report: RegressionReport) -> None:
    """Print a human-readable regression report to stdout."""
    print()
    print("=" * 60)
    print("REGRESSION ANALYSIS")
    print("=" * 60)

    if not report.baseline_timestamp:
        print("No previous baseline found — this is the first run.")
        return

    print(f"Current:   {report.timestamp}")
    print(f"Baseline:  {report.baseline_timestamp}")
    print()

    for metric, change in report.changes.items():
        icon = {
            "regression": "[FAIL]",
            "improvement": "[OK]  ",
            "no_change": "[INFO]",
        }.get(change.status, "?")

        direction = ""
        if change.status == "regression":
            direction = " (REGRESSION)"
        elif change.status == "improvement":
            direction = " (improved)"

        print(f"{icon} {metric:25s}  {change.previous:8.4f} -> {change.current:8.4f}  "
              f"({change.change_pct:+.1f}%){direction}")

    print()
    print("-" * 60)

    verdict_icons = {
        "IMPROVED": "[PASS] Improved",
        "NO_CHANGE": "[WARN] No significant change",
        "REGRESSION": "[FAIL] REGRESSION",
        "MIXED": "[WARN] Mixed (some improved, some regressed)",
    }
    print(f"Verdict: {verdict_icons.get(report.verdict, report.verdict)}")
    print(f"  Regressions: {report.regression_count}")
    print(f"  Improvements: {report.improvement_count}")
    print(f"  No change: {report.no_change_count}")


# ---------------------------------------------------------------------------
# Summary persistence
# ---------------------------------------------------------------------------

def save_summary(summary: dict[str, Any], path: str | Path) -> Path:
    """Persist a summary dict to JSON for future regression comparison."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure timestamp is present
    if "timestamp" not in summary:
        summary = dict(summary)
        summary["timestamp"] = datetime.now().isoformat()
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("[Report] Summary saved to %s", path)
    return path


def load_summary(path: str | Path) -> dict[str, Any] | None:
    """Load a previously saved summary, or None if it doesn't exist."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[Report] Could not load previous summary: %s", exc)
        return None




def export_csv(results: list[BenchmarkResult], path: str | Path) -> Path:
    """Write per-utterance results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "utterance_id", "category", "expected", "recognized",
        "audio_duration_s", "stt_inference_ms", "end_to_end_ms",
        "wer", "cer", "confidence", "vad_decision", "accepted", "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "utterance_id": r.utterance_id,
                "category": r.category,
                "expected": r.expected,
                "recognized": r.recognized,
                "audio_duration_s": f"{r.audio_duration_s:.2f}",
                "stt_inference_ms": f"{r.stt_inference_ms:.0f}",
                "end_to_end_ms": f"{r.end_to_end_ms:.0f}",
                "wer": f"{r.wer:.4f}",
                "cer": f"{r.cer:.4f}",
                "confidence": f"{r.confidence:.2f}",
                "vad_decision": r.vad_decision,
                "accepted": "YES" if r.accepted else "NO",
                "error": r.error,
            })
    logger.info("[Report] CSV written to %s", path)
    return path


def export_markdown(
    results: list[BenchmarkResult],
    path: str | Path,
    model_size: str = "",
    language: str = "en",
) -> Path:
    """Write a human-readable Markdown benchmark report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = _summarize(results)

    lines = [
        f"# JARVIS 3.0 — STT Benchmark Report",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Configuration",
        f"",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Model | {model_size} |",
        f"| Language | {language} |",
        f"| Dataset size | {len(results)} |",
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
        cat_sum = _summarize(cat_results)
        lines.append(f"### {cat}")
        lines.append(f"")
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| Count | {cat_sum['total']} |")
        lines.append(f"| Accepted | {cat_sum['accepted']} |")
        lines.append(f"| Avg WER | {cat_sum['avg_wer']:.1%} |")
        lines.append(f"| Avg latency | {cat_sum['avg_latency_ms']:.0f} ms |")
        lines.append(f"")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("[Report] Markdown written to %s", path)
    return path


# ---------------------------------------------------------------------------
# Internal helpers (also used by stt_benchmark.py)
# ---------------------------------------------------------------------------

def _summarize(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Compute aggregate stats from results."""
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
        "avg_confidence": avg([r for r in accepted if r.confidence > 0], "confidence"),
        "by_category": by_cat,
    }


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a quick aggregate summary to stdout."""
    s = _summarize(results)
    print()
    print("=" * 55)
    print("AGGREGATE STATISTICS")
    print("=" * 55)
    print(f"Total utterances:    {s['total']}")
    print(f"Accepted:            {s['accepted']}")
    print(f"Rejected:            {s['rejected']}")
    print(f"Rejection rate:      {s['rejection_rate']:.1%}")
    print(f"Avg latency:         {s['avg_latency_ms']:.0f} ms")
    print(f"Avg STT inference:   {s['avg_stt_inference_ms']:.0f} ms")
    print(f"Avg WER:             {s['avg_wer']:.1%}")
    print(f"Avg CER:             {s['avg_cer']:.1%}")
    print(f"Avg confidence:      {s['avg_confidence']:.2f}")
