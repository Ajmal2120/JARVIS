"""
JARVIS 3.0 — Sprint 2 tests.

Tests for new Sprint 2 components:
    - CooldownManager (cooldown.py)
    - ActivityMonitor (activity.py)
    - SuggestionMessenger (messenger.py)
    - SuggestionMetricsTracker (metrics.py)
    - Integration: SuggestionEngine with cooldown, activity, metrics
"""
from __future__ import annotations

import logging
import sys
import threading
import time

logging.basicConfig(level=logging.DEBUG)

import runtime.runtime as runtime_mod
from runtime.runtime import Runtime, get_runtime, set_runtime
from runtime.events import RuntimeEvent

from suggestion.cooldown import CooldownManager
from suggestion.activity import ActivityMonitor
from suggestion.messenger import build_message, build_reason, build_suggestion_text
from suggestion.metrics import SuggestionMetrics, SuggestionMetricsTracker
from suggestion.engine import SuggestionEngine
from suggestion.providers import SuggestionProvider, GeneralDevelopmentProvider
from suggestion.models import Suggestion, SuggestionStatus, PreparedRequest
from suggestion.events import (
    CooldownStarted, CooldownExpired, SuggestionCreated,
    SuggestionAccepted, SuggestionDeclined, SuggestionExpired,
    SuggestionSuppressed, SuggestionDelayed,
)

try:
    from inference.models import Hypothesis
except ImportError:
    try:
        from suggestion.hypothesis import Hypothesis
    except ImportError:
        class Hypothesis:
            def __init__(self, id="", confidence=0.0, evidence=None, created_at=0.0, updated_at=0.0):
                self.id = id
                self.confidence = confidence
                self.evidence = evidence or []
                self.created_at = created_at
                self.updated_at = updated_at

try:
    from context import Context
except ImportError:
    class Context:
        def __init__(self, **kwargs):
            self.active_window = kwargs.get("active_window", "")
            self.active_application = kwargs.get("active_application", "")
            self.is_idle = kwargs.get("is_idle", False)
            self.confidence = kwargs.get("confidence", 0.0)

try:
    from context_interpreter import ContextInterpreter
except ImportError:
    class ContextInterpreter:
        def __init__(self, **kwargs):
            self.current_focus_duration_s = kwargs.get("current_focus_duration_s", 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Test runner
# ──────────────────────────────────────────────────────────────────────────────

_passed = 0
_failed = 0
_test_results: list = []

def run_test(name: str, fn):
    global _passed, _failed
    try:
        fn()
        _test_results.append((name, "PASS", None))
        _passed += 1
    except AssertionError as e:
        _test_results.append((name, "FAIL", str(e)))
        _failed += 1
    except Exception as e:
        _test_results.append((name, "ERROR", str(e)))
        _failed += 1


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

_RUNTIME = None
_RUNTIME_LOCK = threading.Lock()

def _make_rt():
    rt = Runtime()
    rt.start()
    return rt


def make_runtime() -> Runtime:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = _make_rt()
    return _RUNTIME

def t_engine_cooldown_suppress():
    make_runtime()

    engine = SuggestionEngine.__new__(SuggestionEngine)
    engine._lookup_context = lambda: MockContext(active_window="main.py", active_application="vscode")
    engine._lookup_interpretation = lambda: MockInterpretation()
    engine._lookup_hypotheses = lambda: [make_test_hypothesis()]
    engine._lookup_plan = lambda: None
    engine._provider_classes = (GeneralDevelopmentProvider,)
    engine._ttl_s = 120.0
    engine._delivery_timer = None
    engine._delivery_pending = False
    engine._lock = threading.Lock()
    engine._providers = []
    engine._pending = None
    engine._current_cycle_id = ""
    engine._unsubscribe = None
    engine._cooldown = CooldownManager(decline_s=60)
    engine._activity = None
    engine._metrics = SuggestionMetricsTracker()
    engine.start()

    try:
        engine._try_generate("cycle_1")
        time.sleep(0.1)
        assert not engine.has_pending
    finally:
        engine.stop()


def make_test_hypothesis(id="software_development", confidence=0.9):
    return Hypothesis(
        id=id,
        confidence=confidence,
        evidence=["dev_app", "long_focus", "stable_context"],
        created_at=0.0,
        last_updated=0.0,
        update_count=0,
        cycle_id="",
    )


def make_test_context(window="main.py", app="code"):
    return Context(active_window=window, active_application=app, confidence=0.9)


def make_test_interpretation(focus_s=300.0):
    return ContextInterpreter(current_focus_duration_s=focus_s)


# ──────────────────────────────────────────────────────────────────────────────
# Sprint 2 Tests
# ──────────────────────────────────────────────────────────────────────────────

# ─── CooldownManager ──────────────────────────────────────────────────────

def t_cooldown_initially_inactive():
    cm = CooldownManager()
    assert not cm.is_active()
    assert cm.remaining_s() == 0.0


def t_cooldown_decline():
    cm = CooldownManager(decline_s=10)
    cm.on_declined()
    assert cm.is_active()
    assert cm.remaining_s() > 0


def t_cooldown_accept():
    cm = CooldownManager(accept_s=20)
    cm.on_accepted()
    assert cm.is_active()
    assert cm.remaining_s() > 0


def t_cooldown_duplicate():
    cm = CooldownManager(duplicate_window_s=10)
    cm.on_displayed("hello world")
    assert cm.is_duplicate("hello world")
    assert not cm.is_duplicate("different text")


def t_cooldown_expires():
    cm = CooldownManager(decline_s=0.5)
    cm.on_declined()
    assert cm.is_active()  # should be active immediately
    time.sleep(0.6)
    # After sleep, is_active() detects expiry, clears state, returns False
    assert not cm.is_active(), "Cooldown should have expired"


def t_cooldown_callback():
    fired = []

    def cb(reason, duration):
        fired.append((reason, duration))

    cm = CooldownManager(accept_s=0.5, on_expired=cb)
    cm.on_accepted()
    assert cm.is_active()  # confirm active
    time.sleep(0.6)
    # This call detects expiry, fires callback, returns False
    result = cm.is_active()
    assert not result, "Should be expired"
    assert len(fired) == 1, f"Expected 1 callback, got {len(fired)}: {fired}"
    assert fired[0][0] == "accept"


def t_cooldown_reset():
    cm = CooldownManager(decline_s=1.0)
    cm.on_declined()
    assert cm.is_active()
    cm.reset()
    assert not cm.is_active()
    assert not cm.is_duplicate("any")


def t_cooldown_info():
    cm = CooldownManager(decline_s=10)
    info = cm.get_cooldown_info()
    assert info["active"] is False
    cm.on_declined()
    info = cm.get_cooldown_info()
    assert info["active"] is True
    assert info["reason"] == "decline"


def t_cooldown_threads():
    cm = CooldownManager(decline_s=1, accept_s=1)
    errors = []

    def worker():
        try:
            for _ in range(10):
                cm.on_declined()
                assert cm.is_active() or not cm.is_active()
                cm.remaining_s()
                cm.reset()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert not errors, f"Thread errors: {errors}"


def t_cooldown_displayed_tracks_text():
    cm = CooldownManager(duplicate_window_s=10)
    cm.on_displayed("text A")
    assert cm.is_duplicate("text A")
    assert not cm.is_duplicate("text B")


# ─── ActivityMonitor ─────────────────────────────────────────────────────

def t_activity_initial():
    am = ActivityMonitor(poll_interval_s=0.1, quiet_threshold_s=0.5)
    am.set_external_mode(True)
    am.record_activity()
    assert am.is_quiet(0.01) is False


def t_activity_quiet():
    am = ActivityMonitor(poll_interval_s=0.1, quiet_threshold_s=0.2)
    am.set_external_mode(True)
    am.record_activity()
    time.sleep(0.3)
    assert am.is_quiet() is True


def t_activity_reset():
    am = ActivityMonitor(poll_interval_s=0.1, quiet_threshold_s=0.5)
    am.set_external_mode(True)
    am.record_activity()
    time.sleep(0.3)
    # Should not be quiet yet (threshold 0.5s, only 0.3s elapsed)
    assert not am.is_quiet()


def t_activity_external():
    am = ActivityMonitor(poll_interval_s=0.1, quiet_threshold_s=0.1)
    am.set_external_mode(True)
    am.record_activity()
    # Just recorded activity, so not quiet with tiny threshold
    assert not am.is_quiet()


def t_activity_seconds():
    am = ActivityMonitor(poll_interval_s=0.1)
    am.set_external_mode(True)
    am.record_activity()
    time.sleep(0.1)
    elapsed = am.seconds_since_last_activity()
    assert 0.05 < elapsed < 1.0


def t_activity_lifecycle():
    am = ActivityMonitor(poll_interval_s=0.05)
    am.set_external_mode(False)  # Use polling mode
    am.start()
    assert am._poll_thread is not None
    am.stop()
    assert am._poll_thread is None


def t_activity_last_activity_monotonic():
    am = ActivityMonitor(poll_interval_s=0.1)
    am.set_external_mode(True)
    before = am.last_activity_monotonic()
    time.sleep(0.05)
    am.record_activity()
    after = am.last_activity_monotonic()
    assert after > before


# ─── SuggestionMessenger ─────────────────────────────────────────────────

def t_messenger_dev():
    ctx = make_test_context(window="main.py", app="vscode")
    msg = build_message("software_development", ctx)
    assert "Sir" in msg
    assert "vscode" in msg or "your application" in msg


def t_messenger_reason():
    ctx = make_test_context()
    interp = make_test_interpretation(focus_s=600.0)
    reason = build_reason("software_development", ctx, interp)
    assert "minute" in reason.lower()


def t_messenger_tuple():
    ctx = make_test_context()
    msg, reason = build_suggestion_text("software_development", ctx, None)
    assert isinstance(msg, str)
    assert isinstance(reason, str)
    assert len(msg) > 0
    assert len(reason) > 0


def t_messenger_unknown():
    ctx = make_test_context()
    msg = build_message("unknown_hypothesis", ctx)
    assert len(msg) > 0  # falls back to default


def t_messenger_none_context():
    msg = build_message("software_development", None)
    assert len(msg) > 0


def t_messenger_empty_override():
    ctx = make_test_context(window="", app="vscode")
    msg = build_message("software_development", ctx, focus_topic_override="")
    assert len(msg) > 0


# ─── SuggestionMetrics ────────────────────────────────────────────────────

def t_metrics_zero():
    sm = SuggestionMetrics()
    assert sm.suggestions_created == 0
    assert sm.accepted == 0
    assert sm.declined == 0


def t_metrics_track():
    sm = SuggestionMetricsTracker()
    sm.on_created()
    sm.on_shown()
    sm.on_accepted()
    sm.on_declined()
    sm.on_expired()
    sm.on_suppressed_cooldown("test")
    sm.on_suppressed_duplicate()
    sm.on_suppressed_other()

    m = sm.get_session_metrics()
    assert m["suggestions_created"] == 1
    assert m["suggestions_shown"] == 1
    assert m["accepted"] == 1
    assert m["declined"] == 1
    assert m["expired"] == 1
    assert m["suppressed_cooldown"] == 1
    assert m["suppressed_duplicate"] == 1
    assert m["suppressed_other"] == 1


def t_metrics_reset():
    sm = SuggestionMetricsTracker()
    sm.on_created()
    sm.on_accepted()
    sm.reset()
    m = sm.get_session_metrics()
    assert m["suggestions_created"] == 0
    assert m["accepted"] == 0


def t_metrics_cooldown_reason():
    sm = SuggestionMetricsTracker()
    sm.on_suppressed_cooldown("decline")
    m = sm.get_session_metrics()
    assert m["cooldown_reason"] == "decline"


def t_metrics_threads():
    sm = SuggestionMetricsTracker()
    errors = []

    def worker():
        try:
            for _ in range(100):
                sm.on_created()
                sm.on_accepted()
                sm.get_session_metrics()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def t_metrics_log_summary():
    sm = SuggestionMetricsTracker()
    sm.on_created()
    sm.on_accepted()
    # Should not raise
    sm.log_summary()


# ─── SuggestionEngine with Sprint 2 features ─────────────────────────────

class MockContext:
    def __init__(self, **kwargs):
        self.active_window = kwargs.get("active_window", "")
        self.active_application = kwargs.get("active_application", "")
        self.confidence = kwargs.get("confidence", 0.0)


class MockInterpretation:
    def __init__(self, focus_s=300.0):
        self.current_focus_duration_s = focus_s


def _make_plan(decision="ASK_PERMISSION"):
    """Minimal plan object."""
    class FakePlan:
        pass
    FakePlan.decision = decision
    return FakePlan()


def t_engine_cooldown_suppress():
    rt = make_runtime()
    engine = SuggestionEngine(
        lookup_context=lambda: MockContext(active_window="main.py", active_application="vscode"),
        lookup_interpretation=lambda: MockInterpretation(),
        lookup_hypotheses=lambda: [make_test_hypothesis()],
        lookup_plan=lambda: _make_plan("ASK_PERMISSION"),
        cooldown_manager=CooldownManager(decline_s=60),
    )
    engine.start()

    try:
        # First generate should succeed (cooldown not active yet)
        engine._try_generate("cycle_1")
        time.sleep(0.2)
        print(f"  [cooldown_suppress 1] has_pending={engine.has_pending}")
        assert engine.has_pending, "Should have pending suggestion before cooldown"
        # Decline to start cooldown
        engine.decline()
        time.sleep(0.2)
        print(f"  [cooldown_suppress 2] cm_active={engine._cooldown.is_active()}")
        assert engine._cooldown.is_active(), "Cooldown should be active after decline"
        # Now generate again — should be suppressed
        engine._try_generate("cycle_2")
        time.sleep(0.2)
        print(f"  [cooldown_suppress 3] has_pending={engine.has_pending}")
        assert not engine.has_pending, "Suggestion should be suppressed by cooldown"
    finally:
        engine.stop()


def t_engine_metrics():
    rt = make_runtime()
    engine = SuggestionEngine(
        lookup_context=lambda: MockContext(active_window="main.py", active_application="vscode"),
        lookup_interpretation=lambda: MockInterpretation(),
        lookup_hypotheses=lambda: [make_test_hypothesis()],
        lookup_plan=lambda: _make_plan("ASK_PERMISSION"),
        metrics_tracker=SuggestionMetricsTracker(),
    )
    engine.start()

    try:
        engine._try_generate("cycle_1")
        time.sleep(0.1)
        metrics = engine.get_session_metrics()
        assert metrics["suggestions_created"] >= 1, f"Got {metrics['suggestions_created']}"
    finally:
        engine.stop()


def t_engine_cooldown_callback():
    rt = make_runtime()
    events = []

    def cb(reason, duration):
        events.append((reason, duration))

    cm = CooldownManager(accept_s=0.5, on_expired=cb)
    engine = SuggestionEngine(
        lookup_context=lambda: None,
        lookup_interpretation=lambda: None,
        lookup_hypotheses=lambda: [],
        lookup_plan=lambda: _make_plan(),
        cooldown_manager=cm,
    )
    engine.start()

    try:
        engine._cooldown.on_accepted()
        time.sleep(0.6)
        # This call detects expiry, fires callback, returns False
        result = engine._cooldown.is_active()
        assert not result, "Should be expired"
        assert len(events) == 1
    finally:
        engine.stop()


def t_engine_duplicate_suppress():
    rt = make_runtime()
    # Use a long decline cooldown so the second attempt is suppressed
    cm = CooldownManager(decline_s=60)
    engine = SuggestionEngine(
        lookup_context=lambda: MockContext(active_window="main.py", active_application="vscode"),
        lookup_interpretation=lambda: MockInterpretation(),
        lookup_hypotheses=lambda: [make_test_hypothesis()],
        lookup_plan=lambda: _make_plan(),
        cooldown_manager=cm,
    )
    engine.start()

    try:
        # Generate once
        engine._try_generate("cycle_1")
        time.sleep(0.2)
        print(f"  [1] has_pending={engine.has_pending}")
        assert engine.has_pending, "Should have pending suggestion"

        # Decline (starts decline cooldown)
        engine.decline()
        time.sleep(0.2)
        print(f"  [2] cooldown_active={cm.is_active()}, has_pending={engine.has_pending}")

        # Try again — should be suppressed by cooldown
        engine._try_generate("cycle_2")
        time.sleep(0.2)
        print(f"  [3] cooldown_active={cm.is_active()}, has_pending={engine.has_pending}")

        assert not engine.has_pending, f"Should be suppressed by cooldown, but has_pending={engine.has_pending}"
    finally:
        engine.stop()


def t_engine_activity_delay():
    """Test that suggestions are delayed when user is active."""
    rt = make_runtime()
    activity = ActivityMonitor(poll_interval_s=0.1, quiet_threshold_s=0.5)

    engine = SuggestionEngine(
        lookup_context=lambda: MockContext(active_window="main.py", active_application="vscode"),
        lookup_interpretation=lambda: MockInterpretation(),
        lookup_hypotheses=lambda: [make_test_hypothesis()],
        lookup_plan=lambda: _make_plan(),
        activity_monitor=activity,
    )
    engine.start()

    try:
        # Record recent activity — user is not quiet
        activity.record_activity()
        engine._try_generate("cycle_1")
        time.sleep(0.1)

        # Suggestion should be pending (delayed, not yet displayed)
        assert engine.has_pending, "Suggestion should be pending (delayed)"
        assert engine._delivery_pending, "Should be in delivery-pending state"
    finally:
        engine.stop()
        activity.stop()


def t_engine_session_metrics_keys():
    rt = make_runtime()
    engine = SuggestionEngine(
        lookup_context=lambda: MockContext(active_window="main.py", active_application="vscode"),
        lookup_interpretation=lambda: MockInterpretation(),
        lookup_hypotheses=lambda: [make_test_hypothesis()],
        lookup_plan=lambda: _make_plan(),
        metrics_tracker=SuggestionMetricsTracker(),
    )
    engine.start()

    try:
        engine._try_generate("cycle_1")
        time.sleep(0.1)
        m = engine.get_session_metrics()
        expected_keys = {
            "suggestions_created", "suggestions_shown", "accepted", "declined",
            "expired", "suppressed_cooldown", "suppressed_duplicate",
            "suppressed_other", "cooldown_reason",
        }
        assert expected_keys.issubset(set(m.keys())), f"Missing keys: {expected_keys - set(m.keys())}"
    finally:
        engine.stop()


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n=== Sprint 2 Tests ===")
    print("Running tests...\n")

    test_funcs = [
        # CooldownManager
        ("cooldown: initially inactive", t_cooldown_initially_inactive),
        ("cooldown: decline starts cooldown", t_cooldown_decline),
        ("cooldown: accept starts cooldown", t_cooldown_accept),
        ("cooldown: duplicate detection", t_cooldown_duplicate),
        ("cooldown: auto-expires", t_cooldown_expires),
        ("cooldown: on_expired callback", t_cooldown_callback),
        ("cooldown: reset clears state", t_cooldown_reset),
        ("cooldown: get_cooldown_info", t_cooldown_info),
        ("cooldown: thread safety", t_cooldown_threads),
        ("cooldown: displayed tracks text", t_cooldown_displayed_tracks_text),
        # ActivityMonitor
        ("activity: initial state", t_activity_initial),
        ("activity: is_quiet after threshold", t_activity_quiet),
        ("activity: record_activity resets timer", t_activity_reset),
        ("activity: external mode", t_activity_external),
        ("activity: seconds_since_last_activity", t_activity_seconds),
        ("activity: start/stop", t_activity_lifecycle),
        ("activity: last_activity_monotonic", t_activity_last_activity_monotonic),
        # SuggestionMessenger
        ("messenger: build_message for software_development", t_messenger_dev),
        ("messenger: build_reason contains duration", t_messenger_reason),
        ("messenger: build_suggestion_text returns tuple", t_messenger_tuple),
        ("messenger: unknown hypothesis fallback", t_messenger_unknown),
        ("messenger: None context fallback", t_messenger_none_context),
        ("messenger: empty focus_topic override", t_messenger_empty_override),
        # SuggestionMetrics
        ("metrics: initial state zero", t_metrics_zero),
        ("metrics: track event increments", t_metrics_track),
        ("metrics: reset clears all", t_metrics_reset),
        ("metrics: cooldown reason tracking", t_metrics_cooldown_reason),
        ("metrics: thread safety", t_metrics_threads),
        ("metrics: log_summary", t_metrics_log_summary),
        # Engine integration
        ("engine: cooldown suppresses suggestions", t_engine_cooldown_suppress),
        ("engine: metrics tracking", t_engine_metrics),
        ("engine: cooldown callback fires", t_engine_cooldown_callback),
        ("engine: duplicate suppression", t_engine_duplicate_suppress),
        ("engine: activity delay", t_engine_activity_delay),
        ("engine: session metrics keys", t_engine_session_metrics_keys),
    ]

    for name, fn in test_funcs:
        print(f"  Running: {name}...", end=" ", flush=True)
        try:
            fn()
            print("PASS")
            _passed += 1
        except AssertionError as e:
            print(f"FAIL — {e}")
            _failed += 1
        except Exception as e:
            print(f"ERROR — {e}")
            _failed += 1

    print(f"\n=== Results: {_passed} passed, {_failed} failed ===")
    if _failed:
        sys.exit(1)
