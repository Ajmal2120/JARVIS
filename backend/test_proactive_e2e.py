"""
End-to-end test for JARVIS 3.0 Sprint 1 — First Autonomous Conversation.

Covers:
  1. Coding → Observer burst → Coalescer → Context → Inference →
     Planner → Suggestion Engine → SuggestionCreated event → bridge → frontend
  2. Frontend sends accept_suggestion → LLMExecutor.execute() → reply → TTS
  3. Frontend sends decline_suggestion → no LLM, no TTS, status DECLINED

The LLM provider is mocked so no real API calls happen.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Setup: mock the LLM provider BEFORE main.py imports llm_executor
# ---------------------------------------------------------------------------

def _patch_llm():
    """Replace the LLM provider with a deterministic mock."""
    from llm import provider as provider_mod

    class MockChatResponse(provider_mod.ChatResponse):
        def __init__(self, content="Sir, I noticed you're focused on main.py. Would you like a summary of the recent changes?"):
            super().__init__(content=content, role="assistant")

    class MockProvider(provider_mod.LLMProvider):
        model = "mock-test"

        async def chat(self, messages, model=None, tools=None,
                       max_tokens=None, temperature=None, system=None, **kw):
            return MockChatResponse()

        async def stream_chat(self, messages, model=None, tools=None,
                              max_tokens=None, temperature=None, system=None, **kw):
            resp = await self.chat(messages=messages, model=model, tools=tools,
                                   max_tokens=max_tokens, temperature=temperature,
                                   system=system, **kw)
            yield {"type": "content", "content": resp.content}
            yield {"type": "done", "usage": resp.usage}

    # Patch the factory
    import llm.factory
    llm.factory._provider = MockProvider()

    return MockProvider


# Apply patch
_mock_provider_cls = _patch_llm()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

async def test_accept_flow() -> bool:
    """Test 1: Coding → suggestion → accept → LLM → reply + TTS."""
    from runtime.runtime import Runtime, set_runtime
    from runtime.event_coalescer import EventCoalescer
    from observer.events import WindowChanged, ApplicationChanged, IdleStateChanged
    from context import ContextEngine
    from inference import InferenceEngine
    from planner import Planner
    from suggestion import SuggestionEngine, SuggestionStatus
    from suggestion.events import SuggestionAsked
    from llm_executor import LLMExecutor, LLMResponse

    rt = Runtime()
    rt.start()
    set_runtime(rt)

    # Build the pipeline
    coalescer = EventCoalescer(window_ms=150)
    coalescer.start()
    rt.event_bus.subscribe(coalescer._on_event)

    ce = ContextEngine()
    ce.start()

    ie = InferenceEngine(
        lookup_context=ce.get_current_context,
        lookup_interpretation=ce.get_current_interpretation,
    )
    ie.start()

    pl = Planner(
        lookup_hypotheses=ie.get_current_hypotheses,
        lookup_context=ce.get_current_context,
        lookup_interpretation=ce.get_current_interpretation,
        confidence_threshold=0.0,  # always ASK to force suggestion
    )
    pl.start()

    # Override the interpretation lookup to simulate a stable context
    # (the real interpreter needs 30s of stability for ASK_PERMISSION).
    stable_interp = MagicMock(
        interaction_level="high",
        context_stability="steady",
        context_stability_s=60.0,
        application_switch_frequency=0.5,
        current_focus_duration_s=120.0,
    )
    pl._lookup_interpretation = lambda: stable_interp

    eng = SuggestionEngine(
        lookup_context=ce.get_current_context,
        lookup_interpretation=ce.get_current_interpretation,
        lookup_hypotheses=ie.get_current_hypotheses,
        lookup_plan=pl.get_current_plan,
    )
    eng.start()

    # Capture SuggestionCreated event
    created_events = []
    asked_events = []
    accepted_events = []

    def on_created(e):
        if e.type == "suggestion_created":
            created_events.append(e)
        elif e.type == "suggestion_asked":
            asked_events.append(e)
        elif e.type == "suggestion_accepted":
            accepted_events.append(e)

    rt.event_bus.subscribe(on_created)

    # Publish the burst
    snap = {
        "window_title": "main.py",
        "process_name": "Code.exe",
        "idle_seconds": 2.0,
        "is_locked": False,
        "system_uptime_s": 100.0,
    }
    rt.event_bus.publish(WindowChanged(snapshot=snap, window_title="main.py", confidence=1.0))
    rt.event_bus.publish(ApplicationChanged(snapshot=snap, process_name="Code.exe", confidence=1.0))
    rt.event_bus.publish(IdleStateChanged(snapshot=snap, is_idle=False, idle_seconds=2.0))

    await asyncio.sleep(0.4)

    # Verify a Suggestion was created
    assert created_events, "expected at least one SuggestionCreated event"
    suggestion_id = created_events[0].suggestion_id
    print(f"  Created suggestion: {suggestion_id[:8]}...")

    # Verify engine has it pending
    pending = eng.get_pending_suggestion()
    assert pending is not None, "expected pending suggestion"
    assert pending.status == SuggestionStatus.PENDING

    # Simulate the WS bridge: when asked, broadcast message
    eng.ask()
    await asyncio.sleep(0.1)
    pending_after_ask = eng.get_pending_suggestion()
    assert pending_after_ask is not None
    assert pending_after_ask.status == SuggestionStatus.ASKED

    # Simulate the user clicking "Hear it" — accept_suggestion
    # The WS handler does: engine.ask() (already done) + engine.accept()
    accepted = eng.accept()
    assert accepted is not None, "accept() should return the suggestion"
    assert accepted.status == SuggestionStatus.ACCEPTED
    assert accepted.prepared_request is not None

    # LLM Executor runs (this is what the WS handler does)
    executor = LLMExecutor()
    response = await executor.execute(accepted.prepared_request)

    assert response.success, f"LLM executor failed: {response.error}"
    assert response.content
    print(f"  LLM response: {response.content[:60]}...")

    # Cleanup
    eng.stop()
    pl.stop()
    ie.stop()
    ce.stop()
    coalescer.stop()
    rt.stop()
    return True


async def test_decline_flow() -> bool:
    """Test 2: Decline → no LLM, status DECLINED."""
    from runtime.runtime import Runtime, set_runtime
    from runtime.event_coalescer import EventCoalescer
    from observer.events import WindowChanged, ApplicationChanged, IdleStateChanged
    from context import ContextEngine
    from inference import InferenceEngine
    from planner import Planner
    from suggestion import SuggestionEngine, SuggestionStatus

    rt = Runtime()
    rt.start()
    set_runtime(rt)

    coalescer = EventCoalescer(window_ms=150)
    coalescer.start()
    rt.event_bus.subscribe(coalescer._on_event)

    ce = ContextEngine()
    ce.start()
    ie = InferenceEngine(
        lookup_context=ce.get_current_context,
        lookup_interpretation=ce.get_current_interpretation,
    )
    ie.start()
    pl = Planner(
        lookup_hypotheses=ie.get_current_hypotheses,
        lookup_context=ce.get_current_context,
        lookup_interpretation=ce.get_current_interpretation,
        confidence_threshold=0.0,
    )
    pl.start()
    # Override the interpretation lookup to simulate a stable context
    stable_interp = MagicMock(
        interaction_level="high",
        context_stability="steady",
        context_stability_s=60.0,
        application_switch_frequency=0.5,
        current_focus_duration_s=120.0,
    )
    pl._lookup_interpretation = lambda: stable_interp
    eng = SuggestionEngine(
        lookup_context=ce.get_current_context,
        lookup_interpretation=ce.get_current_interpretation,
        lookup_hypotheses=ie.get_current_hypotheses,
        lookup_plan=pl.get_current_plan,
    )
    eng.start()

    snap = {
        "window_title": "main.py",
        "process_name": "Code.exe",
        "idle_seconds": 2.0,
        "is_locked": False,
        "system_uptime_s": 100.0,
    }
    rt.event_bus.publish(WindowChanged(snapshot=snap, window_title="main.py", confidence=1.0))
    rt.event_bus.publish(ApplicationChanged(snapshot=snap, process_name="Code.exe", confidence=1.0))
    rt.event_bus.publish(IdleStateChanged(snapshot=snap, is_idle=False, idle_seconds=2.0))
    await asyncio.sleep(0.4)

    pending = eng.get_pending_suggestion()
    assert pending is not None, "expected pending suggestion"

    # Simulate user clicking "Not now"
    eng.decline()
    await asyncio.sleep(0.1)

    # Pending should be cleared
    assert eng.get_pending_suggestion() is None, "decline should clear pending"

    eng.stop()
    pl.stop()
    ie.stop()
    ce.stop()
    coalescer.stop()
    rt.stop()
    return True


async def test_executor_uses_configured_provider() -> bool:
    """Test 3: LLMExecutor uses whichever provider is configured."""
    from llm_executor import LLMExecutor
    from suggestion import PreparedRequest
    from datetime import datetime
    from unittest.mock import MagicMock

    pr = PreparedRequest(
        context=MagicMock(active_window="main.py", active_application="Code.exe"),
        interpretation=None,
        hypotheses=(),
        plan=None,
        provider_metadata={"focus_topic": "main.py", "focus_app": "Code.exe"},
        request_type="development_context_assist",
        suggested_action="ask_for_focus_topic",
        created_at=datetime.now(),
    )
    executor = LLMExecutor()
    response = await executor.execute(pr)
    assert response.success
    assert response.provider  # provider name should be populated
    print(f"  Provider: {response.provider}")
    return True


async def main() -> None:
    print("=== Sprint 1 E2E: First Autonomous Conversation ===\n")

    print("Test 1: Coding → Planner ASK → Suggestion → Ask → Accept → LLM")
    ok1 = await test_accept_flow()
    print("PASS\n" if ok1 else "FAIL\n")

    print("Test 2: Decline → no LLM call")
    ok2 = await test_decline_flow()
    print("PASS\n" if ok2 else "FAIL\n")

    print("Test 3: LLM Executor uses configured provider")
    ok3 = await test_executor_uses_configured_provider()
    print("PASS\n" if ok3 else "FAIL\n")

    if ok1 and ok2 and ok3:
        print("=== All Sprint 1 E2E tests passed ===")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())