"""
J.A.R.V.I.S. — FastAPI server.

Endpoints:
  POST   /chat              - one-shot conversation
  POST   /tts               - synthesize text to MP3 (returns base64)
  GET    /memory            - list all facts + patterns + recent moods
  DELETE /memory/{key}      - forget a fact
  GET    /rules             - list proactivity rules
  POST   /rules             - add a rule
  PATCH  /rules/{id}        - enable/disable a rule
  DELETE /rules/{id}        - remove a rule
  POST   /admin/reflect     - trigger reflection pass now
  POST   /admin/proactivity - trigger proactivity tick now
  GET    /health            - liveness
  WS     /ws                - bidirectional; auth required first

Lifespan: starts APScheduler (reflection every N hours, proactivity every N min).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import (
    Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from auth import passphrase
from brain import proactivity, reflection
from config import settings
from jarvis_core import chat as jarvis_chat
from memory import long_term
from runtime.runtime import Runtime, set_runtime
from streaming_patch import apply_streaming_patch
from tts import speaker


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("jarvis.main")


# -----------------------------------------------------------------------------
# WebSocket manager
# -----------------------------------------------------------------------------

class WebSocketManager:
    """Tracks connected WebSocket clients and broadcasts messages."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        logger.info("WS connected. total=%d", len(self._clients))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        logger.info("WS disconnected. total=%d", len(self._clients))

    async def broadcast(self, payload: dict) -> None:
        async with self._lock:
            clients = list(self._clients)
        dead: list[WebSocket] = []
        for c in clients:
            try:
                await c.send_text(json.dumps(payload))
            except Exception:
                dead.append(c)
        if dead:
            async with self._lock:
                for d in dead:
                    self._clients.discard(d)


ws_manager = WebSocketManager()


# -----------------------------------------------------------------------------
# Inference Engine fallbacks (used only when the Context Engine is unavailable)
# -----------------------------------------------------------------------------

def _noop_context():
    """Default Context used when the Context Engine hasn't started."""
    from context import Context
    return Context()


def _noop_interpretation():
    """Default ContextInterpretation used when the Context Engine is missing."""
    from context import ContextInterpretation, InteractionLevel, StabilityLevel
    return ContextInterpretation(
        interaction_level=InteractionLevel.IDLE,
        session_duration_s=0.0,
        context_stability_s=0.0,
        context_stability=StabilityLevel.FLUCTUATING,
        application_switch_frequency=0.0,
        current_focus_duration_s=0.0,
        confidence=0.0,
    )


# -----------------------------------------------------------------------------
# Lifespan: scheduler + initialisation
# -----------------------------------------------------------------------------

scheduler: Optional[AsyncIOScheduler] = None
main_suggestion_engine = None
main_ws_bridge: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler, main_suggestion_engine, main_ws_bridge

    # Init DB + seed default rules
    await long_term.init_db()
    await long_term.seed_default_rules()

    # ── JARVIS 3.0 Runtime (outermost shell — starts before all services) ──
    jarvis_runtime = Runtime()
    jarvis_runtime.start()
    logger.info(
        "JARVIS 3.0 Runtime started (state=%s)",
        jarvis_runtime.state.value,
    )

    # ── JARVIS 3.0 Event Coalescer (buffers Observer events) ──────────────────
    _event_coalescer = None
    try:
        from runtime.event_coalescer import EventCoalescer
        _event_coalescer = EventCoalescer(window_ms=150)
        _event_coalescer.start()
        logger.info("JARVIS 3.0 Event Coalescer started")
    except Exception as exc:
        logger.warning("Event Coalescer could not start (continuing): %s", exc)

    # ── JARVIS 3.0 Observer (first Operating Intelligence service) ─────────────
    try:
        from observer import Observer
        observer_service = Observer(poll_interval_s=1.0, idle_threshold_s=60.0)
        jarvis_runtime.register_service(observer_service)
        logger.info("JARVIS 3.0 Observer registered")
    except Exception as exc:
        logger.warning("Observer could not start (continuing without it): %s", exc)

    # ── JARVIS 3.0 Context Engine (subscribes to Observer events) ─────────────
    context_engine_service = None
    try:
        from context import ContextEngine
        context_engine_service = ContextEngine()
        jarvis_runtime.register_service(context_engine_service)
        logger.info("JARVIS 3.0 Context Engine registered")
    except Exception as exc:
        logger.warning("Context Engine could not start (continuing without it): %s", exc)

    # ── JARVIS 3.0 Inference Engine (subscribes to Context events) ─────────────
    try:
        from inference import InferenceEngine
        if context_engine_service is not None:
            inference_engine_service = InferenceEngine(
                lookup_context=context_engine_service.get_current_context,
                lookup_interpretation=context_engine_service.get_current_interpretation,
            )
        else:
            # No Context Engine — engine can't evaluate. Register a no-op shim
            # so the rest of the system can still query for empty hypotheses.
            inference_engine_service = InferenceEngine(
                lookup_context=(lambda: _noop_context()),
                lookup_interpretation=(lambda: _noop_interpretation()),
            )
        jarvis_runtime.register_service(inference_engine_service)
        logger.info("JARVIS 3.0 Inference Engine registered")
    except Exception as exc:
        logger.warning("Inference Engine could not start (continuing without it): %s", exc)

    # ── JARVIS 3.0 Cognitive Logger (passive EventBus observer) ───────────────
    try:
        from observability import CognitiveLogger
        _cognitive_logger = CognitiveLogger()
        # Bind to the live services for richer structured entries.
        if context_engine_service is not None:
            _cognitive_logger.bind(
                get_context=context_engine_service.get_current_context,
                get_interpretation=context_engine_service.get_current_interpretation,
                get_hypotheses=(
                    inference_engine_service.get_current_hypotheses
                    if inference_engine_service is not None
                    else None
                ),
            )
        jarvis_runtime.register_service(_cognitive_logger)
        logger.info("JARVIS 3.0 Cognitive Logger registered")
    except Exception as exc:
        logger.warning("Cognitive Logger could not start: %s", exc)

    # ── JARVIS 3.0 Planner (decides whether to act) ──────────────────────────
    try:
        from planner import Planner
        planner_service = Planner(
            lookup_hypotheses=(
                inference_engine_service.get_current_hypotheses
                if inference_engine_service is not None
                else None
            ),
            lookup_context=(
                context_engine_service.get_current_context
                if context_engine_service is not None
                else None
            ),
            lookup_interpretation=(
                context_engine_service.get_current_interpretation
                if context_engine_service is not None
                else None
            ),
        )
        jarvis_runtime.register_service(planner_service)
        logger.info("JARVIS 3.0 Planner registered")
    except Exception as exc:
        logger.warning("Planner could not start (continuing without it): %s", exc)

    # ── JARVIS 3.0 Suggestion Engine (subscribes to Planner decisions) ─────────
    try:
        from suggestion import SuggestionEngine
        suggestion_engine_service = SuggestionEngine(
            lookup_context=(
                context_engine_service.get_current_context
                if context_engine_service is not None
                else None
            ),
            lookup_interpretation=(
                context_engine_service.get_current_interpretation
                if context_engine_service is not None
                else None
            ),
            lookup_hypotheses=(
                inference_engine_service.get_current_hypotheses
                if inference_engine_service is not None
                else None
            ),
            lookup_plan=(
                planner_service.get_current_plan
                if planner_service is not None
                else None
            ),
        )
        jarvis_runtime.register_service(suggestion_engine_service)
        logger.info("JARVIS 3.0 Suggestion Engine registered")
    except Exception as exc:
        logger.warning("Suggestion Engine could not start (continuing without it): %s", exc)

    # ── JARVIS 3.0 WebSocket bridge: EventBus → Frontend ─────────────────────
    # Bridges sync EventBus events (from the Operating Intelligence
    # pipeline) to async WebSocket broadcasts.  The Runtime's EventBus
    # publishes synchronously from its own thread, so we must use
    # asyncio.run_coroutine_threadsafe() to schedule the broadcast on
    # the FastAPI event loop.
    try:
        import asyncio as _asyncio

        def _on_suggestion_created(event) -> None:
            """Publish a suggestion_available message to all WS clients."""
            try:
                payload = {
                    "type": "suggestion_available",
                    "suggestion_id": getattr(event, "suggestion_id", ""),
                    "title": getattr(event, "title", ""),
                    "message": "",  # message set after ask()
                    "provider": getattr(event, "provider", ""),
                    "confidence": getattr(event, "confidence", 0.0),
                    "cycle_id": getattr(event, "cycle_id", ""),
                }
                _asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast(payload),
                    _asyncio.get_event_loop(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("WS bridge: suggestion_created failed: %s", exc)

        def _on_suggestion_asked(event) -> None:
            """Update the frontend with the message ready to speak."""
            try:
                payload = {
                    "type": "suggestion_available",
                    "suggestion_id": getattr(event, "suggestion_id", ""),
                    "title": "",
                    "message": getattr(event, "message", ""),
                    "provider": "",
                    "confidence": 0.0,
                    "cycle_id": getattr(event, "cycle_id", ""),
                }
                _asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast(payload),
                    _asyncio.get_event_loop(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("WS bridge: suggestion_asked failed: %s", exc)

        def _on_suggestion_accepted(event) -> None:
            try:
                payload = {
                    "type": "suggestion_accepted",
                    "suggestion_id": getattr(event, "suggestion_id", ""),
                    "provider": getattr(event, "provider", ""),
                    "cycle_id": getattr(event, "cycle_id", ""),
                }
                _asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast(payload),
                    _asyncio.get_event_loop(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("WS bridge: suggestion_accepted failed: %s", exc)

        def _on_suggestion_declined(event) -> None:
            try:
                payload = {
                    "type": "suggestion_declined",
                    "suggestion_id": getattr(event, "suggestion_id", ""),
                    "cycle_id": getattr(event, "cycle_id", ""),
                }
                _asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast(payload),
                    _asyncio.get_event_loop(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("WS bridge: suggestion_declined failed: %s", exc)

        def _on_suggestion_expired(event) -> None:
            try:
                payload = {
                    "type": "suggestion_expired",
                    "suggestion_id": getattr(event, "suggestion_id", ""),
                    "cycle_id": getattr(event, "cycle_id", ""),
                }
                _asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast(payload),
                    _asyncio.get_event_loop(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("WS bridge: suggestion_expired failed: %s", exc)

        # Subscribe to the EventBus for suggestion lifecycle events.
        jarvis_runtime.event_bus.subscribe(_on_suggestion_created)
        jarvis_runtime.event_bus.subscribe(_on_suggestion_asked)
        jarvis_runtime.event_bus.subscribe(_on_suggestion_accepted)
        jarvis_runtime.event_bus.subscribe(_on_suggestion_declined)
        jarvis_runtime.event_bus.subscribe(_on_suggestion_expired)

        # Store references so the WS endpoint can call accept/decline.
        main_suggestion_engine = suggestion_engine_service
        main_ws_bridge = {
            "suggestion_created": _on_suggestion_created,
            "suggestion_asked": _on_suggestion_asked,
            "suggestion_accepted": _on_suggestion_accepted,
            "suggestion_declined": _on_suggestion_declined,
            "suggestion_expired": _on_suggestion_expired,
        }
        logger.info("JARVIS 3.0 WebSocket bridge registered")
    except Exception as exc:
        logger.warning("WS bridge could not start: %s", exc)
        main_suggestion_engine = None
        main_ws_bridge = {}

    # Configure scheduler
    scheduler = AsyncIOScheduler(timezone=settings.TIMEZONE)

    async def _reflection_job():
        try:
            res = await reflection.run_reflection_pass()
            logger.info("Reflection pass result: %s", res)
        except Exception as e:
            logger.exception("Reflection job failed: %s", e)

    async def _proactivity_job():
        try:
            fired = await proactivity.proactivity_tick()
            if fired:
                logger.info("Proactivity fired: %s", fired)
        except Exception as e:
            logger.exception("Proactivity job failed: %s", e)

    scheduler.add_job(
        _reflection_job,
        "interval",
        hours=settings.REFLECTION_INTERVAL_HOURS,
        id="reflection",
    )
    scheduler.add_job(
        _proactivity_job,
        "interval",
        minutes=settings.PROACTIVITY_TICK_MINUTES,
        id="proactivity",
    )

    # ── Screen memory: capture screen every 10 minutes ─────────────────────────────
    async def _screen_memory_job():
        try:
            from memory.screen_memory import capture_and_describe
            await capture_and_describe()
        except Exception as e:
            logger.debug("Screen memory capture failed: %s", e)

    scheduler.add_job(
        _screen_memory_job,
        "interval",
        minutes=5,
        id="screen_memory",
    )

    # ── Call monitor: check for incoming calls every 5 seconds ──────────────
    _call_monitor_running = True
    _last_call_state: dict = {}  # tracks active calls to avoid duplicates

    async def _call_monitor_job():
        """Poll Android for incoming calls and announce them."""
        nonlocal _call_monitor_running, _last_call_state
        if not _call_monitor_running:
            return
        try:
            from tools.system import monitor_calls
            r = await monitor_calls()
            if r.get("incoming_call"):
                # Extract caller info from the telecom log
                raw = r.get("details", "")
                # Try to get the phone number from dumpsys
                caller = "Unknown"
                for line in raw.split("\n"):
                    if "number:" in line.lower():
                        caller = line.strip().split("number:")[-1].strip()
                        break
                    if "display" in line.lower() and ":" in line:
                        caller = line.strip().split("display")[-1].strip().split(":")[-1].strip()
                        break

                key = f"call_{caller}"
                if key != _last_call_state.get("key"):
                    _last_call_state = {"key": key}
                    logger.info("Incoming call detected from: %s", caller)

                    # Clean the caller string (remove asterisks and special chars)
                    import re
                    caller_clean = re.sub(r'[*#]+', '', caller).strip()
                    if not caller_clean:
                        caller_clean = "an unknown number"
                    announce_text = f"Incoming call from {caller_clean}"

                    # Announce via WebSocket toast
                    msg = {
                        "type": "call_alert",
                        "caller": caller_clean,
                        "message": announce_text,
                    }
                    await ws_manager.broadcast(msg)

                    # Speak it aloud via TTS
                    try:
                        from tts.speaker import speak
                        speak(announce_text)  # Kokoro speak - non-blocking
                    except Exception as e:
                        logger.warning("TTS for call announcement failed: %s", e)
        except Exception as e:
            logger.warning("Call monitor error: %s", e)

    scheduler.add_job(_call_monitor_job, "interval", seconds=5, id="call_monitor")
    # Activate sentence streaming (patches jarvis_core.chat non-destructively)
    try:
        await apply_streaming_patch()
        logger.info("Sentence streaming activated")
    except Exception as e:
        logger.warning("Streaming patch failed (continuing without streaming): %s", e)

    # Start OS context tracking for Operating Intelligence
    try:
        from brain.os_context import start_context_tracking
        start_context_tracking()
        logger.info("OS context tracking started")
    except Exception as e:
        logger.warning("Context tracking failed (continuing without OS context): %s", e)

    scheduler.start()
    logger.info(
        "Scheduler started (reflection every %sh, proactivity every %sm)",
        settings.REFLECTION_INTERVAL_HOURS, settings.PROACTIVITY_TICK_MINUTES,
    )

    yield

    # Shutdown
    try:
        # ── JARVIS 3.0 Runtime shutdown (outermost shell — stops after all services) ──
        try:
            jarvis_runtime.stop()
            logger.info("JARVIS 3.0 Runtime stopped")
        except Exception as exc:
            logger.error("JARVIS 3.0 Runtime shutdown error: %s", exc)

        if scheduler:
            scheduler.shutdown(wait=False)
    finally:
        set_runtime(None)
    logger.info("JARVIS shutdown complete.")


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

app = FastAPI(
    title="J.A.R.V.I.S.",
    description="Just A Rather Very Intelligent System",
    version="0.1.0-phase1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # local dev only; tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------

class ChatIn(BaseModel):
    message: str
    stream: bool = False

class ChatOut(BaseModel):
    reply: str
    mood: str
    tools_used: list[str] = Field(default_factory=list)
    audio_b64: Optional[str] = None

class TTSIn(BaseModel):
    text: str
    voice: Optional[str] = None

class TTSOut(BaseModel):
    audio_b64: str

class RuleIn(BaseModel):
    name: str
    condition: dict
    action: str
    action_payload: Optional[dict] = None
    cooldown_hours: float = 12.0

class PassphraseIn(BaseModel):
    passphrase: str

class AuthOut(BaseModel):
    ok: bool
    reason: Optional[str] = None


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "configured": settings.is_configured(),
        "user": settings.USER_NAME,
        "tz": settings.TIMEZONE,
        "version": "0.1.0-phase1",
    }


@app.post("/internal/voice-start")
async def voice_start():
    """Broadcast to all WebSocket clients to start voice recognition."""
    await ws_manager.broadcast({"type": "start_voice"})
    return {"ok": True}


# -----------------------------------------------------------------------------
# Auth
# -----------------------------------------------------------------------------

@app.post("/auth/status", response_model=AuthOut)
async def auth_status():
    set_ = await passphrase.is_passphrase_set()
    return {"ok": True, "reason": "set" if set_ else "not_set"}


@app.post("/auth/set", response_model=AuthOut)
async def auth_set(body: PassphraseIn):
    await passphrase.set_passphrase(body.passphrase)
    return {"ok": True}


@app.post("/auth/verify", response_model=AuthOut)
async def auth_verify(body: PassphraseIn):
    ok = await passphrase.verify_passphrase(body.passphrase)
    return {"ok": ok, "reason": "match" if ok else "mismatch"}


@app.post("/auth/clear", response_model=AuthOut)
async def auth_clear():
    await passphrase.clear_passphrase()
    return {"ok": True, "reason": "cleared"}


# -----------------------------------------------------------------------------
# Chat
# -----------------------------------------------------------------------------

@app.post("/chat", response_model=ChatOut)
async def chat_endpoint(body: ChatIn):
    if not settings.is_configured():
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured")
    res = await jarvis_chat(settings.USER_ID, body.message)
    return ChatOut(**res)


# -----------------------------------------------------------------------------
# TTS
# -----------------------------------------------------------------------------

@app.post("/tts", response_model=TTSOut)
async def tts_endpoint(body: TTSIn):
    audio = await speaker.synthesize_bytes(body.text, voice=body.voice)
    if not audio:
        raise HTTPException(status_code=400, detail="empty text")
    return TTSOut(audio_b64=base64.b64encode(audio).decode("ascii"))


@app.post("/interrupt")
async def interrupt_endpoint():
    """Interrupt any currently playing speech."""
    interrupted = speaker.interrupt()
    return {"interrupted": interrupted, "speaking": speaker.speaking()}


@app.get("/speaking")
async def speaking_endpoint():
    """Check if JARVIS is currently speaking."""
    return {"speaking": speaker.speaking()}


@app.get("/voices")
async def voices_endpoint():
    return speaker.list_voices()


# -----------------------------------------------------------------------------
# Voice Control (Native STT + TTS)
# -----------------------------------------------------------------------------

_voice_coordinator = None

class VoiceStatus(BaseModel):
    running: bool
    active: bool
    listening: bool

class VoiceCommand(BaseModel):
    action: str  # start, stop, activate, deactivate
    text: Optional[str] = None  # For manual command input

@app.get("/voice/status", response_model=VoiceStatus)
async def voice_status():
    """Get voice assistant status."""
    # Voice status — handled by jarvis_embedded.py, not the backend
    def is_voice_assistant_running(): return False
    def is_jarvis_awake(): return False
    def is_listening(): return False
    return VoiceStatus(
        running=is_voice_assistant_running(),
        active=is_jarvis_awake(),
        listening=is_listening()
    )

@app.post("/voice/control")
async def voice_control(body: VoiceCommand):
    """Control voice assistant — disabled, handled by jarvis_embedded.py."""
    return {"ok": False, "status": "voice controlled by jarvis_embedded"}


# -----------------------------------------------------------------------------
# Memory
# -----------------------------------------------------------------------------

@app.get("/memory")
async def memory_list():
    return {
        "facts": await long_term.list_facts(limit=200),
        "patterns": await long_term.list_patterns(min_confidence=0.0, limit=200),
        "recent_moods": await long_term.recent_moods(limit=20),
        "stats": {
            "fact_count": len(await long_term.list_facts(limit=10000)),
            "pattern_count": len(await long_term.list_patterns(min_confidence=0.0, limit=10000)),
        },
    }


@app.delete("/memory/{key}")
async def memory_delete(key: str):
    ok = await long_term.delete_fact(key)
    if not ok:
        raise HTTPException(status_code=404, detail=f"no fact with key: {key}")
    return {"ok": True, "forgot": key}


# -----------------------------------------------------------------------------
# Activity log (PC-control audit trail)
# -----------------------------------------------------------------------------

@app.get("/activity")
async def activity_list(limit: int = Query(50, ge=1, le=500),
                        kind: Optional[str] = Query(None)):
    return await long_term.recent_activity(limit=limit, kind=kind)


# -----------------------------------------------------------------------------
# Approve / deny a pending tool action
# -----------------------------------------------------------------------------

class ApprovalIn(BaseModel):
    approved: bool = True
    reason: str = ""


@app.post("/approve/{action_id}")
async def approve_action(action_id: str, body: ApprovalIn = Body(default_factory=ApprovalIn)):
    """Resolve a pending approval_request that JARVIS pushed to the HUD."""
    from jarvis_core import resolve_approval
    ok = resolve_approval(action_id, body.approved, body.reason)
    if not ok:
        raise HTTPException(status_code=404,
                            detail="no pending action with that id (maybe timed out?)")
    return {"ok": True, "approved": body.approved, "action_id": action_id}


# -----------------------------------------------------------------------------
# Filesystem helpers exposed for direct UI use (read-only)
# -----------------------------------------------------------------------------

@app.get("/files/search")
async def files_search(q: str = Query(..., min_length=1),
                       max_results: int = Query(25, ge=1, le=100),
                       ext: Optional[str] = Query(None,
                                                   description="comma-sep, e.g. pdf,docx")):
    from tools import files as files_tool
    exts = [e.strip() for e in ext.split(",")] if ext else None
    return files_tool.find_files(q, max_results=max_results, extensions=exts)


@app.post("/screenshot")
async def screenshot_now():
    """Capture screen + stream inline; doesn't need user approval."""
    from tools import system as system_tool
    from jarvis_core import _stream_screenshot
    result = await system_tool.screenshot(monitor=0)
    if result.get("ok"):
        await _stream_screenshot(result)
        await long_term.log_activity(
            "screenshot", result.get("path", ""), "ok",
            f"{result.get('size_bytes', 0)} bytes",
        )
    return result


# -----------------------------------------------------------------------------
# Media shortcut — play music/video without going through the chat pipeline
# -----------------------------------------------------------------------------
class MediaIn(BaseModel):
    query: str = ""


@app.post("/media/play")
async def media_play(body: MediaIn):
    """Direct play endpoint. Skips the LLM round-trip and just runs play_media."""
    from tools import system as system_tool
    from jarvis_core import _broadcast_toast, _log_activity
    q = (body.query or "").strip()
    if not q:
        return {"ok": False, "error": "empty query"}
    _broadcast_toast(f"🎵 Searching for '{q}'…", kind="info")
    r = await system_tool.play_media(q)
    title = r.get("title") or q
    if r.get("ok"):
        _broadcast_toast(f"🎵 Playing {title}", kind="ok")
    else:
        _broadcast_toast(f"🎵 Couldn't find {q}", kind="error")
    _log_activity("play_media", title,
                  "ok" if r.get("ok") else "error",
                  r.get("uri") or r.get("error", ""))
    return r


# -----------------------------------------------------------------------------
# Direct system commands — bypass chat completely (for critical operations)
# -----------------------------------------------------------------------------

class VolumeIn(BaseModel):
    level: int


@app.post("/volume/set")
async def set_volume(body: VolumeIn):
    """Set system volume directly, bypassing chat."""
    from tools import system as system_tool
    from jarvis_core import _broadcast_toast, _log_activity
    level = max(0, min(100, int(body.level)))
    r = await system_tool.set_volume(level)
    _broadcast_toast(f"🔊 Volume {level}%", kind="ok" if r.get("ok") else "error")
    _log_activity("set_volume", str(level),
                  "ok" if r.get("ok") else "error",
                  "")
    return r


class OpenAppIn(BaseModel):
    name: str


@app.post("/app/open")
async def open_app(body: OpenAppIn):
    """Launch an app directly, bypassing chat."""
    from tools import system as system_tool
    from jarvis_core import _broadcast_toast, _log_activity
    r = await system_tool.open_app(body.name)
    if r.get("ok"):
        _broadcast_toast(f"🖥️ Opened {r.get('display', body.name)}", kind="ok")
    else:
        _broadcast_toast(f"❌ Couldn't find '{body.name}'", kind="error")
    _log_activity("open_app", body.name,
                  "ok" if r.get("ok") else "error",
                  r.get("error", ""))
    return r


class KillProcessIn(BaseModel):
    name: str


@app.post("/process/kill")
async def kill_process(body: KillProcessIn):
    """Kill a process directly, bypassing chat."""
    from tools import system as system_tool
    from jarvis_core import _broadcast_toast, _log_activity
    # The kill_process function already queries PIDs first
    r = await system_tool.kill_process(body.name)
    if r.get("ok"):
        _broadcast_toast(f"🗑️ Terminated {body.name} processes", kind="ok")
    else:
        _broadcast_toast(f"❌ {r.get('error', 'No processes found')}", kind="error")
    _log_activity("kill_process", body.name,
                  "ok" if r.get("ok") else "error",
                  r.get("error", ""))
    return r


class WriteFileIn(BaseModel):
    path: str
    content: str
    overwrite: bool = True


@app.post("/file/write")
async def write_file(body: WriteFileIn):
    """Write a file directly, bypassing chat."""
    from tools import files as files_tool
    from jarvis_core import _broadcast_toast, _log_activity
    r = await files_tool.write_file(body.path, body.content, body.overwrite)
    if r.get("ok"):
        _broadcast_toast(f"💾 Saved {body.path}", kind="ok")
    else:
        _broadcast_toast(f"❌ Write failed: {r.get('error', '')}", kind="error")
    _log_activity("write_file", body.path,
                  "ok" if r.get("ok") else "error",
                  r.get("error", ""))
    return r


@app.post("/volume/set/{level}")
async def set_volume_quick(level: int):
    """Quick volume setting: POST /volume/set/50 sets to 50%"""
    from tools import system as system_tool
    from jarvis_core import _broadcast_toast, _log_activity
    level = max(0, min(100, level))
    r = await system_tool.set_volume(level)
    _broadcast_toast(f"🔊 Volume {level}%", kind="ok" if r.get("ok") else "error")
    _log_activity("set_volume", str(level),
                  "ok" if r.get("ok") else "error",
                  "")
    return r


@app.get("/quick/kill/{name}")
async def quick_kill(name: str):
    """Quick process kill: GET /quick/kill/notepad kills notepad"""
    return await kill_process(KillProcessIn(name=name))


# -----------------------------------------------------------------------------
# Settings: trust destructive actions
# -----------------------------------------------------------------------------
class TrustDestructiveIn(BaseModel):
    enabled: bool


@app.post("/settings/trust-destructive")
async def settings_trust_destructive(body: TrustDestructiveIn):
    """When on, destructive tools (kill_process, shutdown, etc.) skip approval."""
    from jarvis_core import set_trust_destructive
    set_trust_destructive(body.enabled)
    return {"ok": True, "trust_destructive_actions": body.enabled}


@app.get("/settings/trust-destructive")
async def settings_trust_destructive_get():
    from jarvis_core import _trust_destructive
    return {"enabled": await _trust_destructive()}


# -----------------------------------------------------------------------------
# Rules
# -----------------------------------------------------------------------------

@app.get("/rules")
async def rules_list(enabled_only: bool = Query(False)):
    return await long_term.list_rules(enabled_only=enabled_only)


@app.post("/rules")
async def rules_add(body: RuleIn):
    rid = await long_term.add_rule(
        name=body.name,
        condition=body.condition,
        action=body.action,
        action_payload=body.action_payload,
        cooldown_hours=body.cooldown_hours,
    )
    return {"id": rid, "name": body.name}


@app.patch("/rules/{rule_id}")
async def rules_toggle(rule_id: int, enabled: bool = Query(...)):
    await long_term.set_rule_enabled(rule_id, enabled)
    return {"ok": True}


@app.delete("/rules/{rule_id}")
async def rules_delete(rule_id: int):
    await long_term.delete_rule(rule_id)
    return {"ok": True}


# -----------------------------------------------------------------------------
# Admin / debug
# -----------------------------------------------------------------------------

@app.post("/admin/reflect")
async def admin_reflect():
    res = await reflection.run_reflection_pass()
    return res


@app.post("/admin/proactivity")
async def admin_proactivity():
    fired = await proactivity.proactivity_tick()
    return {"fired": fired}


@app.post("/admin/clear-conversations")
async def admin_clear_convos():
    await long_term.clear_conversations()
    return {"ok": True}


# -----------------------------------------------------------------------------
# WebSocket
# -----------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """
    Bidirectional channel. Protocol:
      1. Client must send {"type": "auth", "passphrase": "..."} first
         OR {"type": "auth", "skip": true} if no passphrase is set
      2. Server replies {"type": "auth_result", "ok": bool}
      3. If ok: client can send {"type": "user_message", "content": "..."}
         Server replies with {"type": "jarvis_reply", "reply": "...", "mood": "..."}
         OR {"type": "status", "state": "thinking|..."} during processing
      4. Server can also push {"type": "jarvis_message", ...} for proactive messages
    """
    await ws_manager.connect(websocket)
    authenticated = False

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "error": "invalid_json"}))
                continue

            mtype = msg.get("type")

            # --- Auth handshake ---
            if mtype == "auth":
                if msg.get("skip"):
                    authenticated = True
                else:
                    pp = msg.get("passphrase", "")
                    if not await passphrase.is_passphrase_set():
                        # No passphrase configured — open access on first run
                        authenticated = True
                    else:
                        authenticated = await passphrase.verify_passphrase(pp)
                await websocket.send_text(json.dumps({
                    "type": "auth_result",
                    "ok": authenticated,
                }))
                if not authenticated:
                    await websocket.send_text(json.dumps({
                        "type": "jarvis_message",
                        "source": "system",
                        "content": "Voice identification failed. This conversation is locked.",
                        "mood": "neutral",
                    }))
                    await asyncio.sleep(0.5)
                    break

                # Quick greeting - no AI needed
                hour = datetime.now().hour
                if hour < 12:
                    greeting = "Good morning, Sir. JARVIS is online and ready."
                elif hour < 17:
                    greeting = "Good afternoon, Sir. JARVIS is online and ready."
                elif hour < 21:
                    greeting = "Good evening, Sir. JARVIS is online and ready."
                else:
                    greeting = "Good evening, Sir, it late, lets begin with our batman work. JARVIS is online and ready."

                # Send greeting to frontend
                await websocket.send_text(json.dumps({
                    "type": "jarvis_message",
                    "source": "jarvis",
                    "content": greeting,
                    "mood": "neutral",
                }))

                # Play greeting TTS on PC speakers
                import threading
                def _greet():
                    try:
                        # Use Kokoro TTS
                        from tts.speaker import speak
                        speak(greeting)
                    except Exception as e:
                        logger.debug("Greeting TTS failed: %s", e)

                threading.Thread(target=_greet, daemon=True).start()
                continue

            if not authenticated:
                await websocket.send_text(json.dumps({
                    "type": "error", "error": "not_authenticated"
                }))
                continue

            # --- User message ---
            if mtype == "user_message":
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                # Route command through the router — handles local intent
                # detection and falls back to Claude when needed.
                from command_router import route_command
                result = await route_command(command=content)
                response_text = result.reply
                mood = result.mood
                tools_used = result.tools_used

                if result.local_executed:
                    # Local intent: send immediate response, no Claude involved
                    await websocket.send_text(json.dumps({
                        "type": "status", "state": "speaking"
                    }))
                    await websocket.send_text(json.dumps({
                        "type": "jarvis_reply",
                        "reply": response_text,
                        "mood": mood,
                        "tools_used": tools_used,
                    }))

                    # TTS in background
                    async def _generate_tts(ws, reply_text):
                        try:
                            import threading
                            clean = re.sub(r'[*_`#>\[\]|]+', ' ', reply_text)
                            clean = re.sub(r'\s+', ' ', clean).strip()
                            if not clean:
                                return
                            def _play():
                                try:
                                    speaker.speak(clean)
                                except Exception as e:
                                    logger.debug("Kokoro playback failed: %s", e)
                            threading.Thread(target=_play, daemon=True).start()
                        except Exception as e:
                            logger.debug("TTS failed: %s", e)
                        finally:
                            try:
                                await ws.send_text(json.dumps({"type": "status", "state": "idle"}))
                            except Exception:
                                pass

                    asyncio.create_task(_generate_tts(websocket, response_text))
                    continue

                # No local intent — proceeded with Claude API
                res = {"reply": response_text, "mood": mood, "tools_used": tools_used, "audio_b64": None}
                # Status: speaking
                await websocket.send_text(json.dumps({
                    "type": "status", "state": "speaking"
                }))
                await websocket.send_text(json.dumps({
                    "type": "jarvis_reply",
                    "reply": res["reply"],
                    "mood": res["mood"],
                    "tools_used": res.get("tools_used", []),
                }))
                # Spawn TTS generation as background task — reply is sent immediately
                # Also plays on PC speakers directly (no browser interaction needed)
                async def _generate_tts(ws, reply_text, mood_val):
                    try:
                        # Strip JSON prefix and markdown from reply text
                        json_match = re.match(r'\s*\{[^}]+\}\s*', reply_text)
                        clean = reply_text[json_match.end():] if json_match else reply_text
                        clean = re.sub(r'[*_`#>\[\]|]+', ' ', clean)
                        clean = re.sub(r'\s+', ' ', clean).strip()

                        if not clean:
                            return

                        # Use Kokoro TTS directly (non-blocking via thread)
                        import threading

                        def _play():
                            try:
                                speaker.speak(clean)  # Kokoro blocks until done
                            except Exception as e:
                                logger.debug("Kokoro playback failed: %s", e)

                        threading.Thread(target=_play, daemon=True).start()

                        # Also send audio to frontend if available
                        try:
                            audio_bytes = await speaker.synthesize_bytes(clean)
                            if audio_bytes:
                                await ws.send_text(json.dumps({
                                    "type": "audio",
                                    "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
                                    "format": "raw",
                                    "mood": mood_val,
                                }))
                        except Exception:
                            pass
                    except Exception as e:
                        logger.debug("TTS failed: %s", e)
                    finally:
                        try:
                            await ws.send_text(json.dumps({"type": "status", "state": "idle"}))
                        except Exception:
                            pass

                # Fire and forget — reply goes out instantly, audio follows
                asyncio.create_task(_generate_tts(websocket, res["reply"], res.get("mood", "neutral")))

            # --- Suggestion accept/decline (JARVIS 3.0 proactive) ---
            elif mtype == "accept_suggestion":
                await _handle_accept_suggestion(websocket, msg)

            elif mtype == "decline_suggestion":
                await _handle_decline_suggestion(websocket, msg)

            elif mtype == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

            else:
                await websocket.send_text(json.dumps({
                    "type": "error", "error": f"unknown_message_type: {mtype}"
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("WS error: %s", e)
    finally:
        await ws_manager.disconnect(websocket)


# -----------------------------------------------------------------------------
# JARVIS 3.0 — Suggestion handlers (Sprint 1: First Autonomous Conversation)
# -----------------------------------------------------------------------------

async def _handle_accept_suggestion(
    websocket: WebSocket,
    msg: dict,
) -> None:
    """Accept a pending Suggestion, execute via LLM, and reply."""
    suggestion_id = (msg.get("suggestion_id") or "").strip()
    if not suggestion_id:
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": "missing_suggestion_id",
        }))
        return

    engine = main_suggestion_engine
    if engine is None:
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": "suggestion_engine_unavailable",
        }))
        return

    # Mark the suggestion as ASKED (transitions to user-facing state)
    # then accept it (transitions to ACCEPTED).  Both are no-ops if
    # the suggestion no longer exists.
    engine.ask()

    # Tell the frontend we're processing.
    await websocket.send_text(json.dumps({
        "type": "status",
        "state": "thinking",
    }))

    accepted = engine.accept()
    if accepted is None:
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": "no_pending_suggestion",
        }))
        return

    # Execute the prepared_request via the configured LLM provider.
    from llm_executor import LLMExecutor
    executor = LLMExecutor()
    request = accepted.prepared_request

    # provider.chat() is async, so just await it directly.
    response = await executor.execute(request)

    if not response.success:
        logger.warning("LLM executor failed: %s", response.error)
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": "llm_execution_failed",
            "detail": response.error,
        }))
        await websocket.send_text(json.dumps({
            "type": "status",
            "state": "idle",
        }))
        return

    reply_text = (response.content or "").strip()
    if not reply_text:
        reply_text = "I have nothing to add right now, Sir."

    # Send the reply + TTS in parallel — exact same shape as user_message.
    await websocket.send_text(json.dumps({
        "type": "jarvis_reply",
        "reply": reply_text,
        "mood": "neutral",
        "tools_used": [],
        "source": "proactive",
    }))
    await websocket.send_text(json.dumps({
        "type": "status",
        "state": "speaking",
    }))

    async def _playback_then_idle() -> None:
        try:
            import threading as _th
            def _play() -> None:
                try:
                    from tts.speaker import speak
                    speak(reply_text)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("TTS playback failed: %s", exc)
            _th.Thread(target=_play, daemon=True).start()
        except Exception as exc:  # noqa: BLE001
            logger.debug("TTS scheduling failed: %s", exc)
        finally:
            try:
                await websocket.send_text(json.dumps({
                    "type": "status", "state": "idle",
                }))
            except Exception:
                pass

    asyncio.create_task(_playback_then_idle())


async def _handle_decline_suggestion(
    websocket: WebSocket,
    msg: dict,
) -> None:
    """Decline a pending Suggestion. No LLM call, no TTS."""
    if main_suggestion_engine is None:
        return
    main_suggestion_engine.decline()
    await websocket.send_text(json.dumps({
        "type": "status",
        "state": "idle",
    }))


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )