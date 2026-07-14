# Phase 4: Reliability Improvements

## Summary
Added robustness and self-healing capabilities to voice pipeline components.

### ✅ 1. STT Queue Overflow Protection
- **What**: Bounded queues to prevent memory OOM
- **Changes**:
  - `_audio_queue`: maxsize=50 (6s of audio at 0.5s/chunk)
  - `_result_queue`: maxsize=20 (prevents backlog buildup)
  - `_push_audio()`: Helper that drops oldest chunk when full
  - `get_audio_drop_count()`: Diagnostic to count drops
- **Result**: If STT outpaces processing, system won't crash with OOM

### ✅ 2. STT Atomic Restart Function
- **What**: `restart_listening()` helper for watchdog recovery
- **Changes**:
  - Stops cleanly (force-kill after timeout)
  - Clears queues and drops counter
  - Reinitializes all resources
- **Result**: Watchdog can force-restart STT when it hangs

### ✅ 3. STT Graceful Shutdown
- **What**: Improved `stop_listening()`
- **Changes**:
  - 5s thread join timeout (was 2s)
  - Checks if thread survived timeout
  - Clears `_transcribe_thread` reference
- **Result**: Cleaner shutdown, no zombie threads

### ✅ 4. Watchdog TTS Thread Recovery
- **What**: Detect dead TTS threads and recover
- **Changes**:
  - `_check_tts_thread()` in watchdog
  - Detects when `is_speaking()` was True but thread died
  - Clears `_last_speak_success` to mark as recovered
- **Result**: TTS dead thread doesn't permanently block

### ✅ 5. Speech Collector Restart with Backoff
- **What**: Coordinator restarts collector on crash with exponential backoff
- **Changes**:
  - Tracks restart count
  - Waits `min(restarts * 2, 10)` seconds between restarts
  - Logs restart attempts to avoid silent failures
- **Result**: System recovers from speech collector crashes gracefully

## Testing
All features tested and validated:
- Bounded queues drop items, not memory
- restart_listening() is callable
- Watchdog singleton and methods work
- Backoff logic in place
- SAPI fallback wired correctly
- TTS thread detection functional
- Diagnostics available

## Status: COMPLETE
JARVIS voice pipeline is now significantly more resilient to crashes and resource exhaustion.