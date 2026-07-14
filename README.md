# J.A.R.V.I.S.

> Just A Rather Very Intelligent System.
> Personal AI assistant inspired by Tony Stark's JARVIS — proactive, emotionally aware, time-of-day aware, and (Phase 2) with a custom-cloned voice.

---

## What this is

A full-stack AI assistant with:
- **Dynamic personality** — speaks differently at 2 AM vs 10 AM, differently when you're tired vs focused
- **Long-term memory** — remembers facts *and* learned behavioral patterns
- **Reflection** — every 6 hours, reviews conversations and updates what it knows
- **Proactivity** — pings you with morning briefings, reminders, and check-ins via browser, browser Notification API, and Windows toast
- **Voice in/out** — Browser Web Speech API (STT) + edge-tts in `en-GB-RyanNeural` (TTS)
- **Tools** — web search, weather, news, file ops
- **Voice lock** — passphrase gate before JARVIS engages

Voice cloning from MCU samples is **Phase 2** (same audio pipeline).

---

## Folder structure

```
jarvis/
├── backend/                 Python + FastAPI
│   ├── main.py              FastAPI app, WebSocket, scheduler
│   ├── jarvis_core.py       The brain — wires prompt, context, Claude
│   ├── config.py            Settings
│   ├── brain/
│   │   ├── system_prompt.py   Dynamic prompt builder
│   │   ├── context.py         Context enrichment
│   │   ├── sentiment.py       Mood inference
│   │   ├── reflection.py      Periodic self-update
│   │   └── proactivity.py     Rules engine
│   ├── memory/
│   │   ├── long_term.py       SQLite (facts, patterns, moods, rules, convos)
│   │   └── short_term.py      In-memory buffer
│   ├── tools/              search, files, system, apis
│   ├── tts/                edge-tts wrapper
│   ├── notifications/      Windows toast
│   ├── auth/               Passphrase gate (Phase 1)
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
└── frontend/
    └── index.html          Single-file HUD (no build step)
```

---

## Setup (local, Windows)

### 1. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # or: source .venv/bin/activate (Mac/Linux)
pip install -r requirements.txt
copy .env.example .env          # then edit .env to add your ANTHROPIC_API_KEY
```

### 2. Run the backend

```bash
uvicorn main:app --reload
```

Server will start on `http://127.0.0.1:8000`. Swagger docs at `/docs`.

### 3. Frontend

Open `frontend/index.html` directly in Chrome or Edge. It connects to `ws://localhost:8000/ws` automatically.

### 4. First run

1. Open the frontend — the lock screen appears.
2. Backend will accept any passphrase on first run (no passphrase configured yet).
3. To set one: `curl -X POST http://localhost:8000/auth/set -H "Content-Type: application/json" -d '{"passphrase":"your-secret-here"}'`
4. Refresh the frontend — now it will require your passphrase.

---

## Endpoints

| Method | Path                  | Purpose                                              |
|--------|-----------------------|------------------------------------------------------|
| GET    | `/health`             | Liveness + config status                             |
| POST   | `/auth/status`        | Is a passphrase set?                                 |
| POST   | `/auth/set`           | Set passphrase                                       |
| POST   | `/auth/verify`        | Verify passphrase                                    |
| POST   | `/auth/clear`         | Clear passphrase (dev only)                          |
| POST   | `/chat`               | One-shot conversation                                |
| POST   | `/tts`                | Synthesize text → base64 MP3                         |
| GET    | `/voices`             | List available TTS voices                            |
| GET    | `/memory`             | List facts, patterns, recent moods                   |
| DELETE | `/memory/{key}`       | Forget a fact                                        |
| GET    | `/rules`              | List proactivity rules                               |
| POST   | `/rules`              | Add a rule                                           |
| PATCH  | `/rules/{id}`         | Enable/disable a rule                                |
| DELETE | `/rules/{id}`         | Remove a rule                                        |
| POST   | `/admin/reflect`      | Trigger reflection pass now                          |
| POST   | `/admin/proactivity`  | Trigger proactivity tick now                         |
| POST   | `/admin/clear-conversations` | Wipe conversation history                      |
| WS     | `/ws`                 | Real-time bidirectional channel                      |

---

## WebSocket protocol

1. **Connect**, then send auth handshake:
   ```json
   { "type": "auth", "passphrase": "your-secret" }
   ```
   Or if no passphrase is set on the backend:
   ```json
   { "type": "auth", "skip": true }
   ```

2. **Server responds:**
   ```json
   { "type": "auth_result", "ok": true }
   ```

3. **Send a message:**
   ```json
   { "type": "user_message", "content": "What's the weather in Kanpur?" }
   ```

4. **Server sends a stream of events:**
   - `{ "type": "status", "state": "thinking" }`
   - `{ "type": "status", "state": "speaking" }`
   - `{ "type": "jarvis_reply", "reply": "...", "mood": "neutral", "tools_used": ["get_weather"] }`
   - `{ "type": "audio", "audio_b64": "..." }`
   - `{ "type": "status", "state": "idle" }`

5. **Proactive pushes** (from JARVIS without user input):
   ```json
   { "type": "jarvis_message", "source": "proactive", "action": "morning_briefing", "content": "..." }
   ```

---

## Phase 1 default behaviors

- **Time-of-day variants:** tone shifts at 5am, 9am, 6pm, 10pm, 1am
- **Mood-aware:** tired/frustrated/sad → softer, playful → matches energy
- **Reflection:** every 6 hours, reviews last 50 messages, learns patterns
- **Proactivity rules** seeded automatically:
  - `morning_briefing` at 8 AM (only if no chat today)
  - `late_night_nudge` after 11 PM (if chatting > 2 hours)
  - `silence_checkin` after 3 days of inactivity
  - User reminders via `set_reminder` tool

---

## Phase 2 (next)

- Voice cloning from MCU JARVIS clips (XTTS / RVC pipeline)
- Web Audio API voice sentiment (pitch/energy → mood)
- Voice biometrics (replace passphrase gate)
- Full OS system control (open apps, screenshots)

---

## Cost

- Anthropic Claude API: ~$1–3/month for personal daily use
- Everything else: free

---

## Troubleshooting

- **"ANTHROPIC_API_KEY not configured"** — edit `backend/.env` and add your key
- **Frontend stuck on "OFFLINE"** — backend not running, or wrong port
- **No microphone** — Chrome/Edge only; check site permissions (lock icon in URL bar)
- **No audio in frontend** — click anywhere first (browser autoplay policy)
- **Stuck authentication** — check `/auth/status` returns `set` or `not_set` as expected

---

## License

Personal project. Do whatever you want with it.