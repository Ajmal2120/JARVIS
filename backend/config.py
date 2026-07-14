"""
J.A.R.V.I.S. Configuration
Loads environment variables and exposes settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the backend directory
BACKEND_DIR = Path(__file__).parent
load_dotenv(BACKEND_DIR / ".env")


class Settings:
    """Centralized settings for JARVIS."""

    # --- LLM Provider ---
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "claude").lower()

    # --- API keys ---
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_BASE_URL: str = os.getenv("ANTHROPIC_BASE_URL", "")
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

    # --- Identity ---
    USER_NAME: str = os.getenv("JARVIS_USER_NAME", "Sir")
    TIMEZONE: str = os.getenv("JARVIS_TIMEZONE", "Asia/Kolkata")

    # --- Location ---
    DEFAULT_LAT: float = float(os.getenv("DEFAULT_LAT", "26.4499"))
    DEFAULT_LON: float = float(os.getenv("DEFAULT_LON", "80.3319"))

    # --- TTS ---
    TTS_VOICE: str = os.getenv("TTS_VOICE", "en-GB-RyanNeural")
    TTS_RATE: str = os.getenv("TTS_RATE", "+0%")
    TTS_PITCH: str = os.getenv("TTS_PITCH", "+0Hz")

    # --- STT (Speech-to-Text) ---
    # Whisper model size: tiny, base, small, medium, large
    # Larger = more accurate but slower
    STT_MODEL: str = os.getenv("STT_MODEL", "small")
    STT_LANGUAGE: str = os.getenv("STT_LANGUAGE", "en")  # Default to English

    # --- Proactivity ---
    PROACTIVITY_ENABLED: bool = os.getenv("PROACTIVITY_ENABLED", "true").lower() == "true"
    PROACTIVITY_TICK_MINUTES: int = int(os.getenv("PROACTIVITY_TICK_MINUTES", "15"))
    REFLECTION_INTERVAL_HOURS: int = int(os.getenv("REFLECTION_INTERVAL_HOURS", "6"))

    # --- Cognitive Logging ---
    ENABLE_COGNITIVE_LOGGING: bool = os.getenv("ENABLE_COGNITIVE_LOGGING", "true").lower() == "true"
    COGNITIVE_LOG_PATH: Path = BACKEND_DIR / "logs" / "cognitive.log"


    # --- Server ---
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    # --- Paths ---
    DB_PATH: Path = BACKEND_DIR / "memory" / "jarvis.db"
    TTS_CACHE_DIR: Path = BACKEND_DIR / "tts" / "cache"
    UPLOAD_DIR: Path = BACKEND_DIR / "uploads"

    # --- User (Phase 1: single user) ---
    USER_ID: str = "default"

    def __post_init__(self):
        # Ensure directories exist
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def is_configured(self) -> bool:
        """Check if the essential configuration is present."""
        return bool(self.ANTHROPIC_API_KEY) and self.ANTHROPIC_API_KEY != "sk-ant-your-key-here"


settings = Settings()
settings.__post_init__()