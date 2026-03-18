"""Application configuration."""

import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # SofaScore
    sofascore_base_url: str = os.getenv(
        "SOFASCORE_BASE_URL", "https://api.sofascore.com/api/v1"
    )
    sofascore_cache_ttl: int = int(os.getenv("SOFASCORE_CACHE_TTL", "300"))

    # Database
    database_url: str = os.getenv(
        "DATABASE_URL", f"sqlite+aiosqlite:///{BASE_DIR}/data/predictions.db"
    )

    # Prediction
    default_confidence_threshold: int = int(
        os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "65")
    )
    parlay_min_confidence: int = int(os.getenv("PARLAY_MIN_CONFIDENCE", "70"))
    max_parlay_legs: int = int(os.getenv("MAX_PARLAY_LEGS", "15"))

    # Bankroll
    default_bankroll: float = float(os.getenv("DEFAULT_BANKROLL", "1000"))
    kelly_fraction: float = float(os.getenv("KELLY_FRACTION", "0.25"))

    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", str(BASE_DIR / "logs" / "agent.log"))


settings = Settings()
