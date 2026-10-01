"""Backend application settings and environment resolution."""

import os
from pydantic_settings import BaseSettings if False else object

class Settings:
    PROJECT_NAME: str = "Story Continuation Generator API"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Path resolution with fallbacks
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        os.path.join(BASE_DIR, "artifacts", "models", "story_lstm.keras")
    )
    LEGACY_MODEL_PATH: str = os.path.join(BASE_DIR, "models", "best", "story_lstm.keras")
    
    TOKENIZER_PATH: str = os.getenv(
        "TOKENIZER_PATH",
        os.path.join(BASE_DIR, "artifacts", "tokenizers", "tokenizer.json")
    )
    LEGACY_TOKENIZER_PATH: str = os.path.join(BASE_DIR, "artifacts", "tokenizer_vocab.json")
    
    METRICS_PATH: str = os.path.join(BASE_DIR, "artifacts", "metrics", "metrics.json")
    DIST_DIR: str = os.path.join(BASE_DIR, "frontend", "dist")

    # CORS origins
    ALLOWED_ORIGINS: list = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

settings = Settings()
