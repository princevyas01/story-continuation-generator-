"""Backend application settings and environment resolution."""

import os

class Settings:
    PROJECT_NAME: str = "Story Continuation Generator API"
    VERSION: str = "3.0.0"
    API_V1_STR: str = "/api/v1"
    
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Active model configuration: "v3" candidate activated as default; rollback to "v2" supported via env var
    ACTIVE_MODEL_VERSION: str = os.getenv("ACTIVE_MODEL_VERSION", "v3").lower()
    
    # Story LSTM v3 paths
    MODEL_V3_PATH: str = os.path.join(BASE_DIR, "artifacts", "models", "story_lstm_v3.keras")
    TOKENIZER_V3_PATH: str = os.path.join(BASE_DIR, "artifacts", "tokenizers", "tokenizer_v3.json")
    METADATA_V3_PATH: str = os.path.join(BASE_DIR, "artifacts", "metadata", "model_v3.json")
    TOKENIZER_V3_METADATA_PATH: str = os.path.join(BASE_DIR, "artifacts", "metadata", "tokenizer_v3_metadata.json")
    METRICS_V3_PATH: str = os.path.join(BASE_DIR, "artifacts", "metrics", "test_metrics_v3.json")
    TRAINING_HISTORY_V3_PATH: str = os.path.join(BASE_DIR, "artifacts", "metrics", "training_history_v3.json")
    SAMPLES_V3_PATH: str = os.path.join(BASE_DIR, "artifacts", "samples", "sample_generations_v3.json")
    TRAINING_REPORT_V3_PATH: str = os.path.join(BASE_DIR, "artifacts", "reports", "training_report_v3.md")
    
    # Story LSTM v2 paths (preserved for instant rollback)
    MODEL_V2_PATH: str = os.path.join(BASE_DIR, "artifacts", "models", "story_lstm_v2.keras")
    MODEL_V2_BEST_PATH: str = os.path.join(BASE_DIR, "models", "best", "story_lstm_v2.keras")
    TOKENIZER_V2_PATH: str = os.path.join(BASE_DIR, "artifacts", "tokenizers", "tokenizer_v2.json")
    METRICS_V2_PATH: str = os.path.join(BASE_DIR, "artifacts", "reports", "evaluation_comparison_v2.json")
    
    # Legacy v1 paths (preserved)
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
