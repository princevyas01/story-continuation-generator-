"""Main FastAPI application entry point with lifespan lifecycle and CORS."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.app.model_service import model_service
from backend.app.health import health_router
from backend.app.routes import api_router
from backend.app.static_server import setup_static_serving

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and tokenizer artifacts at application startup."""
    print("=" * 60)
    print("Starting Story Continuation Generator FastAPI Service...")
    try:
        model_service.load_artifacts()
        print("Model and Tokenizer loaded successfully!")
    except Exception as e:
        print(f"[WARNING] Failed to load model at startup: {e}")
        print("Inference endpoints will return 503 until model is ready.")
    print("=" * 60)
    yield
    print("Shutting down Story Continuation Generator Service...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Locally trained LSTM Neural Language Model for Story Continuation",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount compiled static frontend and SPA fallback handler (must be last)
setup_static_serving(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
