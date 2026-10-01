"""Health check router for process liveness and model readiness."""

import datetime
from fastapi import APIRouter
from backend.app.schemas import HealthResponse
from backend.app.model_service import model_service
from backend.app.config import settings

health_router = APIRouter(tags=["Health"])

@health_router.get("/health", response_model=HealthResponse)
async def get_health():
    """Process health and readiness check."""
    return HealthResponse(
        status="healthy" if model_service.is_loaded else "degraded",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model_loaded=model_service.is_loaded,
        version=settings.VERSION
    )
