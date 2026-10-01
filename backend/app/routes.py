"""API router for model inference, metrics, and status endpoints."""

from fastapi import APIRouter, HTTPException, status
from backend.app.schemas import (
    GenerateRequest,
    GenerateResponse,
    GenerationSettings,
    CompareRequest,
    CompareResponse,
    CompareResultItem,
    ModelStatusResponse,
    ModelMetricsResponse,
)
from backend.app.model_service import model_service
from backend.app.config import settings

api_router = APIRouter(tags=["Generation & Model"])

@api_router.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Retrieve current model loaded status, architecture metadata, and vocabulary size."""
    info = model_service.get_status()
    return ModelStatusResponse(**info)

@api_router.get("/model/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics():
    """Retrieve actual training and evaluation metrics (loss, perplexity, accuracy)."""
    metrics_info = model_service.get_metrics()
    return ModelMetricsResponse(**metrics_info)

@api_router.post("/generate", response_model=GenerateResponse)
async def generate_continuation(request: GenerateRequest):
    """Generate story continuation using the trained local LSTM language model."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Please try again shortly."
        )

    try:
        continuation, runtime_ms = model_service.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            seed=request.seed
        )

        return GenerateResponse(
            prompt=request.prompt,
            generated_text=continuation,
            settings=GenerationSettings(
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                seed=request.seed
            ),
            runtime_ms=runtime_ms,
            model_version=settings.VERSION
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}"
        )

@api_router.post("/generate/compare", response_model=CompareResponse)
async def compare_generations(request: CompareRequest):
    """Generate continuations across multiple hyperparameter settings for side-by-side comparison."""
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready."
        )

    settings_to_run = request.settings_list
    if not settings_to_run:
        settings_to_run = [
            GenerationSettings(max_new_tokens=request.max_new_tokens, temperature=0.3, top_k=20, top_p=0.85),
            GenerationSettings(max_new_tokens=request.max_new_tokens, temperature=0.8, top_k=40, top_p=0.90),
            GenerationSettings(max_new_tokens=request.max_new_tokens, temperature=1.4, top_k=60, top_p=0.95),
        ]

    results = []
    for idx, s in enumerate(settings_to_run):
        name = f"Preset {idx + 1} (T={s.temperature}, TopK={s.top_k})"
        text, ms = model_service.generate(
            prompt=request.prompt,
            max_new_tokens=s.max_new_tokens,
            temperature=s.temperature,
            top_k=s.top_k,
            top_p=s.top_p,
            seed=s.seed
        )
        results.append(
            CompareResultItem(
                name=name,
                settings=s,
                generated_text=text,
                runtime_ms=ms
            )
        )

    return CompareResponse(prompt=request.prompt, results=results)
