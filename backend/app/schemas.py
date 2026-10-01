"""Pydantic schemas for request validation and response serialization."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="Seed prompt for story continuation")
    max_new_tokens: int = Field(default=50, ge=1, le=200, description="Number of tokens to generate")
    temperature: float = Field(default=0.8, gt=0.0, le=2.5, description="Sampling temperature")
    top_k: int = Field(default=40, ge=0, le=200, description="Top-k sampling threshold")
    top_p: Optional[float] = Field(default=0.90, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    seed: Optional[int] = Field(default=None, description="Optional random seed for reproducible sampling")

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Prompt cannot be empty or whitespace only")
        return trimmed

class GenerationSettings(BaseModel):
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: Optional[float] = None
    seed: Optional[int] = None

class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    settings: GenerationSettings
    runtime_ms: int
    model_version: str

class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=40, ge=1, le=100)
    settings_list: List[GenerationSettings] = Field(default_factory=list)

class CompareResultItem(BaseModel):
    name: str
    settings: GenerationSettings
    generated_text: str
    runtime_ms: int

class CompareResponse(BaseModel):
    prompt: str
    results: List[CompareResultItem]

class ModelStatusResponse(BaseModel):
    status: str
    model_name: str
    version: str
    vocab_size: int
    device: str
    parameters_count: Optional[int] = None

class ModelMetricsResponse(BaseModel):
    status: str
    metrics: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    version: str
