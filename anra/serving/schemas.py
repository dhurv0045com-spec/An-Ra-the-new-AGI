"""Pydantic schemas for the AN-RA HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8192)
    max_new_tokens: int = Field(128, ge=1, le=2048)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=1, le=200)
    session_id: str | None = None
    strategy: str = "nucleus"


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    session_id: str | None = None
    reasoning_trace: list[str] = Field(default_factory=list)
    hal_state: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0


class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    created_at: float | None = None
    last_active: float | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    message: str
    docs: str | None = None
