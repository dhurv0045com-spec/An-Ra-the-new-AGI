"""Tests for anra/serving/schemas.py — Pydantic request/response models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from anra.serving.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    SessionInfo,
)


def test_generate_request_defaults():
    r = GenerateRequest(prompt="hello")
    assert r.max_new_tokens == 128
    assert r.temperature == 0.8
    assert r.top_k == 50
    assert r.session_id is None
    assert r.strategy == "nucleus"


def test_generate_request_rejects_empty_prompt():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="")


def test_generate_request_rejects_prompt_too_long():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="x" * 8193)


def test_generate_request_rejects_invalid_temperature():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hi", temperature=3.0)


def test_generate_response_roundtrip():
    r = GenerateResponse(text="hello world", tokens_generated=2)
    assert r.text == "hello world"
    assert r.tokens_generated == 2
    assert r.reasoning_trace == []
    assert r.hal_state == {}
    assert r.latency_ms == 0.0
    data = r.model_dump()
    r2 = GenerateResponse(**data)
    assert r2.text == r.text


def test_health_response_fields():
    h = HealthResponse(status="ok", model_loaded=True, version="0.3.0", uptime_seconds=42.0)
    assert h.status == "ok"
    assert h.model_loaded is True


def test_error_response_optional_docs():
    e = ErrorResponse(error="not_found", message="Resource missing")
    assert e.docs is None
    e2 = ErrorResponse(
        error="not_found",
        message="Resource missing",
        docs="https://docs.example.com",
    )
    assert e2.docs is not None


def test_session_info_fields():
    s = SessionInfo(session_id="abc123", message_count=5)
    assert s.session_id == "abc123"
    assert s.message_count == 5
    assert s.created_at is None
