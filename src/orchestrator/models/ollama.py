from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelGenerationResponse(BaseModel):
    model: str
    content: str
    raw: dict[str, Any] = Field(default_factory=dict)


class OllamaStreamChunk(BaseModel):
    content: str
    done: bool
    raw: dict[str, Any] = Field(default_factory=dict)
