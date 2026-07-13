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


def extract_assistant_text(value: Any) -> str:
    """
    Normalize a model or API payload down to the assistant's final text.

    This intentionally ignores role/thinking/tool metadata and only preserves
    the textual answer a user should see.
    """
    if value is None:
        return ""

    if isinstance(value, ModelGenerationResponse):
        return extract_assistant_text(value.content)

    if isinstance(value, BaseModel):
        try:
            return extract_assistant_text(value.model_dump(exclude_none=True))
        except Exception:
            return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        parts = [extract_assistant_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()

    if isinstance(value, dict):
        for key in ("content", "response", "text"):
            if key in value:
                extracted = extract_assistant_text(value.get(key))
                if extracted:
                    return extracted

        if "message" in value:
            extracted = extract_assistant_text(value.get("message"))
            if extracted:
                return extracted

        if "choices" in value:
            extracted = extract_assistant_text(value.get("choices"))
            if extracted:
                return extracted

        if "messages" in value:
            extracted = extract_assistant_text(value.get("messages"))
            if extracted:
                return extracted

        return ""

    return ""


def normalize_generation_response(model: str, value: Any) -> ModelGenerationResponse:
    raw: dict[str, Any] = {}
    if isinstance(value, dict):
        raw = value
    elif isinstance(value, BaseModel):
        try:
            dumped = value.model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                raw = dumped
        except Exception:
            raw = {}

    return ModelGenerationResponse(
        model=model,
        content=extract_assistant_text(value),
        raw=raw,
    )
