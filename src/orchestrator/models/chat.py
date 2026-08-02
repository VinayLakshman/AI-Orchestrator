from typing import Any

from pydantic import BaseModel, Field
from pydantic import model_validator

from ..common.enums import ChatRole
from ..common.types import MessageContent

class ChatMessage(BaseModel):
    role: ChatRole
    content: MessageContent
    name: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    def _ensure_metadata_dict(cls, values: dict[str, Any]) -> dict[str, Any]:
        from ..serialization import canonicalize_metadata

        meta = values.get("metadata")
        values["metadata"] = canonicalize_metadata(meta)
        return values


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    thread_id: str | None = None
    model: str | None = None
    stream: bool = True
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    def _ensure_metadata_dict(cls, values: dict[str, Any]) -> dict[str, Any]:
        from ..serialization import canonicalize_metadata

        meta = values.get("metadata")
        values["metadata"] = canonicalize_metadata(meta)
        return values
