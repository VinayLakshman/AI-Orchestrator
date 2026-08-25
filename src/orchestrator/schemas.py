from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NormalizedAttachment(BaseModel):
    """Normalized attachment metadata without raw content.

    Stores only lightweight metadata (type, filename, size, reference)
    to avoid bloating the orchestration state with large file contents.
    The raw file content should only exist temporarily during preprocessing
    and must NOT be propagated through OrchestratorState or checkpoints.
    """
    attachment_type: str
    placeholder: str
    reference: str = ""
    filename: str = ""
    size_bytes: int = 0
    # NOTE: 'raw' field removed intentionally to prevent large file contents
    # from being stored in the orchestration state and LangGraph checkpoints.


class OpenAIMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]
    name: str | None = None
    tool_call_id: str | None = None


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OpenAIChatCompletionChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: str = "stop"


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatCompletionChoice]
    usage: OpenAIUsage = Field(default_factory=OpenAIUsage)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OpenAIModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class OpenAIModelListResponse(BaseModel):
    object: str = "list"
    data: list[OpenAIModelCard]
