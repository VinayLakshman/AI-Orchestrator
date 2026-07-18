from __future__ import annotations

from typing import Any

from orchestrator.models.evidence import EvidenceLedger
from orchestrator.models.execution import ExecutionState
from orchestrator.models.state import DebugState, ResponseState
from pydantic import BaseModel, Field

from .common.enums import RouteType

class RouteDecision(BaseModel):
    route: RouteType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    needs_vision: bool = False
    needs_rag: bool = False
    needs_tools: bool = False
    needs_code: bool = False
    needs_planning: bool = False
    candidate_models: list[str] = Field(default_factory=list)


class NormalizedAttachment(BaseModel):
    attachment_type: str
    placeholder: str
    raw: dict[str, Any] = Field(default_factory=dict)


class RoutingHints(BaseModel):
    repository_likelihood: float = Field(default=0.0, ge=0.0, le=1.0)
    code_likelihood: float = Field(default=0.0, ge=0.0, le=1.0)
    vision_likelihood: float = Field(default=0.0, ge=0.0, le=1.0)


class CoderResult(BaseModel):
    task: str = ""
    summary: str = ""
    code: str = ""
    files: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    raw_text: str = ""


class ToolResult(BaseModel):
    tool_name: str = ""
    status: str = "ok"
    summary: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    raw_text: str = ""


class OrchestratorResponse(BaseModel):
    thread_id: str

    answer: str

    execution: ExecutionState

    evidence: EvidenceLedger

    response: ResponseState

    debug: DebugState | None = None

    used_models: list[str] = Field(default_factory=list)

    used_tools: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


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
