from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .common.enums import ControllerAction, RouteType, SpecialistType
from .models.knowledge import KnowledgeRetrieveResponse
from .models.ollama import ModelGenerationResponse
from .models.vision import VisionAnalysis


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


class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class ControllerPlan(BaseModel):
    intent: str = ""
    summary: str = ""
    complexity: Literal["low", "medium", "high"] = "medium"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    requires_vision: bool = False
    requires_knowledge: bool = False
    requires_coder: bool = False
    requires_tools: bool = False
    requires_reasoning: bool = False
    requires_clarification: bool = False

    clarification_question: str | None = None
    tool_requests: list[ToolRequest] = Field(default_factory=list)
    execution_steps: list[SpecialistType] = Field(default_factory=list)

    route_hint: RouteDecision | None = None


class ControllerValidation(BaseModel):
    action: ControllerAction = ControllerAction.CONTINUE
    summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_reasoning: bool = False
    final_answer_ready: bool = False
    next_steps: list[SpecialistType] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    notes: str = ""


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
    route: RouteDecision | None = None
    controller_plan: ControllerPlan | None = None
    controller_validation: ControllerValidation | None = None
    answer: str
    used_models: list[str] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    knowledge_result: KnowledgeRetrieveResponse | None = None
    vision: VisionAnalysis | None = None
    vision_context: str = ""
    coder_result: CoderResult | None = None
    tool_result: ToolResult | None = None
    reasoning: ModelGenerationResponse | None = None
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
