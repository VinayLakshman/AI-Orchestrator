from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .vision.models import VisionAnalysis


class RouteType(str, Enum):
    GENERAL = "general"
    VISION = "vision"
    CODE = "code"
    RAG = "rag"
    TOOLS = "tools"
    MULTI_STEP = "multi_step"
    CLARIFY = "clarify"


class ToolType(str, Enum):
    MCP = "mcp"
    KNOWLEDGE = "knowledge"
    OLLAMA = "ollama"


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str | list[dict[str, Any]]
    name: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    thread_id: str | None = None
    model: str | None = None
    stream: bool = True
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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


class KnowledgeRetrieveRequest(BaseModel):
    question: str
    top_k: int = 6
    candidate_limit: int = 12
    neighbor_window: int = 1


class KnowledgeHit(BaseModel):
    repository: str
    branch: str
    commit: str
    path: str
    language: str
    chunk_index: int
    chunk_count: int
    score: float
    content: str


class KnowledgeRetrieveResponse(BaseModel):
    question: str
    intent: str
    embedding_time: float
    search_time: float
    rerank_time: float
    expansion_time: float
    total_time: float
    context: str | None = None
    grounded: bool = False
    confidence: float = 0.0
    retrieval_reason: str = ""
    primary_hits: list[KnowledgeHit] = Field(default_factory=list)
    expanded_hits: list[KnowledgeHit] = Field(default_factory=list)


class ModelGenerationRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    options: dict[str, Any] = Field(default_factory=dict)


class ModelGenerationResponse(BaseModel):
    model: str
    content: str
    raw: dict[str, Any] = Field(default_factory=dict)


class OrchestratorResponse(BaseModel):
    thread_id: str
    route: RouteDecision
    answer: str
    used_models: list[str] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    knowledge: list[KnowledgeHit] = Field(default_factory=list)
    vision: VisionAnalysis | None = None
    vision_context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)