from __future__ import annotations

from typing import Any, TypedDict

from ..models.knowledge import KnowledgeRetrieveResponse
from ..models.ollama import ModelGenerationResponse
from ..models.web import WebSearchResult
from ..schemas import (
    ControllerPlan,
    ControllerValidation,
    CoderResult,
    ExecutionPlan,
    NormalizedRequest,
    RouteDecision,
    ToolResult,
)


class OrchestratorState(TypedDict, total=False):
    thread_id: str
    request_id: str

    messages: list[dict[str, Any]]
    controller_messages: list[dict[str, Any]]
    original_messages: list[dict[str, Any]]
    metadata: dict[str, Any]
    normalized_request: dict[str, Any] | NormalizedRequest
    routing_hints: dict[str, Any]
    attachments: list[dict[str, Any]]

    user_text: str
    has_images: bool

    route: dict[str, Any] | RouteDecision
    controller_plan: dict[str, Any] | ControllerPlan
    controller_validation: dict[str, Any] | ControllerValidation
    execution_plan: dict[str, Any] | ExecutionPlan

    executed_specialists: list[str]
    pending_specialists: list[str]
    failed_specialists: list[str]
    retry_counts: dict[str, int]
    pending_steps: list[str]
    completed_steps: list[str]
    current_step: str
    needs_reasoning: bool
    requires_clarification: bool
    controller_cycles: int
    specialist_executions: int
    workflow_progress: int
    workflow_stall_count: int
    last_progress_signature: str
    last_controller_decision: str
    last_specialist: str
    validation_status: str
    retry_limit: int
    specialist_status: str
    clarification_question: str

    knowledge_result: KnowledgeRetrieveResponse | dict[str, Any] | None
    web_search_result: WebSearchResult | dict[str, Any] | None
    vision: dict[str, Any] | None
    vision_context: str
    coder_result: CoderResult | dict[str, Any] | None
    tool_result: ToolResult | dict[str, Any] | None
    reasoning_result: ModelGenerationResponse | dict[str, Any] | None

    used_models: list[str]
    used_tools: list[str]

    answer: str
    final_answer_ready: bool
    error: str
