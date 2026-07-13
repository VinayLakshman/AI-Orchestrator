from __future__ import annotations

from typing import Any, TypedDict

from ..models.knowledge import KnowledgeRetrieveResponse
from ..models.ollama import ModelGenerationResponse
from ..schemas import ControllerPlan, ControllerValidation, CoderResult, RouteDecision, ToolResult


class OrchestratorState(TypedDict, total=False):
    thread_id: str
    request_id: str

    messages: list[dict[str, Any]]
    metadata: dict[str, Any]

    user_text: str
    has_images: bool

    route: dict[str, Any] | RouteDecision
    controller_plan: dict[str, Any] | ControllerPlan
    controller_validation: dict[str, Any] | ControllerValidation

    pending_steps: list[str]
    completed_steps: list[str]
    current_step: str
    needs_reasoning: bool
    requires_clarification: bool
    clarification_question: str

    knowledge_result: KnowledgeRetrieveResponse | dict[str, Any] | None
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
