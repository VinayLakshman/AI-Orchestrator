from __future__ import annotations

from typing import Any

from ..schemas import (
    CoderResult,
    ControllerPlan,
    ControllerValidation,
    KnowledgeRetrieveResponse,
    ModelGenerationResponse,
    OrchestratorResponse,
    RouteDecision,
    ToolResult,
)


def build_generation_response(
    *,
    thread_id: str,
    answer: str,
    route: RouteDecision | None = None,
    controller_plan: ControllerPlan | None = None,
    controller_validation: ControllerValidation | None = None,
    knowledge_result: KnowledgeRetrieveResponse | None = None,
    coder_result: CoderResult | None = None,
    tool_result: ToolResult | None = None,
    reasoning: ModelGenerationResponse | None = None,
    used_models: list[str] | None = None,
    used_tools: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> OrchestratorResponse:
    return OrchestratorResponse(
        thread_id=thread_id,
        route=route,
        controller_plan=controller_plan,
        controller_validation=controller_validation,
        answer=answer,
        used_models=used_models or [],
        used_tools=used_tools or [],
        knowledge_result=knowledge_result,
        coder_result=coder_result,
        tool_result=tool_result,
        reasoning=reasoning,
        metadata=metadata or {},
    )


def build_retrieval_failure_response(
    *,
    thread_id: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> OrchestratorResponse:
    return OrchestratorResponse(
        thread_id=thread_id,
        answer=message,
        metadata=metadata or {},
    )