from __future__ import annotations

from typing import Any

from ..models.state import OrchestratorState
from ..schemas import OrchestratorResponse


def build_generation_response(
    *,
    thread_id: str,
    state: OrchestratorState,
    metadata: dict[str, Any] | None = None,
) -> OrchestratorResponse:
    """
    Builds the API response from the canonical orchestration state.

    This is the only place that translates internal orchestration
    models into the public API schema.
    """

    return OrchestratorResponse(
        thread_id=thread_id,

        answer=state.response.final_response,

        execution=state.execution,

        evidence=state.evidence,

        response=state.response,

        debug=state.debug,

        used_models=state.debug.used_models,

        used_tools=state.debug.used_tools,

        metadata={
            **state.response.metadata,
            **(metadata or {}),
        },
    )


def build_failure_response(
    *,
    thread_id: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> OrchestratorResponse:
    """
    Builds a failure response without requiring a complete
    orchestration state.
    """

    state = OrchestratorState()

    state.request.thread_id = thread_id

    state.response.final_response = message

    if metadata:
        state.response.metadata.update(metadata)

    return build_generation_response(
        thread_id=thread_id,
        state=state,
    )
