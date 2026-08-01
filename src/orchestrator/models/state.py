from __future__ import annotations

from datetime import datetime
from typing import Any

from orchestrator.models.chat import ChatMessage
from pydantic import BaseModel, Field

from orchestrator.models.evidence import EvidenceLedger
from orchestrator.models.execution import ExecutionState


class RequestState(BaseModel):
    """
    Normalized request information.

    Populated once by the request normalizer/router.
    Never modified afterwards.
    """

    request_id: str = ""

    conversation_id: str = ""

    thread_id: str = ""

    model: str = ""

    stream: bool = False

    messages: list[ChatMessage] = Field(default_factory=list)

    user_message: str = ""

    original_query: str = ""

    resolved_query: str = ""

    is_followup: bool = False

    followup_confidence: float = 0.0

    images: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    received_at: datetime = Field(default_factory=datetime.utcnow)


class ResponseState(BaseModel):
    """
    Output generated during orchestration.

    Only the finalizer writes final_response.
    """

    final_response: str = ""

    finish_reason: str = "stop"

    usage: dict[str, Any] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(default_factory=dict)


class DebugState(BaseModel):
    """
    Internal debugging information.

    Never shown to the user.

    Safe to log.

    Useful for tracing controller decisions.
    """

    planner_prompt: str = ""

    planner_response: dict[str, Any] = Field(default_factory=dict)

    validator_prompt: str = ""

    validator_response: dict[str, Any] = Field(default_factory=dict)

    used_models: list[str] = Field(default_factory=list)

    used_tools: list[str] = Field(default_factory=list)

    timings: dict[str, float] = Field(default_factory=dict)

    execution_trace: list[dict[str, Any]] = Field(default_factory=list)


class OrchestratorState(BaseModel):
    """
    Canonical orchestration state.

    This object is passed through every LangGraph node.

    Nodes should ONLY modify the section they own.

    request
        Immutable.

    execution
        Planner + execution engine.

    evidence
        Specialists.

    response
        Finalizer.

    debug
        Internal logging/tracing.
    """

    request: RequestState = Field(default_factory=RequestState)

    execution: ExecutionState = Field(default_factory=ExecutionState)

    evidence: EvidenceLedger = Field(default_factory=EvidenceLedger)

    response: ResponseState = Field(default_factory=ResponseState)

    debug: DebugState = Field(default_factory=DebugState)

    def reset_response(self) -> None:
        """
        Clears response state before a new execution.
        """

        self.response = ResponseState()

    def reset_evidence(self) -> None:
        """
        Clears accumulated evidence.
        """

        self.evidence = EvidenceLedger()

    def initialize_execution(self) -> None:
        """
        Initializes runtime execution from the immutable plan.
        """

        self.execution.initialize()

    @property
    def current_specialist(self):
        return self.execution.current_specialist

    @property
    def finished(self) -> bool:
        return self.execution.finished
